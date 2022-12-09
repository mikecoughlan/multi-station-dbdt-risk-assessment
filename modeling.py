##########################################################################################
#
#	multi-station-dbdt-risk-assessment/modeling.py
#
#	Script for defining the neural network, fitting the model, and making a prediction
#	on the testing data. Fits models for the 100 unique train-val splits and makes
# 	100 corresponding predictions for each testing storm. Saves the results to a
# 	csv/feather file used for analyzing results in the analyzing_results.py script.
# 	Uses an argparser for choosing the magnetometer station the models will be created
# 	for.
#
#
##########################################################################################

import argparse
import gc
import json
import os
import pickle
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.python.keras.backend import get_session

# stops this program from hogging the GPU
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


old_params = ['Date_UTC', 'N', 'E', 'sinMLT', 'cosMLT', 'B_Total', 'BY_GSM',
	   					'BZ_GSM', 'Vx', 'Vy', 'Vz', 'proton_density', 'T',
	   					 'AE_INDEX', 'SZA', 'dBHt', 'B']

# loading config and specific model config files. Using them as dictonaries
with open('config.json', 'r') as con:
	CONFIG = json.load(con)

with open('model_config.json', 'r') as mcon:
	MODEL_CONFIG = json.load(mcon)


# setting the random seeds for reproducibility
random.seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])

# Reset Keras Session
def reset_keras(model):
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try:
        del model # this is from global space - change this as you need
    except:
        pass

    print(gc.collect()) # if it's done something you should see a number being outputted



def loading_data_and_indicies(station):
	'''
	Loading the preprepared traing and testing dictonaries for the station being examined.

	Args:
		station (str): 3 diget code indicating the data from which supermag station is being loaded

	Returns:
		train_dict, test_dict: dictonaries containing the training and target arrays (X, y)
		and testing data (X, y) for each of the testing storms

		train_indicies, val_indicies: dataframes containing the indicies for doing the boothstrapping
	'''

	with open('../data/prepared_data/SW_only_{0}_train_dict.pkl'.format(station), 'rb') as train:
		train_dict = pickle.load(train)
	with open('../data/prepared_data/SW_only_{0}_test_dict.pkl'.format(station), 'rb') as test:
		test_dict = pickle.load(test)
	train_indicies = pd.read_feather('../data/prepared_data/SW_only_{0}_train_indicies.feather'.format(station))
	val_indicies = pd.read_feather('../data/prepared_data/SW_only_{0}_val_indicies.feather'.format(station))

	return train_dict, test_dict, train_indicies, val_indicies


def create_CNN_model(n_features, loss='categorical_crossentropy', early_stop_patience=3):
	'''
	Initializing our model

	Args:
		n_features (int): number of input features into the model
		loss (str, optional): loss function to be uesd for training. Defaults to 'categorical_crossentropy'.
		early_stop_patience (int, optional): number of epochs the model will continue training once there
												is no longer val loss improvements. Defaults to 3.

	Returns:
		object: model configuration ready for training
		object: early stopping conditions
	'''


	model = Sequential()						# initalizing the model

	model.add(Conv2D(MODEL_CONFIG['filters'], 2, padding='same',
									activation='relu', input_shape=(MODEL_CONFIG['time_history'], n_features, 1)))			# adding the CNN layer
	model.add(MaxPooling2D())
	model.add(Flatten())							# changes dimensions of model. Not sure exactly how this works yet but improves results
	model.add(Dense(MODEL_CONFIG['filters'], activation='relu'))		# Adding dense layers with dropout in between
	model.add(Dropout(0.2))
	model.add(Dense(MODEL_CONFIG['filters']//2, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(2, activation='softmax'))
	opt = tf.keras.optimizers.Adam(learning_rate=MODEL_CONFIG['learning_rate'])		# learning rate that actually started producing good results
	model.compile(optimizer=opt, loss=loss, metrics = ['accuracy'])					# Ive read that cross entropy is good for this type of model
	early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=early_stop_patience)		# early stop process prevents overfitting


	return model, early_stop


def fit_CNN(model, xtrain, xval, ytrain, yval, early_stop, split, station, first_time=True):
	'''
	Performs the actual fitting of the model.

	Args:
		model (keras model): model as defined in the create_model function.
		xtrain (3D np.array): training data inputs
		xval (3D np.array): validation inputs
		ytrain (2D np.array): training target vectors
		yval (2D np.array): validation target vectors
		early_stop (keras early stopping dict): predefined early stopping function
		split (int): split being trained. Used for saving model.
		station (str): station being trained.
		first_time (bool, optional): if True model will be trainined, False model will be loaded. Defaults to True.

	Returns:
		model: fit model ready for making predictions.
	'''

	if first_time:

		# reshaping the model input vectors for a single channel
		Xtrain = xtrain.reshape((xtrain.shape[0], xtrain.shape[1], xtrain.shape[2], 1))
		Xval = xval.reshape((xval.shape[0], xval.shape[1], xval.shape[2], 1))

		model.fit(Xtrain, ytrain, validation_data=(Xval, yval),
					verbose=1, shuffle=True, epochs=MODEL_CONFIG['epochs'], callbacks=[early_stop])			# doing the training! Yay!

		# checking that the model directory exists. Creates it if not.
		if not os.path.exists('models/{0}'.format(station)):
			os.makedirs('models/{0}'.format(station))

		# saving the model
		model.save('models/{0}/CNN_SW_only_split_{1}.h5'.format(station, split))

	if not first_time:

		# loading the model if it has already been trained.
		model = load_model('models/{0}/CNN_SW_only_split_{1}.h5'.format(station, split))				# loading the models if already trained

	return model


def making_predictions(model, test_dict, split):
	'''
	Function using the trained models to make predictions with the testing data.

	Args:
		model (object): pre-trained model
		test_dict (dict): dictonary with the testing model inputs and the real data for comparison
		split (int): which split is being tested

	Returns:
		dict: test dict now containing columns in the dataframe with the model predictions for this split
	'''

	# looping through the sub dictonaries for each storm
	for key in test_dict:

		Xtest = test_dict[key]['Y']							# defining the testing inputs
		Xtest = Xtest.reshape((Xtest.shape[0], Xtest.shape[1], Xtest.shape[2], 1))			# reshpaing for one channel input
		print('Test input Nans: '+str(np.isnan(Xtest).sum()))

		predicted = model.predict(Xtest, verbose=1)						# predicting on the testing input data

		predicted = tf.gather(predicted, [1], axis=1)					# grabbing the positive node
		predicted = predicted.numpy()									# turning to a numpy array
		predicted = pd.Series(predicted.reshape(len(predicted),))		# and then into a pd.series

		df = test_dict[key]['real_df']									# calling the correct dataframe
		df['predicted_split_{0}'.format(split)] = predicted		# and storing the results
		re = df['crossing']

		# checking for nan data in the results
		print('Pred has Nan: '+str(predicted.isnull().sum()))
		print('Real has Nan: '+str(re.isnull().sum()))

	return test_dict


def main(station):
	'''
	Pulls all the above functions together. Loops through the number of splits to create, fit ,
	and predict with a unique model for each train-val split. Outputs a saved file with the results.

	Args:
		station (str): 3 diget code for the station being examined. Passed to the script via arg parsing.
	'''

	# loading all data and indicies
	train_dict, test_dict, train_indicies, val_indicies = loading_data_and_indicies(station)

	# this is the bulk of the shuffeled k-fold splitting. We loop through the list of indexes and train on the different train-val indices
	for split in range(MODEL_CONFIG['splits']):

		train_index = train_indicies['split_{0}'.format(split)].to_numpy()
		val_index = val_indicies['split_{0}'.format(split)].to_numpy()

		print('Split: '+ str(split))

		# clearing the information from any old models so we can run clean new ones.
		if 'MODEL' in locals():
			reset_keras(MODEL)
		MODEL, early_stop = create_CNN_model(n_features=train_dict['X'].shape[2], loss='categorical_crossentropy', early_stop_patience=5)					# creating the model

		# pulling the data and catagorizing it into the train-val pairs
		xtrain = train_dict['X'][train_index]
		xval =  train_dict['X'][val_index]
		ytrain = train_dict['crossing'][train_index]
		yval = train_dict['crossing'][val_index]

		# if the saved model already exists, loads the pre-fit model
		if os.path.exists('models/{0}/CNN_SW_only_split_{1}.h5'.format(station, split)):
			model = fit_CNN(MODEL, xtrain, xval, ytrain, yval, early_stop, split, station, first_time=False)

		# if model has not been fit, fits the model
		else:
			model = fit_CNN(MODEL, xtrain, xval, ytrain, yval, early_stop, split, station, first_time=True)

		test_dict = making_predictions(model, test_dict, split)					# defines the test dictonary for storing results


	# for each storm we set the datetime index and saves the data in a CSV/feather file
	for i in range(len(test_dict)):
		real_df = test_dict['storm_{0}'.format(i)]['real_df']
		pd.to_datetime(real_df['Date_UTC'], format='%Y-%m-%d %H:%M:%S')
		real_df.reset_index(drop=True, inplace=True)

		if not os.path.exists('outputs/{0}'.format(station)):
			os.makedirs('outputs/{0}'.format(station))

		real_df.to_feather('outputs/{0}/SW_only_storm_{1}.feather'.format(station, i))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('station',
						action='store',
						choices=['OTT', 'STJ', 'VIC', 'NEW', 'ESK', 'WNG', 'LER', 'BFE', 'NGK'],
						type=str,
						help='input station code for the SuperMAG station to be examined.')

	args=parser.parse_args()

	main(args.station)

	print('It ran. God job!')
