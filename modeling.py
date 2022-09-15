##########################################################################################
#
#
#
#
#
#
#
#
##########################################################################################

import argparse
import os
import pickle
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

CONFIG = {'stations': ['VIC', 'NEW', 'OTT', 'STJ', 'ESK', 'LER', 'WNG', 'NGK', 'BFE'],
			'thresholds': [7.15],	# list of thresholds to be examined.
			'params': ['Date_UTC', 'N', 'E', 'sinMLT', 'cosMLT', 'B_Total', 'BY_GSM',
	   					'BZ_GSM', 'Vx', 'Vy', 'Vz', 'proton_density', 'T',
	   					 'AE_INDEX', 'SZA', 'dBHt', 'B'],								# List of parameters that will be used for training.
	   																							# Date_UTC will be removed, kept here for resons that will be evident below
			'test_storm_stime': ['2001-03-29 09:59:00', '2001-08-29 21:59:00', '2005-05-13 21:59:00',
								 '2005-08-30 07:59:00', '2006-12-13 09:59:00', '2010-04-03 21:59:00',
								 '2011-08-04 06:59:00', '2015-03-15 23:59:00'],						# These are the start times for testing storms
			'test_storm_etime': ['2001-04-02 12:00:00', '2001-09-02 00:00:00', '2005-05-17 00:00:00',
									'2005-09-02 12:00:00', '2006-12-17 00:00:00', '2010-04-07 00:00:00',
									'2011-08-07 09:00:00', '2015-03-19 14:00:00'],	# end times for testing storms. This will remove them from training
			'plot_stime': ['2011-08-05 16:00', '2006-12-14 12:00', '2001-03-30 21:00'],		# start times for the plotting widow. Focuses on the main sequence of the storm
			'plot_etime': ['2011-08-06 18:00', '2006-12-15 20:00', '2001-04-01 02:00'],		# end plotting times
			'plot_titles': ['2011_storm', '2006_storm', '2001_storm'],						# list used for plot titles so I don't have to do it manually
			'forecast': 30,
			'window': 30,																	# time window over which the metrics will be calculated
			'k_fold_splits': 100,													# amount of k fold splits to be performed. Program will create this many models
			'lead': 12,																# lead time added to each storm minimum in SYM-H
			'recovery':24,
			'random_seed':42}															# recovery time added to each storm minimum in SYM-H

MODEL_CONFIG = {'version':0,
					'time_history': 60, 	# How much time history the model will use, defines the 2nd dimension of the model input array
					'epochs': 100, 		# Maximum amount of empoch the model will run if not killed by early stopping
					'layers': 1, 		# How many CNN layers the model will have.
					'filters': 64, 		# Number of filters in the first CNN layer. Will decrease by half for any subsequent layers if "layers">1
					'dropout': 0.2, 		# Dropout rate for the layers
					'loss':'mse',
					'learning_rate': 1e-5,		# Learning rate, used as the inital learning rate if a learning rate decay function is used
					'lr_decay_steps':230,				# If a learning ray decay funtion is used, dictates the number of decay steps
					'early_stop_patience':5}

# setting the random seeds for reproducibility
random.seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])

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

	with open('../data/prepared_data/{0}_train_dict.pkl'.format(station), 'rb') as train:
		train_dict = pickle.load(train)
	with open('../data/prepared_data/{0}_test_dict.pkl'.format(station), 'rb') as test:
		test_dict = pickle.load(test)
	train_indicies = pd.read_feather('../data/prepared_data/{0}_train_indicies.feather'.format(station))
	val_indicies = pd.read_feather('../data/prepared_data/{0}_val_indicies.feather'.format(station))

	return train_dict, test_dict, train_indicies, val_indicies


def create_CNN_model(n_features, loss='categorical_crossentropy', early_stop_patience=3):
	'''Initializing our model
		Inputs:
		model_config: predefined model configuration dictonary
		n_features: amount of input features into the model
		loss: loss function to be uesd for training
		early_stop_patience: amount of epochs the model will continue training once there is no longer val loss improvements'''


	model = Sequential()						# initalizing the model

	model.add(Conv2D(MODEL_CONFIG['filters'], 4, padding='same',
									activation='relu', input_shape=(MODEL_CONFIG['time_history'], n_features, 1)))			# adding the CNN layer
	model.add(MaxPooling2D())						# maxpooling layer reduces the demensions of the training data. Speeds up models and improves results
	model.add(Conv2D(MODEL_CONFIG['filters']*2, 3, padding='same', activation='relu'))
	model.add(Conv2D(MODEL_CONFIG['filters']*2, 3, padding='same', activation='relu'))
	model.add(MaxPooling2D())
	model.add(Conv2D(MODEL_CONFIG['filters']*4, 2, padding='same', activation='relu'))
	model.add(Conv2D(MODEL_CONFIG['filters']*4, 2, padding='same', activation='relu'))
	model.add(MaxPooling2D())
	model.add(Flatten())							# changes dimensions of model. Not sure exactly how this works yet but improves results
	model.add(Dense(MODEL_CONFIG['filters']*2, activation='relu'))		# Adding dense layers with dropout in between
	model.add(Dropout(0.2))
	model.add(Dense(MODEL_CONFIG['filters'], activation='relu'))
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

		if not os.path.exists('models/{0}'.format(station)):
			os.makedirs('models/{0}'.format(station))

		model.save('models/{0}/CNN_version_{1}_split_{2}.h5'.format(station, MODEL_CONFIG['version'], split))

	if not first_time:

		model = load_model('models/{0}/CNN_version_{1}_split_{2}.h5'.format(station, MODEL_CONFIG['version'], split))						# loading the models if already trained

	return model


def making_predictions(model, test_dict, split):
	'''function used to make the predictions with the testing data
		Inputs:
		model: pre-trained model
		test_dict: dictonary with the testing model inputs and the real data for comparison
		thresholds: which threshold is being examined
		split: which split is being tested'''

	for key in test_dict:									# looping through the sub dictonaries for each storm

		Xtest = test_dict[key]['Y']							# defining the testing inputs
		Xtest = Xtest.reshape((Xtest.shape[0], Xtest.shape[1], Xtest.shape[2], 1))				# reshpaing for one channel input

		predicted = model.predict(Xtest, verbose=1)						# predicting on the testing input data

		predicted = tf.gather(predicted, [1], axis=1)					# grabbing the positive node
		predicted = predicted.numpy()									# turning to a numpy array
		predicted = pd.Series(predicted.reshape(len(predicted),))		# and then into a pd.series

		df = test_dict[key]['real_df']									# calling the correct dataframe
		df['predicted_split_{0}'.format(split)] = predicted		# and storing the results
		re = df['crossing']
		print('RMSE: '+str(np.sqrt(mean_squared_error(re,predicted))))

	return test_dict


def main(station):

	train_dict, test_dict, train_indicies, val_indicies = loading_data_and_indicies(station)

	for split in range(CONFIG['splits']):		# this is the bulk of the K fold. We loop through the list of indexes and train on the different train-val indices

		train_index = train_indicies['split_{0}'.format(split)].to_numpy()
		val_index = val_indicies['split_{0}'.format(split)].to_numpy()

		print('Split: '+ str(split))
		tf.keras.backend.clear_session() 				# clearing the information from any old models so we can run clean new ones.
		MODEL, early_stop = create_CNN_model(n_features=train_dict['X'].shape[2], loss='categorical_crossentropy', early_stop_patience=5)					# creating the model
		print(MODEL.summary())

		# pulling the data and catagorizing it into the train-val pairs
		xtrain = train_dict['X'][train_index]
		array_sum = np.sum(xtrain)
		print(np.isnan(array_sum))
		xval =  train_dict['X'][val_index]
		array_sum = np.sum(xval)
		print(np.isnan(array_sum))
		ytrain = train_dict['crossing'][train_index]
		array_sum = np.sum(ytrain)
		print(np.isnan(array_sum))
		yval = train_dict['crossing'][val_index]
		array_sum = np.sum(yval)
		print(np.isnan(array_sum))

		model = fit_CNN(MODEL, xtrain, xval, ytrain, yval, early_stop, split, station, first_time=True)			# does the model fit!

		test_dict = making_predictions(model, test_dict, split)					# defines the test dictonary for storing results


	# for each storm we set the datetime index and saves the data in a CSV file
	for i in range(len(test_dict)):
		real_df = test_dict['storm_{0}'.format(i)]['real_df']
		pd.to_datetime(real_df['Date_UTC'], format='%Y-%m-%d %H:%M:%S')
		real_df.reset_index(drop=True, inplace=True)
		real_df.set_index('Date_UTC', inplace=True, drop=False)
		real_df.index = pd.to_datetime(real_df.index)
		real_df.to_feather('outputs/{0}/version_{1}_storm_{2}.feather'.format(station, MODEL_CONFIG['version'], i))


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
