##########################################################################################
#
#	multi-station-dbdt-risk-assessment/CNN_cross_val.py
#
#	Performs a cross validation for the CNN models for hyperparameter turning.
#	Displays the value for several metric scores for anaylsis.
#
##########################################################################################

import gc
import random
from datetime import datetime

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import auc, mean_squared_error, precision_recall_curve
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.backend import clear_session, get_session

# Data directories
projectDir = '~/projects/ml_helio_paper/'
pd.options.mode.chained_assignment = None # muting an irrelevent warning

# stops this program from hogging the GPU
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

CONFIG = {'stations': ['OTT'],
			'thresholds': 0.99,	# list of thresholds to be examined.
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
			'forecast': 30,
			'window': 30,																	# time window over which the metrics will be calculated
			'k_fold_splits': 1,													# amount of k fold splits to be performed. Program will create this many models
			'lead': 12,																# lead time added to each storm minimum in SYM-H
			'recovery':24,
			'random_seed':42,															# recovery time added to each storm minimum in SYM-H
			'cv_splits':3}

MODEL_CONFIG = {'time_history': 30, 	# How much time history the model will use, defines the 2nd dimension of the model input array
					'epochs': 100, 		# Maximum amount of empoch the model will run if not killed by early stopping
					'layers': 1, 		# How many CNN layers the model will have.
					'filters': 128, 		# Number of filters in the first CNN layer. Will decrease by half for any subsequent layers if "layers">1
					'dropout': 0.2, 		# Dropout rate for the layers
					'loss':'mse',
					'learning_rate': 1e-6,		# Learning rate, used as the inital learning rate if a learning rate decay function is used
					'lr_decay_steps':230,				# If a learning ray decay funtion is used, dictates the number of decay steps
					'early_stop_patience':5}

# setting the random seeds for reproducibility
random.seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])

# # Reset Keras Session
# def reset_keras(model):
#     sess = get_session()
#     clear_session()
#     sess.close()
#     sess = get_session()

#     try:
#         del model # this is from global space - change this as you need
#     except:
#         pass

#     print(gc.collect()) # if it's done something you should see a number being outputted


def classification_column(df, param, thresh, forecast, window):
	'''creating a new column which labels whether there will be a dBT that crosses the threshold in the forecast window.
		Inputs:
		df: the dataframe containing all of the relevent data.
		param: the paramaeter that is being examined for threshold crossings (dBHt for this study).
		thresholds: threshold or list of thresholds to define parameter crossing.
		forecast: how far out ahead we begin looking in minutes for threshold crossings. If forecast=30, will begin looking 30 minutes ahead.
		window: time frame in which we look for a threshold crossing starting at t=forecast. If forecast=30, window=30, we look for threshold crossings from t+30 to t+60'''

	predicting = forecast+window																# defining the end point of the prediction window
	df['shifted_{0}'.format(param)] = df[param].shift(-forecast)								## creates a new column that is the shifted parameter. Because time moves foreward with increasing
																									## index, the shift time is the negative of the forecast instead of positive.
	indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window)						# Yeah this is annoying, have to create a forward rolling indexer because it won't do it automatically.
	df['window_max'] = df.shifted_dBHt.rolling(indexer, min_periods=1).max()					# creates new coluimn in the df labeling the maximum parameter value in the forecast:forecast+window time frame
	df.reset_index(drop=True, inplace=True)														# just resets the index


	'''This section creates a binary column for each of the thresholds. Binary will be one if the parameter
		goes above the given threshold, and zero if it does not.'''

	conditions = [(df['window_max'] < thresh), (df['window_max'] >= thresh)]			# defining the conditions

	binary = [0, 1] 																	# 0 if not cross 1 if cross

	df['crossing'] = np.select(conditions, binary)						# new column created using the conditions and the binary


	df.drop(['window_max', 'shifted_dBHt'], axis=1, inplace=True)							# removes the two working columns for memory purposes

	return df


def ace_prep(path):

	'''Preparing the omnidata for plotting.
		Inputs:
		path: path to project directory
	'''
	df = pd.read_feather('../data/SW/solarwind_and_indicies.feather') 		# loading the omni data

	df.reset_index(drop=True, inplace=True) 		# reseting the index so its easier to work with integer indexes

	# reassign the datetime object as the index
	pd.to_datetime(df['Date_UTC'], format='%Y-%m-%d %H:%M:%S')
	df.set_index('Date_UTC', inplace=True, drop=True)
	df.index = pd.to_datetime(df.index)

	df = df.dropna() # shouldn't be any empty rows after that but in case there is we drop them here

	return df


def data_prep(path, station, thresholds, params, forecast, window, do_calc=True):
	''' Preparing the magnetometer data for the other functions and concatinating with the other loaded dfs.
		Inputs:
		path: the file path to the project directory
		station: the ground magnetometer station being examined.
		thresholds(float or list of floats): data threhold beyond which will count as events.
		params: list of input paramaters to add to the features list. This is done because of
				memory limitiations and all other columns will be dropped.
		forecast: how far out the data will be examined for a threshold crossing.
		window: the size of the window that will be exaimned for a threshold crossing. i.e. 30 means the maximum
				value within a 30 minute window will be examined.
		do_calc: (bool) is true if the calculations need to be done, false if this is not the first time
				running this specific CONFIGuration and a csv has been saved. If this is the case the csv
				file will be loaded.
	'''
	print('preparing data...')
	if do_calc:
		print('Reading in CSV...')
		df = pd.read_feather('../data/supermag/{0}.feather'.format(station)) # loading the station data.
		print('Doing calculations...')
		df['dN'] = df['N'].diff(1) # creates the dN column
		df['dE'] = df['E'].diff(1) # creates the dE column
		df['B'] = np.sqrt((df['N']**2)+((df['E']**2))) # creates the combined dB/dt column
		df['dBHt'] = np.sqrt(((df['N'].diff(1))**2)+((df['E'].diff(1))**2)) # creates the combined dB/dt column
		df['direction'] = (np.arctan2(df['dN'], df['dE']) * 180 / np.pi)	# calculates the angle of dB/dt
		df['sinMLT'] = np.sin(df.MLT * 2 * np.pi * 15 / 360)
		df['cosMLT'] = np.cos(df.MLT * 2 * np.pi * 15 / 360)

		print('Setting Datetime...')
		# setting datetime index
		pd.to_datetime(df['Date_UTC'], format='%Y-%m-%d %H:%M:%S')
		df.reset_index(drop=True, inplace=True)
		df.set_index('Date_UTC', inplace=True, drop=False)
		df.index = pd.to_datetime(df.index)

		print('Getting ACE data...')
		acedf = ace_prep(path)

		print('Concatinating dfs...')
		df = pd.concat([df, acedf], axis=1, ignore_index=False)	# adding on the omni data

		threshold = df['dBHt'].quantile(CONFIG['thresholds'])

		print('Isolating selected Features...')	# defining the features to be kept
		df = df[params][1:]	# drops all features not in the features list above and drops the first row because of the derivatives

		print('Creating Classification column...')
		df = classification_column(df, 'dBHt', threshold, forecast=forecast, window=window)		# calling the classification column function
		datum = df.reset_index(drop=True)
		# datum.to_feather('../data/ace_and_supermag/{0}_prepared.feather'.format(station))

	if not do_calc:		# does not do the above calculations and instead just loads a csv file, then creates the cross column
		df = pd.read_feather('../data/{0}_prepared.feather'.format(station))
		pd.to_datetime(df['Date_UTC'], format='%Y-%m-%d %H:%M:%S')
		df.reset_index(drop=True, inplace=True)
		df.set_index('Date_UTC', inplace=True, drop=False)
		df.index = pd.to_datetime(df.index)
		threshold = df['dBHt'].quantile(0.99)

	print('Threshold value: '+str(threshold))
	return df, threshold


def storm_extract(data, storm_list, lead, recovery):
	'''Pulling out storms using a defined list of datetime strings, adding a lead and recovery time to it and
		appending each storm to a list which will be later processed.
		Inputs:
		data: dataframe of OMNI and Supermag data with teh test set's already removed.
		storm_list: datetime list of storms minimums as strings.
		lead: how much time in hours to add to the beginning of the storm.
		recovery: how much recovery time in hours to add to the end of the storm.
		'''
	storms, y_1 = list(), list()					# initalizing the lists
	df = pd.concat(data, ignore_index=True)		# putting all of the dataframes together, makes the searching for stomrs easier

	# setting the datetime index
	pd.to_datetime(df['Date_UTC'], format='%Y-%m-%d %H:%M:%S')
	df.reset_index(drop=True, inplace=True)
	df.set_index('Date_UTC', inplace=True, drop=True)
	df.index = pd.to_datetime(df.index)

	stime, etime = [], []						# will store the resulting time stamps here then append them to the storm time df
	for date in storm_list:					# will loop through the storm dates, create a datetime object for the lead and recovery time stamps and append those to different lists
		stime.append((datetime.strptime(date, '%Y-%m-%d %H:%M:%S'))-pd.Timedelta(hours=lead))
		etime.append((datetime.strptime(date, '%Y-%m-%d %H:%M:%S'))+pd.Timedelta(hours=recovery))
	# adds the time stamp lists to the storm_list dataframes
	storm_list['stime'] = stime
	storm_list['etime'] = etime
	for start, end in zip(storm_list['stime'], storm_list['etime']):		# looping through the storms to remove the data from the larger df
		storm = df[(df.index >= start) & (df.index <= end)]
		storm.dropna(inplace=True)
		if len(storm) != 0:
			storms.append(storm)			# creates a list of smaller storm time dataframes

	for storm in storms:
		storm.reset_index(drop=True, inplace=True)		# resetting the storm index and simultaniously dropping the date so it doesn't get trained on
		y_1.append(to_categorical(storm['crossing'].to_numpy(), num_classes=2))			# turns the one demensional resulting array for the storm into a
		storm.drop('crossing', axis=1, inplace=True)  	# removing the target variable from the storm data so we don't train on it

	return storms, y_1


def split_sequences(sequences, result_y1=None, n_steps=30, include_target=True):
	'''takes input from the data frames and creates the input and target arrays that can go into the models.
		Inputs:
		sequences: dataframe of the input features.
		results_y: series data of the targets for each threshold.
		n_steps: the time history that will define the 2nd demension of the resulting array.
		include_target: true if there will be a target output. False for the testing data.'''

	X, y1 = list(), list()		# creating lists for storing results
	for i in range(len(sequences)-n_steps):										# going to the end of the dataframes
		end_ix = i + n_steps													# find the end of this pattern
		if end_ix > len(sequences):												# check if we are beyond the dataset
			break
		seq_x = sequences[i:end_ix, :]
		if include_target:										# grabs the appropriate chunk of the data
			if np.isnan(seq_x).any():														# doesn't add arrays with nan values to the training set
				continue
		if include_target:
			seq_y1 = result_y1[end_ix]											# gets the appropriate target
			y1.append(seq_y1)
		X.append(seq_x)

	if include_target:
		return np.array(X), np.array(y1)
	if not include_target:
		return np.array(X)


def prep_train_data(df, stime, etime, lead, recovery, time_history, feature_importance):
	''' function that prepares the training data.
		Inputs:
		df: the full, prepared dataframe.
		time_history: amount of time history to be included as input to the model.
		lead: how much time in hours to add to the beginning of the storm.
		recovery: how much recovery time in hours to add to the end of the storm.
	'''

	# using date time index so we can segment out the testing data
	pd.to_datetime(df['Date_UTC'], format='%Y-%m-%d %H:%M:%S')
	df.reset_index(drop=True, inplace=True)
	df.set_index('Date_UTC', inplace=True, drop=True)
	df.index = pd.to_datetime(df.index)

	# Very lazy way to do this but these are the segments that are not those segmented for testing
	start = sorted([((datetime.strptime(s, '%Y-%m-%d %H:%M:%S'))) for s in stime])
	end = sorted([((datetime.strptime(e, '%Y-%m-%d %H:%M:%S'))) for e in etime])
	data = []
	for i in range(len(start)):
		if i == 0:
			data.append(df[df.index < start[i]])
		if (i > 0) & (i < len(start)-1):
			data.append(df[(df.index > end[i-1]) & (df.index < start[i])])
		elif i == len(start)-1:
			data.append(df[(df.index > end[i-1]) & (df.index < start[i])])
			data.append(df[df.index > end[i]])

	# resetting the indexes. The sequence_splitting and storm_search functions are not written to handle datetime index
	for df in data:
		df.reset_index(inplace=True, drop=False)

	print('Loading storm list...')
	storm_list = pd.read_csv('stormList.csv', header=None, names=['dates'])		# loading the list of storms as defined by SYM-H minimum
	for i in range(len(storm_list)):						# cross checking it with testing storms, dropping storms if they're in the test storm list
		d = datetime.strptime(storm_list['dates'][i], '%Y-%m-%d %H:%M:%S')		# converting list of dates to datetime
		for s, e, in zip(start, end):									# drops any storms in the list that overlap with the testing storms
			if (d >= s) & (d <= e):
				storm_list.drop(i, inplace=True)
				print('found one! Get outta here!')

	dates = storm_list['dates']				# just saving it to a variable so I can work with it a bit easier

	print('\nFinding storms...')
	storms_initial, y_1 = storm_extract(data, dates, lead=lead, recovery=recovery)		# extracting the storms using list method
	print('Number of storms: '+str(len(storms_initial)))
	train_dict = {}
	for j in range(16, len(feature_importance)+1):
		storms = []
		for storm in storms_initial:
			storms.append(storm[feature_importance[:j]])
		to_scale_with = pd.concat(storms, axis=0, ignore_index=True)			# finding the largest storm with which we can scale the data. Not sure this is the best way to do this
		scaler = StandardScaler()									# defining the type of scaler to use
		print('Fitting scaler')
		scaler.fit(to_scale_with)									# fitting the scaler to the longest storm
		print('Scaling storms train')
		storms = [scaler.transform(storm) for storm in storms]		# doing a scaler transform to each storm individually
		n_features = storms[1].shape[1]

													# creating a training dictonary for storing everything
		Train, train1 = np.empty((1,time_history,n_features)), np.empty((1,2))	# creating empty arrays for storing sequences
		for storm, y1, i in zip(storms, y_1, range(len(storms))):		# looping through the storms
			X, x1 = split_sequences(storm, y1, n_steps=time_history)				# splitting the sequences for each storm individually
			if X.size == 0:
				continue
			# concatiningting all of the results together into one array for training
			Train = np.concatenate([Train, X])
			if j == 16:
				train1 = np.concatenate([train1, x1])

		# adding all of the training arrays to the dict
		train_dict['X_{0}'.format(j)] = Train
		if j == 16:
			train_dict['crossing'] = train1

	print('Finished calculating percent')

	return train_dict


def define_model(n_features, loss='categorical_crossentropy', early_stop_patience=3):
	'''Initializing our model
		Inputs:
		model_config: predefined model configuration dictonary
		n_features: amount of input features into the model
		loss: loss function to be uesd for training
		early_stop_patience: amount of epochs the model will continue training once there is no longer val loss improvements'''


	model = Sequential()						# initalizing the model

	model.add(Conv2D(MODEL_CONFIG['filters'], 2, padding='same',
									activation='relu', input_shape=(MODEL_CONFIG['time_history'], n_features, 1)))			# adding the CNN layer
	if n_features > 3:
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


def fitting_and_predicting(xtrain, xval, xtest, ytrain, yval):
	'''
		Function that does the fitting of the defined model on the
		training data, saves the fit model, and then makes a prediction
		on the testing data.

		INPUTS:
		X_train (pd.DataFrame): prepared training input data
		X_test (pd.DataFrame): prepared testing input data
		y_train (pd.Series): prepared target data for fitting
		predicting (str): either 'epc' or 'mainheat', used for file path for
							saving the fit model.
		to_fit (bool)
		file_name (str): name of the file for saving the model.

		RETURNS:
		y_pred (np.array): array of predicted values from the model.
	'''

	clear_session()
	model, early_stop = define_model(xtrain.shape[2])		# getting the model
	print('Fitting the classifier....')
	print(model.summary())
	# reshaping the model input vectors for a single channel
	Xtrain = xtrain.reshape((xtrain.shape[0], xtrain.shape[1], xtrain.shape[2], 1))
	Xval = xval.reshape((xval.shape[0], xval.shape[1], xval.shape[2], 1))
	Xtest = xtest.reshape((xtest.shape[0], xtest.shape[1], xtest.shape[2], 1))

	print('XTrain null counts: '+str(np.isnan(Xtrain).sum()))
	print('Xval null counts: '+str(np.isnan(Xval).sum()))
	print('ytrain null counts: '+str(np.isnan(ytrain).sum()))
	print('yval null counts: '+str(np.isnan(yval).sum()))

	model.fit(Xtrain, ytrain, validation_data=(Xval, yval),
				verbose=1, shuffle=True, epochs=MODEL_CONFIG['epochs'], callbacks=[early_stop])			# doing the training! Yay!

	print('Predicting....')
	y_pred = model.predict(Xtest)		# predicting the output value

	y_pred = tf.gather(y_pred, [1], axis=1)					# grabbing the positive node
	y_pred = y_pred.numpy()									# turning to a numpy array

	return y_pred


def getting_metrics(ytest, ypred):
	'''
	Calculates the metrics

	Args:
		ytest (list of np.arrays): real values for all of the cross vals
		ypred (list of np.arrays): predicted values for all cross vals

	Returns:
		floats: mean and std of the
	'''

	AUC, rmse = [], []
	# Loops through each cross val results and adds it to a list
	for test, pred in zip(ytest, ypred):
		prec, rec, ____ = precision_recall_curve(test, pred)
		area = auc(rec, prec)
		AUC.append(area)
		rmse.append(np.sqrt(mean_squared_error(test,pred)))

	return np.mean(AUC), np.std(AUC), np.mean(rmse), np.std(rmse)


def plotting_results(results):
	'''
	Plots the Cross validation results

	Args:
		results (pd.DataFrame): Dataframe of the cross validation results metric scores
	'''

	x = [n+1 for n in range(len(results))]
	fig = plt.figure(figsize=(30,25))

	ax1 = fig.add_subplot(211)
	plt.errorbar(x, results['auc_mean'].to_numpy(), yerr=results['auc_std'].to_numpy(), capsize=1)
	plt.ylabel('AUC Score', fontsize=15)


	ax2 = fig.add_subplot(212)
	plt.errorbar(x, results['rmse_mean'].to_numpy(), yerr=results['rmse_std'].to_numpy(), capsize=1)
	plt.ylabel('RMSE', fontsize=15)

	plt.savefig('CNN_forward_feature_importance_results.png')



def main(path, station):
	'''Here we go baby! bringing it all together.
		Inputs:
		path: path to the data.
		CONFIG: dictonary containing different specifications for the data prep and other things that aren;t the models themselves.
		MODEL_CONFIG: dictonary containing model specifications.
		station: the ground magnetometer station being examined.
		first_time: if True the model will be training and the data prep perfromed. If False will skip these stpes and I probably messed up the plotting somehow.
		'''

	print('Entering main...')

	feature_importance = ['dBHt', 'Vx', 'proton_density', 'B', 'BZ_GSM', 'AE_INDEX',
							'N', 'T', 'SZA', 'cosMLT', 'sinMLT', 'Vy', 'Vz',
							'B_Total', 'E', 'BY_GSM']

	df, threshold = data_prep(path, station, CONFIG['thresholds'], CONFIG['params'], CONFIG['forecast'], CONFIG['window'], do_calc=True)		# calling the data prep function
	train_dict = prep_train_data(df, CONFIG['test_storm_stime'], CONFIG['test_storm_etime'], CONFIG['lead'], CONFIG['recovery'],
											MODEL_CONFIG['time_history'], feature_importance)  												# calling the training data prep function

	train_dict['threshold'] = threshold

	sss = ShuffleSplit(n_splits=CONFIG['cv_splits'], test_size=0.2, random_state=CONFIG['random_seed'])		# defines the lists of training and validation indicies to perform the k fold splitting

	y = train_dict['crossing']				# grabbing the target arrays for training

	auc_mean, auc_std, rmse_mean, rmse_std = list(), list(), list(), list()

	for i in range(16,len(feature_importance)+1):


		X = train_dict['X_{0}'.format(i)]		 # grabbing the training data for model input
		ytest, ypred = [], []				# initalizes lists for the indexes to be stored

		for train_index, test_index in sss.split(y):			# looping through the lists, adding them to other differentiated lists

			# if 'model' in locals():
			# 	reset_keras(model)			# clearing the information from any old models so we can run clean new ones.

			xtrain = X[train_index]

			xtest =  X[test_index]

			ytrain = y[train_index]

			ytest.append(y[test_index][:,1])

			xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=0.2, shuffle=True, random_state=CONFIG['random_seed'])

			y_pred = fitting_and_predicting(xtrain, xval, xtest, ytrain, yval)

			ypred.append(y_pred)

		aucmean, aucstd, rmsemean, rmsestd  = getting_metrics(ytest, ypred)

		auc_mean.append(aucmean)
		auc_std.append(aucstd)
		rmse_mean.append(rmsemean)
		rmse_std.append(rmsestd)

		results = pd.DataFrame({'auc_mean':auc_mean,
								'auc_std':auc_std,
								'rmse_mean':rmse_mean,
								'rmse_std':rmse_std})
		results.to_csv('forward_feature_addition_ver6.csv')


	plotting_results(results)


if __name__ == '__main__':

	for station in CONFIG['stations']:
		main(projectDir, station)
		print('Finished {0}'.format(station))

	print('It ran. Good job!')
