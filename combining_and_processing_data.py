##########################################################################################
#
#	multi-station-dbdt-risk-assessment/preparing_SW_data.py
#
#	Combines the preprocessed magnetometer and solar wind data, segments out the training
# 	and testing data and creates the arrays that will be used for input to the models.
# 	Uses dictonaries to store the np.arrays and the relevent data.
#
#
##########################################################################################

import json
# importing relevent packages
import os
import pickle
import random
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

pd.options.mode.chained_assignment = None # muting an irrelevent warning

# stops this program from hogging the GPU
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass


# loading config and specific model config files. Using them as dictonaries
with open('config.json', 'r') as con:
	CONFIG = json.load(con)

with open('model_config.json', 'r') as mcon:
	MODEL_CONFIG = json.load(mcon)


# setting the random seeds for reproducibility
random.seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])

def classification_column(df, param, thresh, forecast, window):
	'''
	Creating a new column which labels whether there will be a crossing of threshold
		by the param selected in the forecast window.

	Args:
		df (pd.dataframe): dataframe containing the param values.
		param (str): the paramaeter that is being examined for threshold crossings (dBHt for this study).
		thresh (float or list of floats): threshold or list of thresholds to define parameter crossing.
		forecast (int): how far out ahead we begin looking in minutes for threshold crossings.
							If forecast=30, will begin looking 30 minutes ahead.
		window (int): time frame in which we look for a threshold crossing starting at t=forecast.
							If forecast=30, window=30, we look for threshold crossings from t+30 to t+60

	Returns:
		pd.dataframe: df containing a bool column called crossing and a persistance colmun
	'''


	df['shifted_{0}'.format(param)] = df[param].shift(-forecast)					# creates a new column that is the shifted parameter. Because time moves foreward with increasing
																					# index, the shift time is the negative of the forecast instead of positive.
	indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window)			# Yeah this is annoying, have to create a forward rolling indexer because it won't do it automatically.
	df['window_max'] = df.shifted_dBHt.rolling(indexer, min_periods=1).max()		# creates new column in the df labeling the maximum parameter value in the forecast:forecast+window time frame
	df['pers_max'] = df.dBHt.rolling(30, min_periods=1).max()						# looks backwards to find the max param value in the time history limit
	df.reset_index(drop=True, inplace=True)											# resets the index

	'''This section creates a binary column for each of the thresholds. Binary will be one if the parameter
		goes above the given threshold, and zero if it does not.'''

	conditions = [(df['window_max'] < thresh), (df['window_max'] >= thresh)]			# defining the conditions
	pers_conditions = [(df['pers_max'] < thresh), (df['pers_max'] >= thresh)]			# defining the conditions for a persistance model

	binary = [0, 1] 																	# 0 if not cross 1 if cross

	df['crossing'] = np.select(conditions, binary)						# new column created using the conditions and the binary
	df['persistance'] = np.select(pers_conditions, binary)				# creating the persistance column

	df.drop(['pers_max', 'window_max', 'shifted_dBHt'], axis=1, inplace=True)			# removes the working columns for memory purposes

	return df


def ace_prep():
	'''
	Loading the ACE data and setting the index to datetime.

	Returns:
		pd.dataframe: ACE data with datetime index
	'''

	df = pd.read_feather('../data/SW/solarwind_and_indicies.feather') 		# loading the ACE data

	df.reset_index(drop=True, inplace=True) 		# reseting the index

	# reassign the datetime object as the index
	pd.to_datetime(df['Date_UTC'], format='%Y-%m-%d %H:%M:%S')
	df.set_index('Date_UTC', inplace=True, drop=True)
	df.index = pd.to_datetime(df.index)

	df = df.dropna() # shouldn't be any empty rows after that but in case there is we drop them here

	return df


def data_prep(station, threshold, params, forecast, window, do_calc=True):
	'''
		Preparing the magnetometer data for the other functions and concatinating with the other loaded dfs.

		Args:
			station (str): the ground magnetometer station being examined.
			threshold (float): percentile above which will count as events for the selected param.
			params (str or list of strs): input paramaters for model. All other columns will be dropped.
			forecast (int): how far out the data will be examined for a threshold crossing.
			window (int): the size of the window that will be exaimned for a threshold crossing. i.e. 30 means the maximum
					value within a 30 minute window will be examined.
			do_calc: (bool) is true if the calculations need to be done, false if this is not the first time
						running this specific configuration and a csv/feather file has been saved. If this is the case the
						csv/feather file will be loaded.
		Returns:
			pd.dataframe: df with combined supermag, solar wind, and indicies data with classification column,
							persistance column and a datetime index
	'''
	print('preparing data...')
	if do_calc:
		print('Reading in CSV...')
		df = pd.read_feather('../data/supermag/{0}.feather'.format(station)) # loading the station data.
		print('Doing calculations...')
		df['dN'] = df['N'].diff(1) # creates the dN column (nT/min)
		df['dE'] = df['E'].diff(1) # creates the dE column (nT/min)
		df['B'] = np.sqrt((df['N']**2)+((df['E']**2))) # creates the combined dB/dt column (nT/min)
		df['dBHt'] = np.sqrt(((df['N'].diff(1))**2)+((df['E'].diff(1))**2)) # creates the combined dB/dt column
		df['direction'] = (np.arctan2(df['dN'], df['dE']) * 180 / np.pi)	# calculates the angle of dB/dt

		# sin and cos MLT are used to avoid the 23 -> 0 hard drop which the model may have trouble with
		df['sinMLT'] = np.sin(df.MLT * 2 * np.pi * 15 / 360)
		df['cosMLT'] = np.cos(df.MLT * 2 * np.pi * 15 / 360)

		print('Setting Datetime...')
		# setting datetime index
		pd.to_datetime(df['Date_UTC'], format='%Y-%m-%d %H:%M:%S')
		df.reset_index(drop=True, inplace=True)
		df.set_index('Date_UTC', inplace=True, drop=False)
		df.index = pd.to_datetime(df.index)

		print('Getting ACE data...')
		acedf = ace_prep()

		print('Concatinating dfs...')
		df = pd.concat([df, acedf], axis=1, ignore_index=False)	# adding on the ace/indicies data

		thresh = df['dBHt'].quantile(threshold)		# finding the threshold percentile value

		print('Isolating selected Features...')	# defining the features to be kept
		df = df[params][1:]	# drops all features not in the features list above and drops the first row because of the derivatives

		print('Creating Classification column...')
		df = classification_column(df, 'dBHt', thresh, forecast=forecast, window=window)		# calling the classification column function
		datum = df.reset_index(drop=True)
		datum.to_feather('../data/ace_and_supermag/SW_only_{0}_prepared.feather'.format(station))

	if not do_calc:		# does not do the above calculations and instead just loads a csv file, then creates the cross column
		df = pd.read_feather('../data/{0}_prepared.feather'.format(station))
		pd.to_datetime(df['Date_UTC'], format='%Y-%m-%d %H:%M:%S')
		df.reset_index(drop=True, inplace=True)
		df.set_index('Date_UTC', inplace=True, drop=False)
		df.index = pd.to_datetime(df.index)
		threshold = df['dBHt'].quantile(0.99)

	print('Threshold value: '+str(thresh))
	return df, thresh


def split_sequences(sequences, result_y1=None, n_steps=30, include_target=True):
	'''
		Takes input from the input array and creates the input and target arrays that can go into the models.

		Args:
			sequences (np.array): input features. Shape = (length of data, number of input features)
			results_y: series data of the targets for each threshold. Shape = (length of data, 1)
			n_steps (int): the time history that will define the 2nd demension of the resulting array.
			include_target (bool): true if there will be a target output. False for the testing data.

		Returns:
			np.array (n, time history, n_features): array for model input
			np.array (n, 1): target array
		'''

	X, y1 = list(), list()							# creating lists for storing results
	for i in range(len(sequences)-n_steps):			# going to the end of the dataframes
		end_ix = i + n_steps						# find the end of this pattern
		if end_ix > len(sequences):					# check if we are beyond the dataset
			break
		seq_x = sequences[i:end_ix, :]				# grabs the appropriate chunk of the data
		if include_target:
			if np.isnan(seq_x).any():				# doesn't add arrays with nan values to the training set
				continue
		if include_target:
			seq_y1 = result_y1[end_ix]				# gets the appropriate target
			y1.append(seq_y1)
		X.append(seq_x)

	if include_target:
		return np.array(X), np.array(y1)
	if not include_target:
		return np.array(X)


def prep_test_data(df, stime, etime, params, scaler, time_history, prediction_length):
	'''
		Function that segments the selected storms for testing the models. Pulls the data out of the
		dataframe, splits the sequences, and stores the model input arrays and the real results.

		Args:
			df (pd.dataframe): Dataframe containing all of the data.
			stime (str or list of str): array of datetime strings that define the start of the testing storms.
			etime (str or list of str): array of datetime strings that define the end of the testing storms.
			params (str or list of strs): input paramaters for model. All other columns will be dropped.
			scaler (object): pre-fit scaler that is uesd to scale the model input data.
			time_history (int): amount of time history used to define the 2nd dimension of the model input arrays.
			prediction_length (int): forecast length + prediction window. Used to cut off the end of the df.

		Returns:
			dict: dictonary with input arrays for each storm and real data from comparison
		'''

	test_dict = {}
	ratios=[]
	total_ratio = pd.DataFrame()								# initalizing the dictonary for storing everything
	for i, (start, end) in enumerate(zip(stime, etime)):		# looping through the different storms
		test_dict['storm_{0}'.format(i)] = {}					# creating a sub-dict for this particular storm

		storm_df = df[start:end]									# cutting out the storm from the greater dataframe
		storm_df.reset_index(inplace=True, drop=False)
		test_dict['storm_{0}'.format(i)]['date'] = storm_df['Date_UTC']		# storing the date series for later plotting
		real_cols = ['Date_UTC', 'dBHt', 'crossing', 'persistance']			# defining real_cols and then adding in the real data to the columns. Used to segment the important data needed for comparison to model outputs

		real_df = storm_df[real_cols][time_history:(len(storm_df)-prediction_length)]		# cutting out the relevent columns. trimmed at the edges to keep length consistent with model outputs
		real_df.reset_index(inplace=True, drop=True)

		storm_df = storm_df[params]												# cuts out the model input parameters
		storm_df.drop(storm_df.tail(prediction_length).index,inplace=True)		# chopping off the prediction length. Cannot predict past the avalable data
		storm_df.drop(['Date_UTC', 'dBHt'], axis=1, inplace=True)				# don't want to train on datetime string
		storm_df.reset_index(inplace=True, drop=True)
		storm_df = scaler.transform(storm_df)									# scaling the model input data
		storm_df = split_sequences(storm_df, n_steps=time_history, include_target=False)	# calling the split sequences function to create the additional demension

		print('Length of testing inputs: '+str(len(storm_df)))
		print('Length of real storm: '+str(len(real_df)))
		test_dict['storm_{0}'.format(i)]['Y'] = storm_df				# creating a dict element for the model input data
		test_dict['storm_{0}'.format(i)]['real_df'] = real_df			# dict element for the real data for comparison
		re = real_df['crossing']
		total_ratio = pd.concat([total_ratio,re], axis=0) 				# checking the ratio of crossings to non crossings
		ratio = re.sum(axis=0)/len(re)
		ratios.append(ratio)

	print(ratios)
	print(total_ratio.sum(axis=0)/len(total_ratio))
	return test_dict


def storm_extract(data, storm_list, lead, recovery):
	'''
		Pulling out storms using a defined list of datetime strings, adding a lead and recovery time to it and
		appending each storm to a list which will be later processed.

		Args:
			data (list of pd.dataframes): ACE and supermag data with the test set's already removed.
			storm_list (str or list of str): datetime list of storms minimums as strings.
			lead (int): how much time in hours to add to the beginning of the storm.
			recovery (int): how much recovery time in hours to add to the end of the storm.

		Returns:
			list: ace and supermag dataframes for storm times
			list: np.arrays of shape (n,2) containing a one hot encoded boolean target array
		'''
	storms, y_1 = list(), list()				# initalizing the lists
	df = pd.concat(data, ignore_index=True)		# putting all of the dataframes together, makes the searching for stomrs easier

	# setting the datetime index
	pd.to_datetime(df['Date_UTC'], format='%Y-%m-%d %H:%M:%S')
	df.reset_index(drop=True, inplace=True)
	df.set_index('Date_UTC', inplace=True, drop=True)
	df.index = pd.to_datetime(df.index)

	stime, etime = [], []					# will store the resulting time stamps here then append them to the storm time df

	# will loop through the storm dates, create a datetime object for the lead and recovery time stamps and append those to different lists
	for date in storm_list:
		stime.append((datetime.strptime(date, '%Y-%m-%d %H:%M:%S'))-pd.Timedelta(hours=lead))
		etime.append((datetime.strptime(date, '%Y-%m-%d %H:%M:%S'))+pd.Timedelta(hours=recovery))

	# adds the time stamp lists to the storm_list dataframes
	storm_list['stime'] = stime
	storm_list['etime'] = etime
	for start, end in zip(storm_list['stime'], storm_list['etime']):		# looping through the storms to remove the data from the larger df
		storm = df[(df.index >= start) & (df.index <= end)]

		if len(storm) != 0:
			storms.append(storm)			# creates a list of smaller storm time dataframes

	for storm in storms:
		storm.reset_index(drop=True, inplace=True)		# resetting the storm index and simultaniously dropping the date so it doesn't get trained on
		y_1.append(to_categorical(storm['crossing'].to_numpy(), num_classes=2))		# turns the one demensional resulting array for the storm into a
		storm.drop(['persistance', 'crossing', 'dBHt'], axis=1, inplace=True)  		# removing the target variable from the storm data so we don't train on it

	return storms, y_1


def prep_train_data(df, stime, etime, lead, recovery, time_history):
	'''
		Function that prepares the training data.

		Args:
			df (pd.dataframe): the combined ace/indicies/supermag dataframe.
			stime (str or list of str): array of datetime strings that define the start of the testing storms.
			etime (str or list of str): array of datetime strings that define the end of the testing storms.
			lead (int): how much time in hours to add to the beginning of the storm.
			recovery (int): how much recovery time in hours to add to the end of the storm.
			time_history (int): amount of time history used to define the 2nd dimension of the model input arrays.

		Returns:
			dict: containing model input data, target data, and any other relevent data
			object: scaler to be saved for repeat use.
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
	storms, y_1 = storm_extract(data, dates, lead=lead, recovery=recovery)		# extracting the storms using list method
	print('Number of storms: '+str(len(storms)))

	to_scale_with = pd.concat(storms, axis=0, ignore_index=True)	# finding the largest storm with which we can scale the data. Not sure this is the best way to do this
	scaler = StandardScaler()									# defining the type of scaler to use
	print('Fitting scaler')
	scaler.fit(to_scale_with)									# fitting the scaler to the longest storm
	print('Scaling storms train')
	storms = [scaler.transform(storm) for storm in storms]		# doing a scaler transform to each storm individually
	n_features = storms[1].shape[1]				# identifying how many features (columns) are in being used.


	train_dict = {}												# creatinga  training dictonary for storing everything
	Train, train1 = np.empty((1,time_history,n_features)), np.empty((1,2))	# creating empty arrays for storing sequences
	for storm, y1, i in zip(storms, y_1, range(len(storms))):		# looping through the storms
		X, x1 = split_sequences(storm, y1, n_steps=time_history)				# splitting the sequences for each storm individually
		if X.size == 0:
			continue

		# concatiningting all of the results together into one array for training
		Train = np.concatenate([Train, X])
		train1 = np.concatenate([train1, x1])

	# adding all of the training arrays to the dict
	train_dict['X'] = Train
	train_dict['crossing'] = train1

	print('Finished calculating percent')

	return train_dict, scaler


def main(station):
	'''
		Bringing it all together. Also creating the indicies for the 100 different training-validation
		splits so unique modes can be trainined for each split. This provides the uncertainty in the model results.

		Args:
			station (str): the ground magnetometer station being examined.

		'''

	print('Entering main...')

	splits = CONFIG['k_fold_splits']		# denines the number of splits
	df, threshold = data_prep(station, CONFIG['threshold'], CONFIG['params'], CONFIG['forecast'], CONFIG['window'], do_calc=True)		# calling the data prep function
	train_dict, scaler = prep_train_data(df, CONFIG['test_storm_stime'], CONFIG['test_storm_etime'], CONFIG['lead'], CONFIG['recovery'],
											MODEL_CONFIG['time_history'])  												# calling the training data prep function
	if not os.path.exists('multi-station-dbdt-risk-assessment/models/scalers/'):
		os.makedirs('multi-station-dbdt-risk-assessment/models/scalers/')

	with open('multi-station-dbdt-risk-assessment/models/scalers/_SW_only_{0}_standardscaler.pkl'.format(station), 'wb') as f:					# saving the scaler in case I have to run this again
		pickle.dump(scaler, f)			# Goes through all the model training processes if first time going through model

	train_indicies = pd.DataFrame()
	val_indicies = pd.DataFrame()

	train_dict['threshold'] = threshold
	test_dict = prep_test_data(df, CONFIG['test_storm_stime'], CONFIG['test_storm_etime'], CONFIG['params'],
								scaler, MODEL_CONFIG['time_history'], prediction_length=CONFIG['forecast']+CONFIG['window'])						# processing the tesing data

	sss = ShuffleSplit(n_splits=splits, test_size=0.2, random_state=CONFIG['random_seed'])		# defines the lists of training and validation indicies to perform the k fold splitting

	y = train_dict['crossing']				# grabbing the target arrays for training

	train_index, val_index = [], []				# initalizes lists for the indexes to be stored
	for train_i, val_i in sss.split(y):			# looping through the lists, adding them to other differentiated lists
		train_index.append(train_i)
		val_index.append(val_i)

	for train_i, val_i, split in zip(train_index, val_index, range(splits)):
		train_indicies['split_{0}'.format(split)] = train_i
		val_indicies['split_{0}'.format(split)] = val_i


	# saving all files for use in modeling scripts
	train_indicies.to_feather('../data/prepared_data/SW_only_{0}_train_indicies.feather'.format(station))
	val_indicies.to_feather('../data/prepared_data/SW_only_{0}_val_indicies.feather'.format(station))

	with open('../data/prepared_data/SW_only_{0}_train_dict.pkl'.format(station), 'wb') as train:
		pickle.dump(train_dict, train)
	with open('../data/prepared_data/SW_only_{0}_test_dict.pkl'.format(station), 'wb') as test:
		pickle.dump(test_dict, test)


if __name__ == '__main__':

	for station in CONFIG['stations']:
		main(station)
		print('Finished {0}'.format(station))

	print('It ran. Good job!')
