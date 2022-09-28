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

import random
from datetime import datetime
from typing import no_type_check

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

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


CONFIG = {'stations': ['VIC', 'NEW', 'OTT', 'STJ', 'ESK', 'LER', 'WNG', 'BFE'],
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

MODEL_CONFIG = {'time_history': 30, 	# How much time history the model will use, defines the 2nd dimension of the model input array
					'epochs': 100, 		# Maximum amount of empoch the model will run if not killed by early stopping
					'layers': 1, 		# How many CNN layers the model will have.
					'filters': 128, 		# Number of filters in the first CNN layer. Will decrease by half for any subsequent layers if "layers">1
					'dropout': 0.2, 		# Dropout rate for the layers
					'loss':'mse',
					'initial_learning_rate': 1e-5,		# Learning rate, used as the inital learning rate if a learning rate decay function is used
					'lr_decay_steps':230,				# If a learning ray decay funtion is used, dictates the number of decay steps
					'early_stop_patience':5}

# setting the random seeds for reproducibility
random.seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])

def classification_column(df, param, thresholds, forecast, window):
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

	for thresh in thresholds:
		'''This section creates a binary column for each of the thresholds. Binary will be one if the parameter
			goes above the given threshold, and zero if it does not.'''

		conditions = [(df['window_max'] < thresh), (df['window_max'] >= thresh)]			# defining the conditions

		binary = [0, 1] 																	# 0 if not cross 1 if cross

		df['crossing'] = np.select(conditions, binary)						# new column created using the conditions and the binary


	df.drop(['window_max', 'shifted_dBHt'], axis=1, inplace=True)							# removes the two working columns for memory purposes

	return df


def ace_prep(name):

	'''Preparing the omnidata for plotting.
		Inputs:
		path: path to project directory
	'''
	df = pd.read_feather('../data/SW/solarwind_and_indicies{0}.feather'.format(name)) 		# loading the omni data

	df.reset_index(drop=True, inplace=True) 		# reseting the index so its easier to work with integer indexes

	# reassign the datetime object as the index
	pd.to_datetime(df['Date_UTC'], format='%Y-%m-%d %H:%M:%S')
	df.set_index('Date_UTC', inplace=True, drop=True)
	df.index = pd.to_datetime(df.index)

	df = df.dropna() # shouldn't be any empty rows after that but in case there is we drop them here

	return df


def data_prep(name, station, thresholds, params, forecast, window, do_calc=True):
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
	if do_calc:
		df = pd.read_feather('../data/supermag/{0}{1}.feather'.format(station, name)) # loading the station data.
		df['dN'] = df['N'].diff(1) # creates the dN column
		df['dE'] = df['E'].diff(1) # creates the dE column
		df['B'] = np.sqrt((df['N']**2)+((df['E']**2))) # creates the combined dB/dt column
		df['dBHt'] = np.sqrt(((df['N'].diff(1))**2)+((df['E'].diff(1))**2)) # creates the combined dB/dt column
		df['direction'] = (np.arctan2(df['dN'], df['dE']) * 180 / np.pi)	# calculates the angle of dB/dt
		df['sinMLT'] = np.sin(df.MLT * 2 * np.pi * 15 / 360)
		df['cosMLT'] = np.cos(df.MLT * 2 * np.pi * 15 / 360)

		# setting datetime index
		pd.to_datetime(df['Date_UTC'], format='%Y-%m-%d %H:%M:%S')
		df.reset_index(drop=True, inplace=True)
		df.set_index('Date_UTC', inplace=True, drop=False)
		df.index = pd.to_datetime(df.index)

		acedf = ace_prep(name)

		df = pd.concat([df, acedf], axis=1, ignore_index=False)	# adding on the omni data

		df = df[params][1:]	# drops all features not in the features list above and drops the first row because of the derivatives

		df = classification_column(df, 'dBHt', thresholds, forecast=forecast, window=window)		# calling the classification column function
		datum = df.reset_index(drop=True)

	if not do_calc:		# does not do the above calculations and instead just loads a csv file, then creates the cross column
		df = pd.read_feather('../data/{0}_prepared.feather'.format(station))
		pd.to_datetime(df['Date_UTC'], format='%Y-%m-%d %H:%M:%S')
		df.reset_index(drop=True, inplace=True)
		df.set_index('Date_UTC', inplace=True, drop=False)
		df.index = pd.to_datetime(df.index)

	return df


def split_sequences(sequences, n_steps=30, remove_nan=True):
	'''takes input from the data frames and creates the input and target arrays that can go into the models.
		Inputs:
		sequences: dataframe of the input features.
		results_y: series data of the targets for each threshold.
		n_steps: the time history that will define the 2nd demension of the resulting array.
		include_target: true if there will be a target output. False for the testing data.'''

	X = list()		# creating lists for storing results
	nans = 0
	sequences = sequences.to_numpy()
	for i in range(len(sequences)-n_steps):										# going to the end of the dataframes
		end_ix = i + n_steps													# find the end of this pattern
		if end_ix > len(sequences):												# check if we are beyond the dataset
			break
		seq_x = sequences[i:end_ix, :]
		if remove_nan:										# grabs the appropriate chunk of the data
			if np.isnan(seq_x).any():														# doesn't add arrays with nan values to the training set
				nans = nans + 1
		X.append(seq_x)

	return len(X), nans


def prep_test_data(df, stime, etime, params, time_history, prediction_length):
	'''function that segments the selected storms for testing the models. Pulls the data out of the
		dataframe, splits the sequences, and stores the model input arrays and the real results.
		Inputs:
		df: Dataframe containing all of the data.
		stime: array of datetime strings that define the start of the testing storms.
		etime: array of datetime strings that define the end of the testing storms.
		thresholds: array on integers that define the crossing binary for each target array.
		params: list of features to be included as inputs to the models.
		scaler: pre-fit scaler that is uesd to scale teh model input data.
		time_history: amount of time history used to define the 2nd dimension of the model input arrays.
		prediction_length: forecast length+prediction window. Used to cut off the end of the df.'''

	test_dict = {}
	ratios=[]
	total_ratio = pd.DataFrame()										# initalizing the dictonary for storing everything
	for i, (start, end) in enumerate(zip(stime, etime)):		# looping through the different storms
		test_dict['storm_{0}'.format(i)] = {}						# creating a sub-dict for this particular storm

		storm_df = df[start:end]									# cutting out the storm from the greater dataframe
		storm_df.reset_index(inplace=True, drop=False)
		test_dict['storm_{0}'.format(i)]['date'] = storm_df['Date_UTC']		# storing the date series for later plotting
		real_cols = ['Date_UTC', 'dBHt', 'crossing']						# defining real_cols and then adding in the real data to the columns. Used to segment the important data needed for comparison to model outputs

		real_df = storm_df[real_cols][time_history:(len(storm_df)-prediction_length)]		# cutting out the relevent columns. trimmed at the edges to keep length consistent with model outputs
		real_df.reset_index(inplace=True, drop=True)

		storm_df = storm_df[params]												# cuts out the model input parameters
		storm_df.drop(storm_df.tail(prediction_length).index,inplace=True)		# chopping off the prediction length. Cannot predict past the avalable data
		storm_df.drop('Date_UTC', axis=1, inplace=True)							# don't want to train on datetime string
		storm_df.reset_index(inplace=True, drop=True)

		test_dict['storm_{0}'.format(i)]['Y'] = storm_df						# creating a dict element for the model input data
		test_dict['storm_{0}'.format(i)]['real_df'] = real_df					# dict element for the real data for comparison
		re = real_df['crossing']
		total_ratio = pd.concat([total_ratio,re], axis=0)
		ratio = re.sum(axis=0)/len(re)
		ratios.append(ratio)

	print('Storm rations: '+str(ratios))
	print('total ratios for all storms: '+str(total_ratio.sum(axis=0)/len(total_ratio)))
	return test_dict


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

		if len(storm) != 0:
			storms.append(storm)			# creates a list of smaller storm time dataframes

	for storm in storms:
		storm.reset_index(drop=True, inplace=True)		# resetting the storm index and simultaniously dropping the date so it doesn't get trained on
		y_1.append(storm['crossing'])			# turns the one demensional resulting array for the storm into a
		storm.drop('crossing', axis=1, inplace=True)  	# removing the target variable from the storm data so we don't train on it

	return storms, y_1


def prep_train_data(df, stime, etime, lead, recovery):
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

	storm_list = pd.read_csv('stormList.csv', header=None, names=['dates'])		# loading the list of storms as defined by SYM-H minimum
	for i in range(len(storm_list)):						# cross checking it with testing storms, dropping storms if they're in the test storm list
		d = datetime.strptime(storm_list['dates'][i], '%Y-%m-%d %H:%M:%S')		# converting list of dates to datetime
		if (d < datetime.strptime('1998-02-05 00:00:00', '%Y-%m-%d %H:%M:%S')) or (d > datetime.strptime('2018-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')):
			storm_list.drop(i, inplace=True)
		for s, e, in zip(start, end):									# drops any storms in the list that overlap with the testing storms
			if (d >= s) & (d <= e):
				storm_list.drop(i, inplace=True)
	storm_list.reset_index(inplace=True, drop=True)
	dates = storm_list['dates']				# just saving it to a variable so I can work with it a bit easier

	print('\nFinding storms...')
	storms, y_1 = storm_extract(data, dates, lead=lead, recovery=recovery)		# extracting the storms using list method
	print('Number of storms: '+str(len(storms)))


	Total, Nans = 0, 0
	Train, train1 = pd.DataFrame(), pd.Series()	# creating empty arrays for storing sequences
	for storm, y1, i in zip(storms, y_1, range(len(storms))):		# looping through the storms
		total, nans = split_sequences(storm)
		Total = Total + total
		Nans = Nans + nans
		Train = pd.concat([Train, storm], axis=0, ignore_index=True)
		train1 = pd.concat([train1, y1], axis=0)

	print('Train dBHt mean: '+str(Train['dBHt'].mean()))
	ratio = train1.sum(axis=0)/len(train1)
	print('Crossing ratio: '+str(ratio))
	dropped_nans = Train.dropna()
	print('Length of training df: '+str(len(Train)))
	print('Length of Train with dropped Nan: '+str(len(dropped_nans)))

	return Total, Nans


def main():
	'''Here we go baby! bringing it all together.
		Inputs:
		path: path to the data.
		CONFIG: dictonary containing different specifications for the data prep and other things that aren;t the models themselves.
		MODEL_CONFIG: dictonary containing model specifications.
		station: the ground magnetometer station being examined.
		first_time: if True the model will be training and the data prep perfromed. If False will skip these stpes and I probably messed up the plotting somehow.
		'''
	no_interp, interp5, interp15 = [], [], []
	interp = [no_interp, interp5, interp15]
	for station in CONFIG['stations']:
		file_names = ['_no_interp', '_5_interp', '']
		print('Entering main...')
		for file, interp_len in zip(file_names, interp:
			df = data_prep(file, station, CONFIG['thresholds'], CONFIG['params'], CONFIG['forecast'], CONFIG['window'], do_calc=True)		# calling the data prep
			Total, Nans = prep_train_data(df, CONFIG['test_storm_stime'], CONFIG['test_storm_etime'], CONFIG['lead'], CONFIG['recovery'])
			interp_len.append(100-((Nans/Total)*100))											# calling the training data prep function
		print(station)
		print('Totals: '+str(Total))
		print('Nans: '+str(Nans))
		print('Precentages: '+str(percentages))
		test_dict = prep_test_data(df, CONFIG['test_storm_stime'], CONFIG['test_storm_etime'], CONFIG['params'],
								MODEL_CONFIG['time_history'], prediction_length=CONFIG['forecast']+CONFIG['window'])

		PR_dict = sorting_PR(results_dict, station)

	fig = plt.figure(figsize=(30,25))
	plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, hspace=0.03)

	ax = fig.add_subplot(111)
	plt.title('Percentage of Data Avalable Based on Limit of Interpolation ', fontsize='30')
	x = [i for in in range(len(CONFIG['stations']))]
	plt.scatter(x,interp[0], label='No interpolation')
	plt.scatter(x,interp[1], label='5 minutes')
	plt.scatter(x,interp[2], label='15 minutes')
	plt.xlabel('Stations', fontsize='40')
	plt.ylabel('Percentage of Avalable Data', fontsize='40')
	plt.legend(fontsize='30', loc='lower center')
	plt.xticks(ticks=x, labels=CONFIG['stations'], fontsize='28')
	plt.yticks(fontsize='28')
	plt.show()
	plt.savefig('plots/avalable_data.png')

if __name__ == '__main__':

	for station in CONFIG['stations']:
		main(station)
		print('Finished {0}'.format(station))

	print('It ran. Good job!')
