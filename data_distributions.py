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

import os
import pickle
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
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


CONFIG = {'stations': ['BFE', 'WNG', 'LER', 'ESK', 'STJ', 'OTT', 'NEW', 'VIC'],
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
	df['pers_max'] = df.dBHt.rolling(30, min_periods=1).max()
	df.reset_index(drop=True, inplace=True)														# just resets the index

	'''This section creates a binary column for each of the thresholds. Binary will be one if the parameter
		goes above the given threshold, and zero if it does not.'''

	conditions = [(df['window_max'] < thresh), (df['window_max'] >= thresh)]			# defining the conditions
	pers_conditions = [(df['pers_max'] < thresh), (df['pers_max'] >= thresh)]			# defining the conditions

	binary = [0, 1] 																	# 0 if not cross 1 if cross

	df['crossing'] = np.select(conditions, binary)						# new column created using the conditions and the binary
	df['persistance'] = np.select(pers_conditions, binary)

	df.drop(['pers_max', 'window_max', 'shifted_dBHt'], axis=1, inplace=True)							# removes the two working columns for memory purposes

	return df


def ace_prep():

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


def data_prep(station, params, forecast, window, do_calc=True):
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
		acedf = ace_prep()

		print('Concatinating dfs...')
		df = pd.concat([df, acedf], axis=1, ignore_index=False)	# adding on the omni data

		threshold = df['dBHt'].quantile(CONFIG['thresholds'])

		if 'crossing' in params:
			params.remove('crossing')

		print('Isolating selected Features...')	# defining the features to be kept
		df = df[params][1:]	# drops all features not in the features list above and drops the first row because of the derivatives

		print('Creating Classification column...')
		df = classification_column(df, 'dBHt', threshold, forecast=forecast, window=window)		# calling the classification column function
		datum = df.reset_index(drop=True)
		datum.to_feather('../data/ace_and_supermag/{0}_prepared.feather'.format(station))

	if not do_calc:		# does not do the above calculations and instead just loads a csv file, then creates the cross column
		df = pd.read_feather('../data/{0}_prepared.feather'.format(station))
		pd.to_datetime(df['Date_UTC'], format='%Y-%m-%d %H:%M:%S')
		df.reset_index(drop=True, inplace=True)
		df.set_index('Date_UTC', inplace=True, drop=False)
		df.index = pd.to_datetime(df.index)
		threshold = df['dBHt'].quantile(0.99)

	return df



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

	test_df = pd.DataFrame()
	ratios=[]
	total_ratio = pd.DataFrame()										# initalizing the dictonary for storing everything
	cols = params
	cols.append('crossing')
	for i, (start, end) in enumerate(zip(stime, etime)):		# looping through the different storms

		storm_df = df[start:end]									# cutting out the storm from the greater dataframe
		storm_df.reset_index(inplace=True, drop=False)
		real_cols = ['Date_UTC', 'dBHt', 'crossing', 'persistance']						# defining real_cols and then adding in the real data to the columns. Used to segment the important data needed for comparison to model outputs

		real_df = storm_df[real_cols][time_history:(len(storm_df)-prediction_length)]		# cutting out the relevent columns. trimmed at the edges to keep length consistent with model outputs
		real_df.reset_index(inplace=True, drop=True)

		storm_df = storm_df[cols]												# cuts out the model input parameters
		storm_df.drop(storm_df.tail(prediction_length).index,inplace=True)		# chopping off the prediction length. Cannot predict past the avalable data
		storm_df.drop('Date_UTC', axis=1, inplace=True)							# don't want to train on datetime string
		storm_df.reset_index(inplace=True, drop=True)

		print('Length of testing inputs: '+str(len(storm_df)))
		print('Length of real storm: '+str(len(real_df)))
		test_df = pd.concat([test_df, storm_df], axis=0, ignore_index=True)						# creating a dict element for the model input data
		re = real_df['crossing']
		total_ratio = pd.concat([total_ratio,re], axis=0)
		ratio = re.sum(axis=0)/len(re)
		ratios.append(ratio)

	print(ratios)
	ratio = total_ratio.sum(axis=0)/len(total_ratio)
	print(total_ratio.sum(axis=0)/len(total_ratio))

	return test_df, ratio


def storm_extract(data, storm_list, lead, recovery):
	'''Pulling out storms using a defined list of datetime strings, adding a lead and recovery time to it and
		appending each storm to a list which will be later processed.
		Inputs:
		data: dataframe of OMNI and Supermag data with teh test set's already removed.
		storm_list: datetime list of storms minimums as strings.
		lead: how much time in hours to add to the beginning of the storm.
		recovery: how much recovery time in hours to add to the end of the storm.
		'''
	storms = list()					# initalizing the lists
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

	return storms


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
	storms = storm_extract(data, dates, lead=lead, recovery=recovery)		# extracting the storms using list method
	print('Number of storms: '+str(len(storms)))

	Train = pd.concat(storms, axis=0)

	# adding all of the training arrays to the dict

	print('Finished calculating percent')

	return Train


def getting_distributions(all_dfs):

	all_mean, all_std = [], []
	train_mean, train_std = [], []
	test_mean, test_std = [], []
	ratios = []

	for station in all_dfs.keys():
		all_mean.append(all_dfs[station]['all']['dBHt'].mean())
		all_std.append(all_dfs[station]['all']['dBHt'].std())
		train_mean.append(all_dfs[station]['train']['dBHt'].mean())
		train_std.append(all_dfs[station]['train']['dBHt'].std())
		test_mean.append(all_dfs[station]['test']['dBHt'].mean())
		test_std.append(all_dfs[station]['test']['dBHt'].std())
		ratios.append(all_dfs[station]['ratio'])

	all_data = pd.DataFrame({'mean':all_mean,
								'std':all_std},
								index=CONFIG['stations'])
	train_data = pd.DataFrame({'mean':train_mean,
								'std':train_std},
								index=CONFIG['stations'])
	test_data = pd.DataFrame({'mean':test_mean,
								'std':test_std},
								index=CONFIG['stations'])
	ratio_data = pd.DataFrame({'ratio':ratios},
								index=CONFIG['stations'])

	return all_data, train_data, test_data, ratio_data


def plotting_data_distributions(all_data, train_data, test_data):

	fig = plt.figure(figsize=(40,35))													# establishing the figure
	plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, hspace=0.03)			# trimming the whitespace in the subplots

	X = [5, 35, 65, 95, 125, 155, 185, 215]				# need to find a better way to do this. Used for labeling the x axis of the plots for each threshold.

	x0 = [(num-4) for num in X]				# need to find a better way to do this. Used for labeling the x axis of the plots for each threshold.
	x1 = [(num-0) for num in X]				# need to find a better way to do this. Used for labeling the x axis of the plots for each threshold.
	x2 = [(num+4) for num in X]				# need to find a better way to do this. Used for labeling the x axis of the plots for each threshold.

	ax = fig.add_subplot(111)					# adding the subplot
	y0 = all_data['mean'].to_numpy()			# defining the y center point
	ystd0 = all_data['std'].to_numpy()			# defining the y upper bound

	y1 = train_data['mean'].to_numpy()			# defining the y center point
	ystd1 = train_data['std'].to_numpy()			# defining the y upper bound

	y2 = test_data['mean'].to_numpy()			# defining the y center point
	ystd2 = test_data['std'].to_numpy()			# defining the y upper bound


	plt.title('All-Train-Test dB/dt Distributions', fontsize='100')		# titling the plot
	ax.errorbar(x0, y0, yerr=ystd0, fmt='.k', color='blue', label='All Data', elinewidth=5, markersize=50, capsize=25, capthick=5)		# plotting the center point with the error bars. list order is important in the y array so it cooresponds to the x label
	ax.errorbar(x1, y1, yerr=ystd1, fmt='.k', color='orange', label='Train Data', elinewidth=5, markersize=50, capsize=25, capthick=5)		# plotting the center point with the error bars. list order is important in the y array so it cooresponds to the x label
	ax.errorbar(x2, y2, yerr=ystd2, fmt='.k', color='green', label='Test Data', elinewidth=5, markersize=50, capsize=25, capthick=5)		# plotting the center point with the error bars. list order is important in the y array so it cooresponds to the x label
	plt.xlabel('Stations', fontsize='70')			# adding the label on the x axis label
	plt.ylabel('dB/dt', fontsize='70')				# adding teh y axis label
	plt.xticks(X, CONFIG['stations'], fontsize='70')		# adding ticks to the points on the x axis
	plt.yticks(fontsize='58')						# making the y ticks a bit bigger. They're a bit more important
	plt.legend(fontsize='65')

	plt.savefig('plots/data_distribution.png')



def main():
	'''Here we go baby! bringing it all together.
		Inputs:
		path: path to the data.
		CONFIG: dictonary containing different specifications for the data prep and other things that aren;t the models themselves.
		MODEL_CONFIG: dictonary containing model specifications.
		station: the ground magnetometer station being examined.
		first_time: if True the model will be training and the data prep perfromed. If False will skip these stpes and I probably messed up the plotting somehow.
		'''

	print('Entering main...')
	all_dfs = {}
	for station in CONFIG['stations']:
		all_dfs[station] = {}
		df = data_prep(station, CONFIG['params'], CONFIG['forecast'], CONFIG['window'], do_calc=True)		# calling the data prep function
		all_dfs[station]['all'] = df
		train_df = prep_train_data(df, CONFIG['test_storm_stime'], CONFIG['test_storm_etime'], CONFIG['lead'], CONFIG['recovery'])  												# calling the training data prep function
		all_dfs[station]['train'] = train_df

		test_df, ratio = prep_test_data(df, CONFIG['test_storm_stime'], CONFIG['test_storm_etime'], CONFIG['params'],
								MODEL_CONFIG['time_history'], prediction_length=CONFIG['forecast']+CONFIG['window'])						# processing the tesing data

		all_dfs[station]['test'] = test_df
		all_dfs[station]['ratio'] = ratio

	all_dfs, train_df, test_df, ratio_df = getting_distributions(all_dfs)

	ratio_df.to_csv('outputs/total_storm_positive_ratios.csv', index=False)

	plotting_data_distributions(all_dfs, train_df, test_df)


if __name__ == '__main__':

	main()

	print('It ran. Good job!')
