##########################################################################################
#
#	multi-station-dbdt-risk-assessment/preparing_SW_data.py
#
#
#
#
#
#
##########################################################################################

import pickle

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc
from tensorflow.keras.models import load_model

# stops this program from hogging the GPU
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

CONFIG = {'version':0,
			'thresholds': [7.15], # list of thresholds to be examined.
      		'params': ['Date_UTC', 'N', 'E', 'sinMLT', 'cosMLT', 'B_Total', 'BY_GSM',
              'BZ_GSM', 'Vx', 'Vy', 'Vz', 'proton_density', 'T',
               'AE_INDEX', 'SZA', 'dBHt', 'B', 'MLT'],                  # List of parameters that will be used for training.
                                                  # Date_UTC will be removed, kept here for resons that will be evident below
      		'test_storm_stime': ['2001-03-29 09:59:00', '2001-08-29 21:59:00', '2005-05-13 21:59:00',
                 '2005-08-30 07:59:00', '2006-12-13 09:59:00', '2010-04-03 21:59:00',
                 '2011-08-04 06:59:00', '2015-03-15 23:59:00'],           # These are the start times for testing storms
      		'test_storm_etime': ['2001-04-02 12:00:00', '2001-09-02 00:00:00', '2005-05-17 00:00:00',
                  '2005-09-02 12:00:00', '2006-12-17 00:00:00', '2010-04-07 00:00:00',
                  '2011-08-07 09:00:00', '2015-03-19 14:00:00'],  # end times for testing storms. This will remove them from training
			'plot_titles': ['March 2001', 'September 2001', 'May 2005', 'September 2005', 'December 2006', 'April 2010', 'August 2011', 'March 2015'],						# list used for plot titles so I don't have to do it manually
			'forecast': 30,
			'window': 30,																	# time window over which the metrics will be calculated
			'splits': 100,															# amount of k fold splits to be performed. Program will create this many models
			'stations': ['BFE', 'WNG', 'LER', 'ESK', 'STJ', 'OTT', 'NEW', 'VIC'],
			'metrics': ['HSS', 'BIAS', 'STD_PRED', 'RMSE', 'AUC']}



def load_stats():

	with open('outputs/stations_results_dict.pkl', 'rb') as f:
		stations_dict = pickle.load(f)

	return stations_dict


def sorting_metrics(results_dict, stations, metrics, length):

	metrics_dict = {}
	for metric in metrics:
		metric_dict = {}
		for i in range(length):
			df = pd.DataFrame()
			for station in stations:
				df[station] = results_dict[station]['storm_{0}'.format(i)]['metrics'][metric]
			metric_dict['storm_{0}'.format(i)] = df.T
		metrics_dict[metric] = metric_dict
		total_df = pd.DataFrame()
		for station in stations:
			total_df[station] = results_dict[station]['total_metrics'][metric]
		metrics_dict['total_{0}'.format(metric)] = total_df.T
	hss, auc, rmse = [], [], []
	for station in stations:
		hss.append(results_dict[station]['pers_HSS'][0])
		auc.append(results_dict[station]['pers_AUC'][0])
		rmse.append(results_dict[station]['pers_RMSE'][0])
	metrics_dict['pers_HSS'] = hss
	metrics_dict['pers_AUC'] = auc
	metrics_dict['pers_RMSE'] = rmse

	return metrics_dict



def plot_metrics(metrics_dict, stations, metrics):
	'''plotting function for the metrics. Not a lot of comments in this one.
		Inputs:
		metrics: dataframe of the saved metrics from the classification function.
		thresholds: list of thresholds. There will be one plot for each threshold.
		'''
	for metric in metrics:

		fig = plt.figure(figsize=(10,7))													# establishing the figure
		plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, hspace=0.03)			# trimming the whitespace in the subplots

		X = [5, 35, 65, 95, 125, 155, 185, 215]				# need to find a better way to do this. Used for labeling the x axis of the plots for each threshold.

		x0 = [(num-3.5) for num in X]				# need to find a better way to do this. Used for labeling the x axis of the plots for each threshold.
		x1 = [(num-2.5) for num in X]				# need to find a better way to do this. Used for labeling the x axis of the plots for each threshold.
		x2 = [(num-1.5) for num in X]				# need to find a better way to do this. Used for labeling the x axis of the plots for each threshold.
		x3 = [(num-0.5) for num in X]				# need to find a better way to do this. Used for labeling the x axis of the plots for each threshold.
		x4 = [(num+0.5) for num in X]				# need to find a better way to do this. Used for labeling the x axis of the plots for each threshold.
		x5 = [(num+1.5) for num in X]				# need to find a better way to do this. Used for labeling the x axis of the plots for each threshold.
		x6 = [(num+2.5) for num in X]				# need to find a better way to do this. Used for labeling the x axis of the plots for each threshold.
		x7 = [(num+3.5) for num in X]				# need to find a better way to do this. Used for labeling the x axis of the plots for each threshold.

		ax = fig.add_subplot(111)					# adding the subplot
		y0 = metrics_dict[metric]['storm_0']['mean'].to_numpy()		# defining the y center point
		ymax0 = metrics_dict[metric]['storm_0']['max'].to_numpy()	# defining the y upper bound
		ymin0 = metrics_dict[metric]['storm_0']['min'].to_numpy()	# defining the y lower bound

		ymax0 = ymax0 - y0
		ymin0 = y0 - ymin0

		y1 = metrics_dict[metric]['storm_1']['mean'].to_numpy()		# defining the y center point
		ymax1 = metrics_dict[metric]['storm_1']['max'].to_numpy()	# defining the y upper bound
		ymin1 = metrics_dict[metric]['storm_1']['min'].to_numpy()	# defining the y lower bound

		ymax1 = ymax1 - y1
		ymin1 = y1 - ymin1

		y2 = metrics_dict[metric]['storm_2']['mean'].to_numpy()		# defining the y center point
		ymax2 = metrics_dict[metric]['storm_2']['max'].to_numpy()	# defining the y upper bound
		ymin2 = metrics_dict[metric]['storm_2']['min'].to_numpy()	# defining the y lower bound

		ymax2 = ymax2 - y2
		ymin2 = y2 - ymin2

		y3 = metrics_dict[metric]['storm_3']['mean'].to_numpy()		# defining the y center point
		ymax3 = metrics_dict[metric]['storm_3']['max'].to_numpy()	# defining the y upper bound
		ymin3 = metrics_dict[metric]['storm_3']['min'].to_numpy()	# defining the y lower bound

		ymax3 = ymax3 - y3
		ymin3 = y3 - ymin3

		y4 = metrics_dict[metric]['storm_4']['mean'].to_numpy()		# defining the y center point
		ymax4 = metrics_dict[metric]['storm_4']['max'].to_numpy()	# defining the y upper bound
		ymin4 = metrics_dict[metric]['storm_4']['min'].to_numpy()	# defining the y lower bound

		ymax4 = ymax4 - y4
		ymin4 = y4 - ymin4

		y5 = metrics_dict[metric]['storm_5']['mean'].to_numpy()		# defining the y center point
		ymax5 = metrics_dict[metric]['storm_5']['max'].to_numpy()	# defining the y upper bound
		ymin5 = metrics_dict[metric]['storm_5']['min'].to_numpy()	# defining the y lower bound

		ymax5 = ymax5 - y5
		ymin5 = y5 - ymin5

		y6 = metrics_dict[metric]['storm_6']['mean'].to_numpy()		# defining the y center point
		ymax6 = metrics_dict[metric]['storm_6']['max'].to_numpy()	# defining the y upper bound
		ymin6 = metrics_dict[metric]['storm_6']['min'].to_numpy()	# defining the y lower bound

		ymax6 = ymax6 - y6
		ymin6 = y6 - ymin6

		y7 = metrics_dict[metric]['storm_7']['mean'].to_numpy()		# defining the y center point
		ymax7 = metrics_dict[metric]['storm_7']['max'].to_numpy()	# defining the y upper bound
		ymin7 = metrics_dict[metric]['storm_7']['min'].to_numpy()	# defining the y lower bound

		ymax7 = ymax7 - y7
		ymin7 = y7 - ymin7

		plt.title(metric, fontsize='20')		# titling the plot
		ax.errorbar(x0, y0, yerr=[ymin0, ymax0], fmt='.k', color='blue', label='Mar 2001', elinewidth=2, markersize=15, capsize=4, capthick=2)		# plotting the center point with the error bars. list order is important in the y array so it cooresponds to the x label
		ax.errorbar(x1, y1, yerr=[ymin1, ymax1], fmt='.k', color='orange', label='Sep 2001', elinewidth=2, markersize=15, capsize=4, capthick=2)		# plotting the center point with the error bars. list order is important in the y array so it cooresponds to the x label
		ax.errorbar(x2, y2, yerr=[ymin2, ymax2], fmt='.k', color='green', label='May 2005', elinewidth=2, markersize=15, capsize=4, capthick=2)		# plotting the center point with the error bars. list order is important in the y array so it cooresponds to the x label
		ax.errorbar(x3, y3, yerr=[ymin3, ymax3], fmt='.k', color='red', label='Sep 2005', elinewidth=2, markersize=15, capsize=4, capthick=2)		# plotting the center point with the error bars. list order is important in the y array so it cooresponds to the x label
		ax.errorbar(x4, y4, yerr=[ymin4, ymax4], fmt='.k', color='purple', label='Dec 2006', elinewidth=2, markersize=15, capsize=4, capthick=2)		# plotting the center point with the error bars. list order is important in the y array so it cooresponds to the x label
		ax.errorbar(x5, y5, yerr=[ymin5, ymax5], fmt='.k', color='brown', label='Apr 2010', elinewidth=2, markersize=15, capsize=4, capthick=2)		# plotting the center point with the error bars. list order is important in the y array so it cooresponds to the x label
		ax.errorbar(x6, y6, yerr=[ymin6, ymax6], fmt='.k', color='pink', label='Aug 2011', elinewidth=2, markersize=15, capsize=4, capthick=2)		# plotting the center point with the error bars. list order is important in the y array so it cooresponds to the x label
		ax.errorbar(x7, y7, yerr=[ymin7, ymax7], fmt='.k', color='black', label='Mar 2015', elinewidth=2, markersize=15, capsize=4, capthick=2)		# plotting the center point with the error bars. list order is important in the y array so it cooresponds to the x label
		plt.axhline(0, color='black')
		# plt.ylim(0,1)																# keeping the plot within limits to eliminate as much white space as possible.
		# plt.xlim(0,70)
		plt.xlabel('Stations', fontsize='15')			# adding the label on the x axis label
		plt.ylabel(metric, fontsize='15')				# adding teh y axis label
		plt.xticks(X, stations, fontsize='15')		# adding ticks to the points on the x axis
		plt.yticks(fontsize='15')						# making the y ticks a bit bigger. They're a bit more important
		plt.legend(fontsize='10')

		plt.savefig('plots/{0}_version_{1}.png'.format(metric, CONFIG['version']), bbox_inches='tight')


def plot_total_metrics(metrics_dict, stations, metrics=['HSS', 'AUC', 'RMSE']):
	'''plotting function for the metrics. Not a lot of comments in this one.
		Inputs:
		metrics: dataframe of the saved metrics from the classification function.
		thresholds: list of thresholds. There will be one plot for each threshold.
		'''



	fig = plt.figure(figsize=(10,7))													# establishing the figure
	plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, hspace=0.03)			# trimming the whitespace in the subplots

	X = [5, 15, 25, 35, 45, 55, 65, 75]				# need to find a better way to do this. Used for labeling the x axis of the plots for each threshold.

	x0 = [(num-1.5) for num in X]				# need to find a better way to do this. Used for labeling the x axis of the plots for each threshold.
	x1 = [(num+1.5) for num in X]				# need to find a better way to do this. Used for labeling the x axis of the plots for each threshold.


	ax = fig.add_subplot(111)					# adding the subplot
	plt.title('Metric Scores', fontsize='20')		# titling the plot
	bar_colors = ['blue', 'tomato']
	persistance_colors = ['deepskyblue', 'orange']
	for metric, color0, color1 in zip(metrics, bar_colors, persistance_colors):
		y0 = metrics_dict['total_{0}'.format(metric)]['mean'].to_numpy()		# defining the y center point
		ymax0 = metrics_dict['total_{0}'.format(metric)]['max'].to_numpy()	# defining the y upper bound
		ymin0 = metrics_dict['total_{0}'.format(metric)]['min'].to_numpy()	# defining the y lower bound

		ymax0 = ymax0 - y0
		ymin0 = y0 - ymin0


		ax.errorbar(X, y0, yerr=[ymin0, ymax0], fmt='.', color=color0, label='{0}'.format(metric), elinewidth=3, markersize=15, capsize=4, capthick=3)		# plotting the center point with the error bars. list order is important in the y array so it cooresponds to the x label
		ax.scatter(X, metrics_dict['pers_{0}'.format(metric)], marker='^', color=color1, label='pers.{0}'.format(metric), s=150)
	# plt.ylim(0,1)																# keeping the plot within limits to eliminate as much white space as possible.
	# plt.xlim(0,70)
	plt.xlabel('Stations', fontsize='15')		# adding the label on the x axis label
	plt.ylabel('Score', fontsize='15')			# adding the y axis label
	plt.xticks(X, stations, fontsize='15')		# adding ticks to the points on the x axis
	plt.yticks(fontsize='15')					# making the y ticks a bit bigger. They're a bit more important
	plt.legend(fontsize='10', loc='lower left')

	plt.savefig('plots/metrics_total.png', bbox_inches='tight')

def prep_k_fold_results(df, splits):
	'''prepares the data from the k-folds for plotting and examination. Creates a dataframe that stores the upper and lower calculated bounds for the plotting.
		Inputs:
		df: dataframe from a particular storm that is being examined. Includes the real data and the predicted data for each split model.
		threshold: threshold that is being examined.
		splits: integer number of split models that have been trained. Used here for indexing.
		stime: datetime string that defines the start time for the plotting.
		etime: datetime string that defines the end time for the plotting.'''

	newdf = pd.DataFrame()				# initalizes the new dataframe
	for split in range(splits):			# looping through the splits
		newdf['split_{0}'.format(split)] = df['predicted_split_{0}'.format(split)]		# grabs the predicted data for the threshold examined and puts them in the newdf
	mean = newdf.mean(axis=1)
	top_perc = newdf.quantile(0.975, axis=1)
	bottom_perc = newdf.quantile(0.025, axis=1)
	newdf['cross'] = df['crossing']
	newdf['persistance'] = df['persistance']
	newdf['mean'] = mean			# calculated the mean for each row and creates a new collumn
	newdf['top_perc'] = top_perc	# creates a new column and populates it with the 97.5th percentile data from each row
	newdf['bottom_perc'] = bottom_perc		# creates a new column and populates it with the 2.5th percentile data from each row
	newdf['date'] = df.index							# establishes a date column that will be used as the index

	# establishes the datetime index from the date column
	pd.to_datetime(newdf['date'], format='%Y-%m-%d %H:%M:%S')
	newdf.reset_index(drop=True, inplace=True)
	newdf.set_index('date', inplace=True, drop=True)
	newdf.index = pd.to_datetime(newdf.index)

	return newdf



def sorting_PR(results_dict, station):

	PR_dict = {}
	for i in range(8):
		PR_dict['storm_{0}'.format(i)] = {}
		df = results_dict[station]['storm_{0}'.format(i)]['precision_recall']
		try:
			PR_dict['storm_{0}'.format(i)]['prec'] = df['prec'].to_numpy()
			PR_dict['storm_{0}'.format(i)]['rec'] = df['rec'].to_numpy()
			PR_dict['storm_{0}'.format(i)]['auc'] = auc(df['rec'].to_numpy(), df['prec'].to_numpy()).round(decimals=3)
		except KeyError:
			print('skipping this storm-station combo')

	return PR_dict


def plot_precision_recall(results_dict, station, plot_titles):

	PR_dict = sorting_PR(results_dict, station)

	fig = plt.figure(figsize=(10,7))
	plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, hspace=0.03)

	ax = fig.add_subplot(111)
	plt.title('Precision-Recall Curves for {0} Station'.format(station), fontsize='20')
	for i, title in enumerate(plot_titles):
		try:
			prec = PR_dict['storm_{0}'.format(i)]['prec']
			rec = PR_dict['storm_{0}'.format(i)]['rec']
			area = PR_dict['storm_{0}'.format(i)]['auc']
			plt.plot(rec, prec, linewidth=4, label='{0} AUC:{1}'.format(title, area))
		except:
			print('skipping this station-storm combo')
	plt.xlabel('Recall', fontsize='15')
	plt.ylabel('Precision', fontsize='15')
	plt.legend(fontsize='10', loc='lower center')
	plt.xticks(fontsize='15')
	plt.yticks(fontsize='15')
	plt.savefig('plots/precision_recall_{0}.png'.format(station))


def reliability_plots(results_dict, station, splits, plot_titles):
	'''plotting the reliability of our model results'''

	storm0 = prep_k_fold_results(results_dict[station]['storm_0']['raw_results'], splits)
	storm1 = prep_k_fold_results(results_dict[station]['storm_1']['raw_results'], splits)
	storm2 = prep_k_fold_results(results_dict[station]['storm_2']['raw_results'], splits)
	storm3 = prep_k_fold_results(results_dict[station]['storm_3']['raw_results'], splits)
	storm4 = prep_k_fold_results(results_dict[station]['storm_4']['raw_results'], splits)
	storm5 = prep_k_fold_results(results_dict[station]['storm_5']['raw_results'], splits)
	storm6 = prep_k_fold_results(results_dict[station]['storm_6']['raw_results'], splits)
	storm7 = prep_k_fold_results(results_dict[station]['storm_7']['raw_results'], splits)

	newdfs = [storm0, storm1, storm2, storm3, storm4, storm5, storm6, storm7]

	newdf = pd.concat(newdfs, axis=0)

	fig = plt.figure(figsize=(20,18))
	plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, hspace=0.03)

	ax = fig.add_subplot(111)
	ax.set_title('{0} Reliability Plot'.format(station), fontsize=30)
	plt.plot([0, 1], [0, 1], 'xkcd:black')
	# for df, title in zip(newdfs, plot_titles):
	# 	true, pred = calibration_curve(df['cross'], df['mean'], n_bins=10)
	# 	plt.plot(pred, true, marker='.', label='{0}'.format(title))
	# 	ax.set_xlabel('Predicted Probability', fontsize=20)
	# 	ax.set_ylabel('Observed Probability', fontsize=20)
	true, pred = calibration_curve(newdf['cross'], newdf['mean'], n_bins=10)
	plt.plot(pred, true, marker='.')
	ax.set_xlabel('Predicted Probability', fontsize=20)
	ax.set_ylabel('Observed Probability', fontsize=20)
	plt.yticks(fontsize=15)
	plt.xticks(fontsize=15)
	ax.margins(x=0, y=0)
	plt.savefig('plots/{0}_reliability_plot_ver2.png'.format(station))


def plot_model_outputs(results_dict, storm, splits, title, stime, etime):
	'''plots all of the model output results with confidence intervals.
		Inputs:
		df: dataframe for a particular storm
		stime: datetime string for starting time of the plot.
		etime: datetime string for ending time of the plot.
		splits: integer number of splits used for model training
	'''

	# calling the prep_k_fold function for each threshold. Should probably find a better way to do this.
	OTT = prep_k_fold_results(results_dict['OTT']['storm_{0}'.format(storm)]['raw_results'], splits)
	BFE = prep_k_fold_results(results_dict['BFE']['storm_{0}'.format(storm)]['raw_results'], splits)
	WNG = prep_k_fold_results(results_dict['WNG']['storm_{0}'.format(storm)]['raw_results'], splits)
	STJ = prep_k_fold_results(results_dict['STJ']['storm_{0}'.format(storm)]['raw_results'], splits)
	NEW = prep_k_fold_results(results_dict['NEW']['storm_{0}'.format(storm)]['raw_results'], splits)
	VIC = prep_k_fold_results(results_dict['VIC']['storm_{0}'.format(storm)]['raw_results'], splits)
	ESK = prep_k_fold_results(results_dict['ESK']['storm_{0}'.format(storm)]['raw_results'], splits)
	LER = prep_k_fold_results(results_dict['LER']['storm_{0}'.format(storm)]['raw_results'], splits)


	# this creats a new dataframe that will allow me to create a bar at the top of the plot to define the periods where the real, binary values have value 1.
	OTT_bar = pd.DataFrame({'OTT_bottom':OTT['cross']*1.01,
							'OTT_top':OTT['cross']*1.06,
							'pers_bottom':OTT['persistance']*1.07,
							'pers_top':OTT['persistance']*1.12},
							index=OTT.index)
	BFE_bar = pd.DataFrame({'BFE_bottom':BFE['cross']*1.01,
							'BFE_top':BFE['cross']*1.06,
							'pers_bottom':BFE['persistance']*1.07,
							'pers_top':BFE['persistance']*1.12},
							index=BFE.index)
	WNG_bar = pd.DataFrame({'WNG_bottom':WNG['cross']*1.01,
							'WNG_top':WNG['cross']*1.06,
							'pers_bottom':WNG['persistance']*1.07,
							'pers_top':WNG['persistance']*1.12},
							index=WNG.index)
	STJ_bar = pd.DataFrame({'STJ_bottom':STJ['cross']*1.01,
							'STJ_top':STJ['cross']*1.06,
							'pers_bottom':STJ['persistance']*1.07,
							'pers_top':STJ['persistance']*1.12},
							index=STJ.index)
	NEW_bar = pd.DataFrame({'NEW_bottom':NEW['cross']*1.01,
							'NEW_top':NEW['cross']*1.06,
							'pers_bottom':NEW['persistance']*1.07,
							'pers_top':NEW['persistance']*1.12},
							index=NEW.index)
	VIC_bar = pd.DataFrame({'VIC_bottom':VIC['cross']*1.01,
							'VIC_top':VIC['cross']*1.06,
							'pers_bottom':VIC['persistance']*1.07,
							'pers_top':VIC['persistance']*1.12},
							index=VIC.index)
	ESK_bar = pd.DataFrame({'ESK_bottom':ESK['cross']*1.01,
							'ESK_top':ESK['cross']*1.06,
							'pers_bottom':ESK['persistance']*1.07,
							'pers_top':ESK['persistance']*1.12},
							index=ESK.index)
	LER_bar = pd.DataFrame({'LER_bottom':LER['cross']*1.01,
							'LER_top':LER['cross']*1.06,
							'pers_bottom':LER['persistance']*1.07,
							'pers_top':LER['persistance']*1.12},
							index=LER.index)

	OTT_bar.index=pd.to_datetime(OTT_bar.index)					# adds datetime index
	BFE_bar.index=pd.to_datetime(BFE_bar.index)					# adds datetime index
	WNG_bar.index=pd.to_datetime(WNG_bar.index)					# adds datetime index
	STJ_bar.index=pd.to_datetime(STJ_bar.index)					# adds datetime index
	NEW_bar.index=pd.to_datetime(NEW_bar.index)					# adds datetime index
	VIC_bar.index=pd.to_datetime(VIC_bar.index)					# adds datetime index
	ESK_bar.index=pd.to_datetime(ESK_bar.index)					# adds datetime index
	LER_bar.index=pd.to_datetime(LER_bar.index)					# adds datetime index


	fig = plt.figure(figsize=(25,10))				# establishing the larger plot
	plt.subplots_adjust(bottom=0.05, top=0.99, left=0.4, right=0.9, hspace=0.02)		# triming the whitespace in between the subplots
	plt.tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)
	plt.xticks([])
	plt.yticks([])
	plt.title(title, fontsize=30)

	ax1 = fig.add_subplot(811)			# initalizing the subplot
	z1=np.array(BFE_bar['BFE_bottom'])		# creates an array from the y_bar dataframe
	z2=np.array(BFE_bar['BFE_top'])			# creates another array. These two arrays are compared to create the bar at the top of the plots.
	w1=np.array(BFE_bar['pers_bottom'])
	w2=np.array(BFE_bar['pers_top'])
	ax1.plot(BFE['mean'], label='mean')								# plots the mean columns of the dataframe.
	ax1.fill_between(BFE.index, BFE['bottom_perc'], BFE['top_perc'], alpha=0.2, label='$95^{th}$ percentile', color='indigo')	# type: ignore # fills the area between the confidence interval with a lighter shade
	ax1.fill_between(BFE_bar.index, BFE_bar['BFE_bottom'], BFE_bar['BFE_top'], where=z2>z1, alpha=1, label='ground truth', color='orange')												# type: ignore # creates a bar at the top of the plot indicating the positve part of the binary real data
	ax1.fill_between(BFE_bar.index, BFE_bar['pers_bottom'], BFE_bar['pers_top'], where=w2>w1, alpha=1, color='black', label='persistance')												# type: ignore # creates a bar at the top of the plot indicating the positve part of the binary real data
	ax1.margins(x=0)							# tightning the plot margins
	ax1.set_ylabel('BFE', fontsize='20')
	plt.legend()
	plt.yticks(fontsize='13')
	# ax1.xaxis.set_major_locator(ticker.NullLocator())
	ax1.set_xticklabels([], fontsize=0)

	ax2 = fig.add_subplot(812, sharex=ax1)
	z1=np.array(WNG_bar['WNG_bottom'])
	z2=np.array(WNG_bar['WNG_top'])
	w1=np.array(WNG_bar['pers_bottom'])
	w2=np.array(WNG_bar['pers_top'])
	ax2.plot(WNG.index, WNG['mean'])
	ax2.fill_between(WNG.index, WNG['bottom_perc'], WNG['top_perc'], alpha=0.2, color='indigo')  # type: ignore
	ax2.fill_between(WNG_bar.index, WNG_bar['WNG_bottom'], WNG_bar['WNG_top'], where=z2>z1, alpha=1, color='orange')  # type: ignore
	ax2.fill_between(WNG_bar.index, WNG_bar['pers_bottom'], WNG_bar['pers_top'], where=w2>w1, alpha=1, color='black')
	ax2.margins(x=0)
	ax2.set_ylabel('WNG', fontsize='20')
	plt.yticks(fontsize='13')
	ax2.set_xticklabels([], fontsize=0)			# adds the date to the bottom of the plot

	ax3 = fig.add_subplot(813, sharex=ax1)			# initalizing the subplot
	z1=np.array(LER_bar['LER_bottom'])		# creates an array from the y_bar dataframe
	z2=np.array(LER_bar['LER_top'])			# creates another array. These two arrays are compared to create the bar at the top of the plots.
	w1=np.array(LER_bar['pers_bottom'])
	w2=np.array(LER_bar['pers_top'])
	ax3.plot(LER['mean'])								# plots the mean columns of the dataframe.
	ax3.fill_between(LER.index, LER['bottom_perc'], LER['top_perc'], alpha=0.2, color='indigo')	# type: ignore # fills the area between the confidence interval with a lighter shade
	ax3.fill_between(LER_bar.index, LER_bar['LER_bottom'], LER_bar['LER_top'], where=z2>z1, alpha=1, color='orange')												# type: ignore # creates a bar at the top of the plot indicating the positve part of the binary real data
	ax3.fill_between(LER_bar.index, LER_bar['pers_bottom'], LER_bar['pers_top'], where=w2>w1, alpha=1, color='black')
	ax3.margins(x=0)							# tightning the plot margins
	ax3.set_ylabel('LER', fontsize='20')
	plt.yticks(fontsize='13')
	ax3.set_xticklabels([], fontsize=0)

	ax4 = fig.add_subplot(814, sharex=ax1)
	z1=np.array(ESK_bar['ESK_bottom'])
	z2=np.array(ESK_bar['ESK_top'])
	w1=np.array(ESK_bar['pers_bottom'])
	w2=np.array(ESK_bar['pers_top'])
	ax4.plot(ESK.index, ESK['mean'])
	ax4.fill_between(ESK.index, ESK['bottom_perc'], ESK['top_perc'], alpha=0.2, color='indigo')  # type: ignore
	ax4.fill_between(ESK_bar.index, ESK_bar['ESK_bottom'], ESK_bar['ESK_top'], where=z2>z1, alpha=1, color='orange')  # type: ignore
	ax4.fill_between(ESK_bar.index, ESK_bar['pers_bottom'], ESK_bar['pers_top'], where=w2>w1, alpha=1, color='black')
	ax4.margins(x=0)
	ax4.set_ylabel('ESK', fontsize='20')
	plt.yticks(fontsize='13')
	plt.xticks(fontsize=15)
	ax4.set_xticklabels([], fontsize=0)

	ax5 = fig.add_subplot(815, sharex=ax1)			# initalizing the subplot
	z1=np.array(STJ_bar['STJ_bottom'])		# creates an array from the y_bar dataframe
	z2=np.array(STJ_bar['STJ_top'])			# creates another array. These two arrays are compared to create the bar at the top of the plots.
	w1=np.array(STJ_bar['pers_bottom'])
	w2=np.array(STJ_bar['pers_top'])
	ax5.plot(STJ['mean'], label='mean')								# plots the mean columns of the dataframe.
	ax5.fill_between(STJ.index, STJ['bottom_perc'], STJ['top_perc'], alpha=0.2, label='$95^{th}$ percentile', color='indigo')	# type: ignore # fills the area between the confidence interval with a lighter shade
	ax5.fill_between(STJ_bar.index, STJ_bar['STJ_bottom'], STJ_bar['STJ_top'], where=z2>z1, alpha=1, label='ground truth', color='orange')												# type: ignore # creates a bar at the top of the plot indicating the positve part of the binary real data
	ax5.fill_between(STJ_bar.index, STJ_bar['pers_bottom'], STJ_bar['pers_top'], where=w2>w1, alpha=1, color='black', label='persistance')
	ax5.margins(x=0)							# tightning the plot margins
	ax5.set_ylabel('STJ', fontsize='20')
	plt.yticks(fontsize='13')
	ax5.set_xticklabels([], fontsize=0)

	ax6 = fig.add_subplot(816, sharex=ax1)
	z1=np.array(OTT_bar['OTT_bottom'])
	z2=np.array(OTT_bar['OTT_top'])
	w1=np.array(OTT_bar['pers_bottom'])
	w2=np.array(OTT_bar['pers_top'])
	ax6.plot(OTT['mean'])
	ax6.fill_between(OTT.index, OTT['bottom_perc'], OTT['top_perc'], alpha=0.2, color='indigo')  # type: ignore
	ax6.fill_between(OTT_bar.index, OTT_bar['OTT_bottom'], OTT_bar['OTT_top'], where=z2>z1, alpha=1, color='orange')  # type: ignore
	ax6.fill_between(OTT_bar.index, OTT_bar['pers_bottom'], OTT_bar['pers_top'], where=w2>w1, alpha=1, color='black')
	ax6.margins(x=0)
	ax6.set_ylabel('OTT', fontsize='20')
	plt.yticks(fontsize='13')
	plt.xticks(fontsize=5)
	ax6.set_xticklabels([], fontsize=0)

	ax7 = fig.add_subplot(817, sharex=ax1)
	z1=np.array(NEW_bar['NEW_bottom'])
	z2=np.array(NEW_bar['NEW_top'])
	w1=np.array(NEW_bar['pers_bottom'])
	w2=np.array(NEW_bar['pers_top'])
	ax7.plot(NEW['mean'])
	ax7.fill_between(NEW.index, NEW['bottom_perc'], NEW['top_perc'], alpha=0.2, color='indigo')  # type: ignore
	ax7.fill_between(NEW_bar.index, NEW_bar['NEW_bottom'], NEW_bar['NEW_top'], where=z2>z1, alpha=1, color='orange')  # type: ignore
	ax7.fill_between(NEW_bar.index, NEW_bar['pers_bottom'], NEW_bar['pers_top'], where=w2>w1, alpha=1, color='black')
	ax7.margins(x=0)
	ax7.set_ylabel('NEW', fontsize='20')
	plt.yticks(fontsize='13')
	plt.xticks(fontsize=5)
	ax7.set_xticklabels([], fontsize=0)

	ax8 = fig.add_subplot(818, sharex=ax1)
	z1=np.array(VIC_bar['VIC_bottom'])
	z2=np.array(VIC_bar['VIC_top'])
	w1=np.array(VIC_bar['pers_bottom'])
	w2=np.array(VIC_bar['pers_top'])
	ax8.plot(VIC['mean'])
	ax8.fill_between(VIC.index, VIC['bottom_perc'], VIC['top_perc'], alpha=0.2, color='indigo')  # type: ignore
	ax8.fill_between(VIC_bar.index, VIC_bar['VIC_bottom'], VIC_bar['VIC_top'], where=z2>z1, alpha=1, color='orange')  # type: ignore
	ax8.fill_between(VIC_bar.index, VIC_bar['pers_bottom'], VIC_bar['pers_top'], where=w2>w1, alpha=1, color='black')
	ax8.margins(x=0)
	ax8.set_ylabel('VIC', fontsize='20')
	plt.yticks(fontsize='13')
	plt.xticks(fontsize=15)
	ax8.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n %H:%M'))			# adds the date to the bottom of the plot

	plt.savefig('plots/k_fold_{0}_storm.png'.format(storm), bbox_inches='tight')		# saves the plot


def main():

	print('Entering main...')

	results_dict = load_stats()
	metrics_dict = sorting_metrics(results_dict, CONFIG['stations'], CONFIG['metrics'], len(CONFIG['test_storm_stime']))

	plot_metrics(metrics_dict, CONFIG['stations'], CONFIG['metrics'])
	plot_total_metrics(metrics_dict, CONFIG['stations'], metrics=['AUC', 'HSS'])

	for station in CONFIG['stations']:
		plot_precision_recall(results_dict, station, CONFIG['plot_titles'])
		reliability_plots(results_dict, station, CONFIG['splits'], CONFIG['plot_titles'])

	for i, title, stime, etime in zip(range(len(CONFIG['test_storm_stime'])), CONFIG['plot_titles'], CONFIG['test_storm_stime'], CONFIG['test_storm_etime']):		# looping through all of the relevent lists to plots the model outputs
		plot_model_outputs(results_dict, i, CONFIG['splits'], title, stime, etime)


if __name__ == '__main__':

	main()

	print('It ran. Good job!')





