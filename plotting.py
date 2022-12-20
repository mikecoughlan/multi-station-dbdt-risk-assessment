##########################################################################################
#
#	multi-station-dbdt-risk-assessment/plotting.py
#
#	Takes the results dictonaries and files and creates plots using matplotlib to
# 	display the results. Saves the plots to the plots directory.
#
#
##########################################################################################

import json
import pickle

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc

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


def load_stats():
	'''
	Loading the results dictonary.

	Returns:
		dict: contains the metric and model results using the stations as the keys
	'''

	with open('outputs/stations_results_dict.pkl', 'rb') as f:
		stations_dict = pickle.load(f)

	return stations_dict


def sorting_metrics(results_dict, stations, metrics, length, sw=False):
	'''
	Sorting the metrics to make the plotting easier. Metrics are
	stored for both each storm and for all storms combined.

	Args:
		results_dict (dict): dictonary containing results.
		stations (str or list of strs): 3 diges code(s) identifying stations being examined
		metrics (str or list of str): codes for metrics that will be sorted and plotted
		length (int): number of testing storms
		sw (bool): if False does the sorting for the combined model. If True does the
					sorting for just the SW only models. Defaults to False.

	Returns:
		dict: contains just the metric scores that will be used for plotting
	'''

	# Initializes a dictfor keeping the sorted metrics
	metrics_dict = {}

	# looping through the metrics being examined
	for metric in metrics:
		metric_dict = {}
		for i in range(length):
			df = pd.DataFrame()
			for station in stations:
				if not sw:
					df[station] = results_dict[station]['storm_{0}'.format(i)]['metrics'][metric]
				else:
					df[station] = results_dict[station]['storm_{0}'.format(i)]['sw_metrics'][metric]

			''' transposing the metric df from the storm so that the columns
				are now the median, and top and bottom percentiles. The station
				names are then the row names which makes the plotting easier'''
			metric_dict['storm_{0}'.format(i)] = df.T
		metrics_dict[metric] = metric_dict
		total_df = pd.DataFrame()
		for station in stations:
			''' adds the total results for all storms to the
				total metrics df using the station code as the column name'''
			if not sw:
				total_df[station] = results_dict[station]['total_metrics'][metric]
			else:
				total_df[station] = results_dict[station]['total_sw_metrics'][metric]
		# transposes the df so the station names are the row names
		metrics_dict['total_{0}'.format(metric)] = total_df.T
	hss, auc, rmse, bias = [], [], [], []

	# saving the persistance metric results
	for station in stations:
		hss.append(results_dict[station]['pers_HSS'][0])
		auc.append(results_dict[station]['pers_AUC'][0])
		rmse.append(results_dict[station]['pers_RMSE'][0])
		bias.append(results_dict[station]['pers_BIAS'][0])
	metrics_dict['pers_HSS'] = hss
	metrics_dict['pers_AUC'] = auc
	metrics_dict['pers_RMSE'] = rmse
	metrics_dict['pers_BIAS'] = bias

	return metrics_dict



def plot_metrics(metrics_dict, stations, metrics=['HSS', 'AUC', 'RMSE'], sw=False):
	'''
	Plotting function for the storm seperated metrics. Makes a unique
	plot for each metric and displays the results for each station bunched
	together on the x-axis. Saves plots to plots directory.

	Args:
		metrics_dict (dict): contains the metric results in a format that makes the plotting simplier
		stations (str or list of str): stations to plot metric results for
		metrics (str or list of str): list of metrics being plotted. Defaults to ['HSS', 'AUC', 'RMSE']
	'''

	for metric in metrics:

		fig = plt.figure(figsize=(10,7))													# establishing the figure
		plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, hspace=0.03)			# trimming the whitespace in the subplots

		# X = [5, 35, 65, 95, 125, 155, 185, 215]				# Used for labeling the x axis of the plots for each station.
		X = [5, 35, 65, 95]				# Used for labeling the x axis of the plots for each station.

		'''subtracting and adding from the main X array above to
			create slight speration in each of the storm metric results'''

		x0 = [(num-3.5) for num in X]
		x1 = [(num-2.5) for num in X]
		x2 = [(num-1.5) for num in X]
		x3 = [(num-0.5) for num in X]
		x4 = [(num+0.5) for num in X]
		x5 = [(num+1.5) for num in X]
		x6 = [(num+2.5) for num in X]
		x7 = [(num+3.5) for num in X]

		ax = fig.add_subplot(111)									# adding the subplot
		y0 = metrics_dict[metric]['storm_0']['mean'].to_numpy()		# defining the y center point
		ymax0 = metrics_dict[metric]['storm_0']['max'].to_numpy()	# defining the y upper bound
		ymin0 = metrics_dict[metric]['storm_0']['min'].to_numpy()	# defining the y lower bound

		# Adding the max and min arrays to the mean to make the error bars
		ymax0 = ymax0 - y0
		ymin0 = y0 - ymin0

		y1 = metrics_dict[metric]['storm_1']['mean'].to_numpy()
		ymax1 = metrics_dict[metric]['storm_1']['max'].to_numpy()
		ymin1 = metrics_dict[metric]['storm_1']['min'].to_numpy()

		ymax1 = ymax1 - y1
		ymin1 = y1 - ymin1

		y2 = metrics_dict[metric]['storm_2']['mean'].to_numpy()
		ymax2 = metrics_dict[metric]['storm_2']['max'].to_numpy()
		ymin2 = metrics_dict[metric]['storm_2']['min'].to_numpy()

		ymax2 = ymax2 - y2
		ymin2 = y2 - ymin2

		y3 = metrics_dict[metric]['storm_3']['mean'].to_numpy()
		ymax3 = metrics_dict[metric]['storm_3']['max'].to_numpy()
		ymin3 = metrics_dict[metric]['storm_3']['min'].to_numpy()

		ymax3 = ymax3 - y3
		ymin3 = y3 - ymin3

		y4 = metrics_dict[metric]['storm_4']['mean'].to_numpy()
		ymax4 = metrics_dict[metric]['storm_4']['max'].to_numpy()
		ymin4 = metrics_dict[metric]['storm_4']['min'].to_numpy()

		ymax4 = ymax4 - y4
		ymin4 = y4 - ymin4

		y5 = metrics_dict[metric]['storm_5']['mean'].to_numpy()
		ymax5 = metrics_dict[metric]['storm_5']['max'].to_numpy()
		ymin5 = metrics_dict[metric]['storm_5']['min'].to_numpy()

		ymax5 = ymax5 - y5
		ymin5 = y5 - ymin5

		y6 = metrics_dict[metric]['storm_6']['mean'].to_numpy()
		ymax6 = metrics_dict[metric]['storm_6']['max'].to_numpy()
		ymin6 = metrics_dict[metric]['storm_6']['min'].to_numpy()

		ymax6 = ymax6 - y6
		ymin6 = y6 - ymin6

		y7 = metrics_dict[metric]['storm_7']['mean'].to_numpy()
		ymax7 = metrics_dict[metric]['storm_7']['max'].to_numpy()
		ymin7 = metrics_dict[metric]['storm_7']['min'].to_numpy()

		ymax7 = ymax7 - y7
		ymin7 = y7 - ymin7

		# Titling the plots
		plt.title(metric, fontsize='20')

		# plotting the center point with the error bars. list order is important in the y array so it cooresponds to the x label
		ax.errorbar(x0, y0, yerr=[ymin0, ymax0], fmt='.k', color='blue', label='Mar 2001', elinewidth=2, markersize=15, capsize=4, capthick=2)
		ax.errorbar(x1, y1, yerr=[ymin1, ymax1], fmt='.k', color='orange', label='Sep 2001', elinewidth=2, markersize=15, capsize=4, capthick=2)
		ax.errorbar(x2, y2, yerr=[ymin2, ymax2], fmt='.k', color='green', label='May 2005', elinewidth=2, markersize=15, capsize=4, capthick=2)
		ax.errorbar(x3, y3, yerr=[ymin3, ymax3], fmt='.k', color='red', label='Sep 2005', elinewidth=2, markersize=15, capsize=4, capthick=2)
		ax.errorbar(x4, y4, yerr=[ymin4, ymax4], fmt='.k', color='purple', label='Dec 2006', elinewidth=2, markersize=15, capsize=4, capthick=2)
		ax.errorbar(x5, y5, yerr=[ymin5, ymax5], fmt='.k', color='brown', label='Apr 2010', elinewidth=2, markersize=15, capsize=4, capthick=2)
		ax.errorbar(x6, y6, yerr=[ymin6, ymax6], fmt='.k', color='pink', label='Aug 2011', elinewidth=2, markersize=15, capsize=4, capthick=2)
		ax.errorbar(x7, y7, yerr=[ymin7, ymax7], fmt='.k', color='black', label='Mar 2015', elinewidth=2, markersize=15, capsize=4, capthick=2)
		plt.axhline(0, color='black')
		plt.xlabel('Stations', fontsize='15')		# adding the label on the x axis label
		plt.ylabel(metric, fontsize='15')			# adding the y axis label
		plt.xticks(X, stations, fontsize='15')		# adding ticks to the points on the x axis
		plt.yticks(fontsize='15')					# making the y ticks a bit bigger. They're a bit more important
		plt.legend(fontsize='10')

		if not sw:
			plt.savefig('plots/{0}_version_{1}.png'.format(metric, CONFIG['version']), bbox_inches='tight')
		else:
			plt.savefig('plots/{0}_version_{1}_sw_models.png'.format(metric, CONFIG['version']), bbox_inches='tight')



def plot_total_metrics(metrics_dict, sw_metrics_dict, stations, metrics=['HSS', 'AUC', 'RMSE']):
	'''
	Plotting the total metrics instead of the individual storm metrics. Very similar
	to the plot_metrics function above. Seperated the functions to make it easier to
	keep track and specify plot options.

	Args:
		metrics_dict (dict): contains the metric results in a format that makes the plotting simplier
		stations (str or list of str): stations to plot metric results for
		metrics (str or list of str): list of metrics being plotted. Defaults to ['HSS', 'AUC', 'RMSE'].
	'''

	fig = plt.figure(figsize=(10,7))													# establishing the figure
	plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, hspace=0.03)			# trimming the whitespace in the subplots

	# X = [5, 15, 25, 35, 45, 55, 65, 75]
	X = [5, 10, 15, 20]

	x0 = [(num-1.5) for num in X]
	x1 = [(num+1.5) for num in X]


	ax = fig.add_subplot(111)					# adding the subplot
	plt.title('{0} and {1} Scores'.format(metrics[0], metrics[1]), fontsize='20')		# titling the plot

	# specifying plot colors
	bar_colors = ['blue', 'darkred']
	sw_bar_colors = ['tab:blue', 'chocolate']
	persistance_colors = ['deepskyblue', 'orange']
	for metric, color0, color1, color_sw in zip(metrics, bar_colors, persistance_colors, sw_bar_colors):
		y0 = metrics_dict['total_{0}'.format(metric)]['mean'].to_numpy()		# defining the y center point
		ymax0 = metrics_dict['total_{0}'.format(metric)]['max'].to_numpy()	# defining the y upper bound
		ymin0 = metrics_dict['total_{0}'.format(metric)]['min'].to_numpy()	# defining the y lower bound

		sw_y0 = sw_metrics_dict['total_{0}'.format(metric)]['mean'].to_numpy()		# defining the y center point
		sw_ymax0 = sw_metrics_dict['total_{0}'.format(metric)]['max'].to_numpy()	# defining the y upper bound
		sw_ymin0 = sw_metrics_dict['total_{0}'.format(metric)]['min'].to_numpy()	# defining the y lower bound

		ymax0 = ymax0 - y0
		ymin0 = y0 - ymin0

		sw_ymax0 = sw_ymax0 - sw_y0
		sw_ymin0 = sw_y0 - sw_ymin0

		ax.errorbar(X, y0, yerr=[ymin0, ymax0], fmt='.', color=color0, label='combined {0}'.format(metric), elinewidth=3, markersize=15, capsize=4, capthick=3)		# plotting the center point with the error bars. list order is important in the y array so it cooresponds to the x label
		ax.errorbar(X, sw_y0, yerr=[sw_ymin0, sw_ymax0], fmt='.', color=color_sw, label='sw {0}'.format(metric), elinewidth=3, markersize=15, capsize=4, capthick=3)		# plotting the center point with the error bars. list order is important in the y array so it cooresponds to the x label
		ax.scatter(X, metrics_dict['pers_{0}'.format(metric)], marker='^', color=color1, label='pers {0}'.format(metric), s=150)

	plt.xlabel('Stations', fontsize='15')		# adding the label on the x axis label
	plt.ylabel('Score', fontsize='15')			# adding the y axis label
	plt.xticks(X, stations, fontsize='15')		# adding ticks to the points on the x axis
	plt.yticks(fontsize='15')					# making the y ticks a bit bigger. They're a bit more important
	plt.legend(fontsize='10')

	plt.savefig('plots/{0}_{1}_metrics_total.png'.format(metrics[0], metrics[1]), bbox_inches='tight')


def prep_k_fold_results(df, splits):
	'''
	Prepares the data from the shuffled k-fold splits for plotting and examination.
	Creates a dataframe that stores the upper and lower percentile bounds for the plotting.

	Args:
		df (pd.dataframe): Dataframe from a particular storm that is being examined.
						Includes the real data and the predicted data for each split model.
		splits (int): integer number of split models that have been trained. Used here for indexing.

	Returns:
		pd.dataframe: dataframe with the ground truth data, persistance,
						top and bottom percentiles, and the mean model outputs.
						Used for ploitting the model outputs.
	'''

	# initalizes the new dataframe
	newdf = pd.DataFrame()

	# looping through the splits
	for split in range(splits):

		# grabs the predicted data for the threshold examined and puts them in the newdf
		newdf['split_{0}'.format(split)] = df['predicted_split_{0}'.format(split)]

	# calculating the mean, nad percentiles of the model outputs
	mean = newdf.mean(axis=1)
	top_perc = newdf.quantile(0.975, axis=1)
	bottom_perc = newdf.quantile(0.025, axis=1)

	# assigning relevant columns to the new dataframe
	newdf['cross'] = df['crossing']
	newdf['persistance'] = df['persistance']
	newdf['mean'] = mean
	newdf['top_perc'] = top_perc
	newdf['bottom_perc'] = bottom_perc

	# establishes a date column that will be used as the index
	newdf['date'] = df.index

	# establishes the datetime index from the date column
	pd.to_datetime(newdf['date'], format='%Y-%m-%d %H:%M:%S')
	newdf.reset_index(drop=True, inplace=True)
	newdf.set_index('date', inplace=True, drop=True)
	newdf.index = pd.to_datetime(newdf.index)

	return newdf



def sorting_PR(results_dict, station):
	'''
	Segments the precision recall curves and AUC scores for plotting simplicity.

	Args:
		results_dict (dict): dictonary containing precision and recall arrays segmented by testing storm
		station (str): 3 diget station code indicating which station's reults are being examined.

	Returns:
		dict: dictonary containing the precison and recall arrays and the auc scores.
	'''

	PR_dict = {}
	# 8 is the number of testing storms being examined.
	for i in range(8):
		PR_dict['storm_{0}'.format(i)] = {}
		df = results_dict[station]['storm_{0}'.format(i)]['precision_recall']

		# try-except here is in case any station is not able to make a prediction for a stomr because of missing data etc.
		try:
			PR_dict['storm_{0}'.format(i)]['prec'] = df['prec'].to_numpy()
			PR_dict['storm_{0}'.format(i)]['rec'] = df['rec'].to_numpy()
			PR_dict['storm_{0}'.format(i)]['auc'] = auc(df['rec'].to_numpy(), df['prec'].to_numpy()).round(decimals=3)
		except KeyError:
			print('skipping this storm-station combo')

	return PR_dict


def plot_precision_recall(results_dict, station, plot_titles):
	'''
	Function for plotting the precision recall curves for each station and storm.
	Outputs one plot for each station containing the PR curves for each storm and
	labeling them with the AUC scores.

	Args:
		results_dict (dict): dictonary containing precision and recall arrays segmented by testing storm
		station (str): 3 diget station code indicating which station's reults are being examined.
		plot_titles (str or list of str): list of testing storms used for labeling each of the PR curves
	'''

	PR_dict = sorting_PR(results_dict, station)

	fig = plt.figure(figsize=(10,7))
	plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, hspace=0.03)

	ax = fig.add_subplot(111)
	plt.title('Precision-Recall Curves for {0} Station'.format(station), fontsize='20')

	# looping through the plot titles using them to plot each storm's PR curve for this station
	for i, title in enumerate(plot_titles):

		# try-except is here in case station doesn't have data for a particular storm
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
	'''
	Plotting the reliability curves of our model results. Saves the plots in the "plots" directory.

	Args:
		results_dict (dict): dictonary containing precision and recall arrays segmented by testing storm
		station (str): 3 diget station code indicating which station's reults are being examined.
		splits (int): number of shuffeled k-fold splits performed. Also the number of models created for each station
		plot_titles (str or list of str): list of testing storms used for labeling each of the PR curves
	'''
	''''''

	# getting the properly formatted results for each storm
	storm0 = prep_k_fold_results(results_dict[station]['storm_0']['raw_results'], splits)
	storm1 = prep_k_fold_results(results_dict[station]['storm_1']['raw_results'], splits)
	storm2 = prep_k_fold_results(results_dict[station]['storm_2']['raw_results'], splits)
	storm3 = prep_k_fold_results(results_dict[station]['storm_3']['raw_results'], splits)
	storm4 = prep_k_fold_results(results_dict[station]['storm_4']['raw_results'], splits)
	storm5 = prep_k_fold_results(results_dict[station]['storm_5']['raw_results'], splits)
	storm6 = prep_k_fold_results(results_dict[station]['storm_6']['raw_results'], splits)
	storm7 = prep_k_fold_results(results_dict[station]['storm_7']['raw_results'], splits)

	# putting them together into a lsit
	newdfs = [storm0, storm1, storm2, storm3, storm4, storm5, storm6, storm7]

	# concatingating the dataframes together
	newdf = pd.concat(newdfs, axis=0)

	# initlaizing the plot
	fig = plt.figure(figsize=(20,18))
	plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, hspace=0.03)

	ax = fig.add_subplot(111)
	ax.set_title('{0} Reliability Plot'.format(station), fontsize=30)

	#drawing the line of "perfect" reliability
	plt.plot([0, 1], [0, 1], 'xkcd:black')

	# calculating the calibration curve
	true, pred = calibration_curve(newdf['cross'], newdf['mean'], n_bins=10)
	plt.plot(pred, true, marker='.')
	ax.set_xlabel('Predicted Probability', fontsize=20)
	ax.set_ylabel('Observed Probability', fontsize=20)
	plt.yticks(fontsize=15)
	plt.xticks(fontsize=15)
	ax.margins(x=0, y=0)

	plt.savefig('plots/{0}_reliability_plot.png'.format(station))


def plot_model_outputs(results_dict, storm, splits, title):
	'''
	Plots all of the model output results with confidence intervals. Plots the ground truth
	data and the persistance model at the top of each station's plot for comparison Saves the
	plots in the "plots" directory.

	Args:
		results_dict (dict): dictonary containing precision and recall arrays segmented by testing storm
		storm (int): integer code identifying which storm is being plotted.
		splits (int): number of shuffeled k-fold splits performed. Also the number of models created for each station
		title (str): plot title
	'''

	# calling the prep_k_fold function for each threshold. Should probably find a better way to do this.
	OTT = prep_k_fold_results(results_dict['OTT']['storm_{0}'.format(storm)]['raw_results'], splits)
	# BFE = prep_k_fold_results(results_dict['BFE']['storm_{0}'.format(storm)]['raw_results'], splits)
	# WNG = prep_k_fold_results(results_dict['WNG']['storm_{0}'.format(storm)]['raw_results'], splits)
	STJ = prep_k_fold_results(results_dict['STJ']['storm_{0}'.format(storm)]['raw_results'], splits)
	# NEW = prep_k_fold_results(results_dict['NEW']['storm_{0}'.format(storm)]['raw_results'], splits)
	# VIC = prep_k_fold_results(results_dict['VIC']['storm_{0}'.format(storm)]['raw_results'], splits)
	ESK = prep_k_fold_results(results_dict['ESK']['storm_{0}'.format(storm)]['raw_results'], splits)
	LER = prep_k_fold_results(results_dict['LER']['storm_{0}'.format(storm)]['raw_results'], splits)

	OTT_sw = prep_k_fold_results(results_dict['OTT']['storm_{0}'.format(storm)]['sw_results'], splits)
	STJ_sw = prep_k_fold_results(results_dict['STJ']['storm_{0}'.format(storm)]['sw_results'], splits)
	LER_sw = prep_k_fold_results(results_dict['LER']['storm_{0}'.format(storm)]['sw_results'], splits)
	ESK_sw = prep_k_fold_results(results_dict['ESK']['storm_{0}'.format(storm)]['sw_results'], splits)
	# NEW_sw = prep_k_fold_results(results_dict['NEW']['storm_{0}'.format(storm)]['sw_results'], splits)
	# VIC_sw = prep_k_fold_results(results_dict['VIC']['storm_{0}'.format(storm)]['sw_results'], splits)
	# BFE_sw = prep_k_fold_results(results_dict['BFE']['storm_{0}'.format(storm)]['sw_results'], splits)
	# WNG_sw = prep_k_fold_results(results_dict['WNG']['storm_{0}'.format(storm)]['sw_results'], splits)





	'''creats a new dataframe that will allow me to create a bar at the top of the
		plot to define the periods where the real, binary values have value 1. Does
		the same for the persistance models. Does this for each of the stations.'''

	OTT_bar = pd.DataFrame({'OTT_bottom':OTT['cross']*1.01,
							'OTT_top':OTT['cross']*1.06,
							'pers_bottom':OTT['persistance']*1.07,
							'pers_top':OTT['persistance']*1.12},
							index=OTT.index)
	# BFE_bar = pd.DataFrame({'BFE_bottom':BFE['cross']*1.01,
	# 						'BFE_top':BFE['cross']*1.06,
	# 						'pers_bottom':BFE['persistance']*1.07,
	# 						'pers_top':BFE['persistance']*1.12},
	# 						index=BFE.index)
	# WNG_bar = pd.DataFrame({'WNG_bottom':WNG['cross']*1.01,
	# 						'WNG_top':WNG['cross']*1.06,
	# 						'pers_bottom':WNG['persistance']*1.07,
	# 						'pers_top':WNG['persistance']*1.12},
	# 						index=WNG.index)
	STJ_bar = pd.DataFrame({'STJ_bottom':STJ['cross']*1.01,
							'STJ_top':STJ['cross']*1.06,
							'pers_bottom':STJ['persistance']*1.07,
							'pers_top':STJ['persistance']*1.12},
							index=STJ.index)
	# NEW_bar = pd.DataFrame({'NEW_bottom':NEW['cross']*1.01,
	# 						'NEW_top':NEW['cross']*1.06,
	# 						'pers_bottom':NEW['persistance']*1.07,
	# 						'pers_top':NEW['persistance']*1.12},
	# 						index=NEW.index)
	# VIC_bar = pd.DataFrame({'VIC_bottom':VIC['cross']*1.01,
	# 						'VIC_top':VIC['cross']*1.06,
	# 						'pers_bottom':VIC['persistance']*1.07,
	# 						'pers_top':VIC['persistance']*1.12},
	# 						index=VIC.index)
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

	# adds datetime index
	OTT_bar.index=pd.to_datetime(OTT_bar.index)
	# BFE_bar.index=pd.to_datetime(BFE_bar.index)
	# WNG_bar.index=pd.to_datetime(WNG_bar.index)
	STJ_bar.index=pd.to_datetime(STJ_bar.index)
	# NEW_bar.index=pd.to_datetime(NEW_bar.index)
	# VIC_bar.index=pd.to_datetime(VIC_bar.index)
	ESK_bar.index=pd.to_datetime(ESK_bar.index)
	LER_bar.index=pd.to_datetime(LER_bar.index)


	fig = plt.figure(figsize=(25,10))				# establishing the larger plot
	plt.subplots_adjust(bottom=0.05, top=0.99, left=0.4, right=0.9, hspace=0.02)		# triming the whitespace in between the subplots

	# removing the outer plot tick labels
	plt.tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)
	plt.xticks([])
	plt.yticks([])
	plt.title(title, fontsize=30)

	# ax1 = fig.add_subplot(811)			# initalizing the subplot

	# # creates an array from the y_bar dataframe
	# z1=np.array(BFE_bar['BFE_bottom'])

	# # creates another array. These two arrays are compared to create the bar at the top of the plots.
	# z2=np.array(BFE_bar['BFE_top'])

	# # This repeats the above for the persistance maodel
	# w1=np.array(BFE_bar['pers_bottom'])
	# w2=np.array(BFE_bar['pers_top'])

	# # plots the mean columns of the dataframe.
	# ax1.plot(BFE['mean'], label='mean')

	# # fills the area between the confidence interval with a lighter shade
	# ax1.fill_between(BFE.index, BFE['bottom_perc'], BFE['top_perc'], alpha=0.2, label='$95^{th}$ percentile', color='indigo')

	# # creates a bar at the top of the plot indicating the positve part of the binary real data
	# ax1.fill_between(BFE_bar.index, BFE_bar['BFE_bottom'], BFE_bar['BFE_top'], where=z2>z1, alpha=1, label='ground truth', color='orange')
	# ax1.fill_between(BFE_bar.index, BFE_bar['pers_bottom'], BFE_bar['pers_top'], where=w2>w1, alpha=1, color='black', label='persistance')

	# # tightning the plot margins
	# ax1.margins(x=0)
	# ax1.set_ylabel('BFE', fontsize='20')
	# plt.legend()
	# plt.yticks(fontsize='13')

	# # clears the x-axis labels. Won't be seen anyway as they will be covered by next subplot.
	# ax1.set_xticklabels([], fontsize=0)

	# # Repeats the above for the other 7 stations
	# ax2 = fig.add_subplot(812, sharex=ax1)
	# z1=np.array(WNG_bar['WNG_bottom'])
	# z2=np.array(WNG_bar['WNG_top'])
	# w1=np.array(WNG_bar['pers_bottom'])
	# w2=np.array(WNG_bar['pers_top'])
	# ax2.plot(WNG.index, WNG['mean'])
	# ax2.fill_between(WNG.index, WNG['bottom_perc'], WNG['top_perc'], alpha=0.2, color='indigo')
	# ax2.fill_between(WNG_bar.index, WNG_bar['WNG_bottom'], WNG_bar['WNG_top'], where=z2>z1, alpha=1, color='orange')
	# ax2.fill_between(WNG_bar.index, WNG_bar['pers_bottom'], WNG_bar['pers_top'], where=w2>w1, alpha=1, color='black')
	# ax2.margins(x=0)
	# ax2.set_ylabel('WNG', fontsize='20')
	# plt.yticks(fontsize='13')
	# ax2.set_xticklabels([], fontsize=0)

	ax3 = fig.add_subplot(411)
	z1=np.array(LER_bar['LER_bottom'])
	z2=np.array(LER_bar['LER_top'])
	w1=np.array(LER_bar['pers_bottom'])
	w2=np.array(LER_bar['pers_top'])
	ax3.plot(LER['mean'])
	ax3.plot(LER_sw['mean'], label='sw mean', color='red')
	ax3.fill_between(LER.index, LER['bottom_perc'], LER['top_perc'], alpha=0.3)
	ax3.fill_between(LER_sw.index, LER_sw['bottom_perc'], LER_sw['top_perc'], alpha=0.3, label='SW only $95^{th}$ percentile', color='red')
	ax3.fill_between(LER_bar.index, LER_bar['LER_bottom'], LER_bar['LER_top'], where=z2>z1, alpha=1, color='tab:green')
	ax3.fill_between(LER_bar.index, LER_bar['pers_bottom'], LER_bar['pers_top'], where=w2>w1, alpha=1, color='black')
	ax3.margins(x=0)
	ax3.set_ylabel('LER', fontsize='20')
	plt.yticks(fontsize='13')
	ax3.set_xticklabels([], fontsize=0)

	ax4 = fig.add_subplot(412, sharex=ax3)
	z1=np.array(ESK_bar['ESK_bottom'])
	z2=np.array(ESK_bar['ESK_top'])
	w1=np.array(ESK_bar['pers_bottom'])
	w2=np.array(ESK_bar['pers_top'])
	ax4.plot(ESK.index, ESK['mean'])
	ax4.plot(ESK_sw['mean'], label='sw mean', color='red')
	ax4.fill_between(ESK.index, ESK['bottom_perc'], ESK['top_perc'], alpha=0.3)
	ax4.fill_between(ESK_sw.index, ESK_sw['bottom_perc'], ESK_sw['top_perc'], alpha=0.3, label='SW only $95^{th}$ percentile', color='red')
	ax4.fill_between(ESK_bar.index, ESK_bar['ESK_bottom'], ESK_bar['ESK_top'], where=z2>z1, alpha=1, color='tab:green')
	ax4.fill_between(ESK_bar.index, ESK_bar['pers_bottom'], ESK_bar['pers_top'], where=w2>w1, alpha=1, color='black')
	ax4.margins(x=0)
	ax4.set_ylabel('ESK', fontsize='20')
	plt.yticks(fontsize='13')
	plt.xticks(fontsize=15)
	ax4.set_xticklabels([], fontsize=0)

	ax5 = fig.add_subplot(413, sharex=ax3)
	z1=np.array(STJ_bar['STJ_bottom'])
	z2=np.array(STJ_bar['STJ_top'])
	w1=np.array(STJ_bar['pers_bottom'])
	w2=np.array(STJ_bar['pers_top'])
	ax5.plot(STJ['mean'], label='mean')
	ax5.plot(STJ_sw['mean'], label='mean', color='red')
	ax5.fill_between(STJ.index, STJ['bottom_perc'], STJ['top_perc'], alpha=0.3, label='$95^{th}$ percentile')
	ax5.fill_between(STJ_sw.index, STJ_sw['bottom_perc'], STJ_sw['top_perc'], alpha=0.3, label='SW only $95^{th}$ percentile', color='red')
	ax5.fill_between(STJ_bar.index, STJ_bar['STJ_bottom'], STJ_bar['STJ_top'], where=z2>z1, alpha=1, label='ground truth', color='tab:green')
	ax5.fill_between(STJ_bar.index, STJ_bar['pers_bottom'], STJ_bar['pers_top'], where=w2>w1, alpha=1, color='black', label='persistance')
	ax5.margins(x=0)
	ax5.set_ylabel('STJ', fontsize='20')
	plt.yticks(fontsize='13')
	ax5.set_xticklabels([], fontsize=0)

	ax6 = fig.add_subplot(414, sharex=ax3)
	z1=np.array(OTT_bar['OTT_bottom'])
	z2=np.array(OTT_bar['OTT_top'])
	w1=np.array(OTT_bar['pers_bottom'])
	w2=np.array(OTT_bar['pers_top'])
	ax6.plot(OTT['mean'])
	ax6.plot(OTT_sw['mean'], color='red')
	ax6.fill_between(OTT.index, OTT['bottom_perc'], OTT['top_perc'], alpha=0.3)
	ax6.fill_between(OTT_sw.index, OTT_sw['bottom_perc'], OTT_sw['top_perc'], alpha=0.3, color='red')
	ax6.fill_between(OTT_bar.index, OTT_bar['OTT_bottom'], OTT_bar['OTT_top'], where=z2>z1, alpha=1, color='tab:green')
	ax6.fill_between(OTT_bar.index, OTT_bar['pers_bottom'], OTT_bar['pers_top'], where=w2>w1, alpha=1, color='black')
	ax6.margins(x=0)
	ax6.set_ylabel('OTT', fontsize='20')
	plt.yticks(fontsize='13')
	plt.xticks(fontsize=15)
	# ax6.set_xticklabels([], fontsize=0)

	# ax7 = fig.add_subplot(817, sharex=ax1)
	# z1=np.array(NEW_bar['NEW_bottom'])
	# z2=np.array(NEW_bar['NEW_top'])
	# w1=np.array(NEW_bar['pers_bottom'])
	# w2=np.array(NEW_bar['pers_top'])
	# ax7.plot(NEW['mean'])
	# ax7.fill_between(NEW.index, NEW['bottom_perc'], NEW['top_perc'], alpha=0.2, color='indigo')
	# ax7.fill_between(NEW_bar.index, NEW_bar['NEW_bottom'], NEW_bar['NEW_top'], where=z2>z1, alpha=1, color='orange')
	# ax7.fill_between(NEW_bar.index, NEW_bar['pers_bottom'], NEW_bar['pers_top'], where=w2>w1, alpha=1, color='black')
	# ax7.margins(x=0)
	# ax7.set_ylabel('NEW', fontsize='20')
	# plt.yticks(fontsize='13')
	# plt.xticks(fontsize=5)
	# ax7.set_xticklabels([], fontsize=0)

	# ax8 = fig.add_subplot(818, sharex=ax1)
	# z1=np.array(VIC_bar['VIC_bottom'])
	# z2=np.array(VIC_bar['VIC_top'])
	# w1=np.array(VIC_bar['pers_bottom'])
	# w2=np.array(VIC_bar['pers_top'])
	# ax8.plot(VIC['mean'])
	# ax8.fill_between(VIC.index, VIC['bottom_perc'], VIC['top_perc'], alpha=0.2, color='indigo')
	# ax8.fill_between(VIC_bar.index, VIC_bar['VIC_bottom'], VIC_bar['VIC_top'], where=z2>z1, alpha=1, color='orange')
	# ax8.fill_between(VIC_bar.index, VIC_bar['pers_bottom'], VIC_bar['pers_top'], where=w2>w1, alpha=1, color='black')
	# ax8.margins(x=0)
	# ax8.set_ylabel('VIC', fontsize='20')
	# plt.yticks(fontsize='13')
	# plt.xticks(fontsize=15)

	# # adds the date to the bottom of the plot
	ax6.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n %H:%M'))

	plt.savefig('plots/{0}_storm.png'.format(storm), bbox_inches='tight')


def main():
	'''
	Pulls together all of the plotting functions.
	'''

	print('Entering main...')

	# loading the data
	results_dict = load_stats()

	# sorting the metrics
	metrics_dict = sorting_metrics(results_dict, CONFIG['stations'], CONFIG['metrics'], len(CONFIG['test_storm_stime']), sw=False)
	sw_metrics_dict = sorting_metrics(results_dict, CONFIG['stations'], CONFIG['metrics'], len(CONFIG['test_storm_stime']), sw=True)


	# Plotting the individual storm metrics and the total metrics for each station
	plot_metrics(metrics_dict, CONFIG['stations'], CONFIG['metrics'], sw=False)
	plot_metrics(metrics_dict, CONFIG['stations'], CONFIG['metrics'], sw=True)
	plot_total_metrics(metrics_dict, sw_metrics_dict, CONFIG['stations'], metrics=['AUC', 'HSS'])
	plot_total_metrics(metrics_dict, sw_metrics_dict, CONFIG['stations'], metrics=['RMSE', 'BIAS'])


	# Plotting the individual storm metrics and the total metrics for each station
	plot_metrics(sw_metrics_dict, CONFIG['stations'], CONFIG['metrics'])

	# plotting the precision recall curves and the reliability diagrams for each station
	for station in CONFIG['stations']:
		plot_precision_recall(results_dict, station, CONFIG['plot_titles'])
		reliability_plots(results_dict, station, CONFIG['splits'], CONFIG['plot_titles'])

	# getting the full model outputs for each testing storm
	for i, title, stime, etime in zip(range(len(CONFIG['test_storm_stime'])), CONFIG['plot_titles'], CONFIG['test_storm_stime'], CONFIG['test_storm_etime']):		# looping through all of the relevent lists to plots the model outputs
		plot_model_outputs(results_dict, i, CONFIG['splits'], title)
		plot_model_outputs(results_dict, i, CONFIG['splits'], title)


if __name__ == '__main__':

	main()

	print('It ran. Good job!')





