##############################################################################################################
#
#	multi-station-dbdt-risk-assessment/calculating_shap_values.py
#
#   Calculates the SHAP values for the solar wind and combined models for each station. 10 randomly chosen
# 	of the 100 split models have their resulting 2D SHAP value arrays averaged. The sum of the values across
# 	the time dimension producing only one value per input parameter. The total values are then normalized to
# 	a percentage of total SHAP values for that input array, giving a percent contribution for each parameter.
# 	Plots the results in a stackplot for the combined and the solar wind models. Also calculates the rolling
# 	average of these percent contributions to smooth the plots. This was not used in analysis.
#
##############################################################################################################



import datetime as dt
import gc
import json
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import tensorflow as tf
from matplotlib import colors
from tensorflow.keras.models import Sequential, load_model
from tqdm import tqdm

# loading config and specific model config files. Using them as dictonaries
with open('config.json', 'r') as con:
	CONFIG = json.load(con)


def main(station):

	# 10 random models to be taken and averaged. This is done to reduce computational
	# time and kept constant to take the same splits for each model. Numbers were
	# generated using a random number generator.

	splits = [2, 13, 19, 20, 39, 43, 54, 65, 72, 97]

	# storm numbers
	storms = [0,1,2,3,4,5,6,7]

	# Loading the training and testing data
	with open(f'../data/prepared_data/SW_only_{station}_train_dict.pkl', 'rb') as f:
		sw_train_dict = pickle.load(f)

	with open(f'../data/prepared_data/SW_only_{station}_test_dict.pkl', 'rb') as d:
		sw_test_dict = pickle.load(d)

	with open(f'../data/prepared_data/combined_{station}_train_dict.pkl', 'rb') as c:
		combined_train_dict = pickle.load(c)

	with open(f'../data/prepared_data/combined_{station}_test_dict.pkl', 'rb') as b:
		combined_test_dict = pickle.load(b)

	# Looping over the storms
	for storm in tqdm(storms):

		# Defining the highlighted boxes for the figures
		if storm == 4:
			shade_stimes = ['2006-12-14 16:00:00', '2006-12-15 04:45:00', '2006-12-15 15:15:00', '2006-12-15 21:00:00']
			shade_etimes = ['2006-12-14 22:00:00', '2006-12-15 14:00:00', '2006-12-15 16:45:00', '2006-12-15 23:30:00']

			# Defining the positions of the box labels
			ann_xpositions = ['2006-12-14 18:30:00', '2006-12-15 09:00:00', '2006-12-15 15:25:00', '2006-12-15 21:45:00']
			ann_xpositions = [dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in ann_xpositions]
			ann_ypositions = [-95, -95, -95, -95]
			ann_labels = ['A', 'B', 'C', 'D']

		elif storm == 7:
			shade_stimes = ['2015-03-17 04:05:00', '2015-03-17 08:45:00', '2015-03-17 23:00:00', '2015-03-18 05:00:00']
			shade_etimes = ['2015-03-17 05:30:00', '2015-03-17 10:15:00', '2015-03-18 03:00:00', '2015-03-19 03:00:00']
			ann_xpositions = ['2015-03-17 04:05:00', '2015-03-17 08:50:00', '2015-03-18 00:30:00', '2015-03-18 14:30:00']
			ann_xpositions = [dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in ann_xpositions]
			ann_ypositions = [-95, -95, -95, -95]
			ann_labels = ['A', 'B', 'C', 'D']

		# Empty boxes for storms not discussed in depth in paper
		else:
			shade_stimes, shade_etimes, ann_xpositions, ann_ypositions, ann_labels = [], [], [], [], []

		# Grabbing the test input arrays
		sw_storm = sw_test_dict[f'storm_{storm}']['Y']
		combined_storm = combined_test_dict[f'storm_{storm}']['Y']

		# Adding a dimension for the channels to conform to the models expected input dims
		sw_storm = sw_storm.reshape(sw_storm.shape[0], sw_storm.shape[1], sw_storm.shape[2], 1)
		combined_storm = combined_storm.reshape(combined_storm.shape[0], combined_storm.shape[1], combined_storm.shape[2], 1)

		# looping through the 10 different models
		for model_split in splits:

			# Checking to see if the SHAP values for the model have been calculated already
			if os.path.exists(f'outputs/shap_values/sw_values_{station}_storm_{storm}_split_{model_split}.pkl') \
				and os.path.exists(f'outputs/shap_values/combined_values_{station}_storm_{storm}_split_{model_split}.pkl'):
				continue

			# Loading the model
			sw_model = load_model(f'models/{station}/CNN_SW_only_split_{model_split}.h5')
			combined_model = load_model(f'models/{station}/CNN_version_5_split_{model_split}.h5')


			# reducing the amount of the training dataset used to find the shap values
			combined_xtrain = combined_train_dict['X']
			combined_xtrain = combined_xtrain.reshape((combined_xtrain.shape[0], combined_xtrain.shape[1], combined_xtrain.shape[2], 1))

			# SHAP documentation list 1000 background samples as being sufficient for accurate SHAP values
			combined_background = combined_xtrain[np.random.choice(combined_xtrain.shape[0], 1000, replace=False)]

			# initalizing the explainer for the combined model
			combined_explainer = shap.DeepExplainer(combined_model, combined_background)

			# Repeating the process for the solar wind model
			sw_xtrain = sw_train_dict['X']
			sw_xtrain = sw_xtrain.reshape((sw_xtrain.shape[0], sw_xtrain.shape[1], sw_xtrain.shape[2], 1))
			sw_background = sw_xtrain[np.random.choice(sw_xtrain.shape[0], 1000, replace=False)]

			sw_explainer = shap.DeepExplainer(sw_model, sw_background)

			# Calculating the SHAP values
			combined_shap_values_check = combined_explainer.shap_values(combined_storm, check_additivity=False)
			print('Finished Combined Shap values. Onto solar wind ones....')
			sw_shap_values_check = sw_explainer.shap_values(sw_storm, check_additivity=False)

			# Saving the SHAP value arrays
			with open(f'outputs/shap_values/sw_values_{station}_storm_{storm}_split_{model_split}.pkl', 'wb') as s:
				pickle.dump(sw_shap_values_check, s)
			with open(f'outputs/shap_values/combined_values_{station}_storm_{storm}_split_{model_split}.pkl', 'wb') as c:
				pickle.dump(combined_shap_values_check, c)

			# Freeing up memory
			gc.collect()

		# Re-loading the SHAP values
		solar, com = [], []
		for split in splits:
			with open(f'outputs/shap_values/sw_values_{station}_storm_{storm}_split_{split}.pkl', 'rb') as s:
				sw_shap_values_check = pickle.load(s)
			with open(f'outputs/shap_values/combined_values_{station}_storm_{storm}_split_{split}.pkl', 'rb') as c:
				combined_shap_values_check = pickle.load(c)

			# Adding the "crossing" arrays to a list
			solar.append(sw_shap_values_check[1])
			com.append(combined_shap_values_check[1])

		# Stacking the SHAP arrays for the 10 models together across a new axis
		sw_shap_values = np.stack(solar, axis=3)
		combined_shap_values = np.stack(com, axis=3)

		# Taking the mean of the arrays across the 10 models
		sw_shap_values_check = np.mean(sw_shap_values, axis=3)
		combined_shap_values_check = np.mean(combined_shap_values, axis=3)

		# Summing of the SHAP values across the time dimension
		sw_storm_condensed = np.sum(sw_shap_values_check, axis=1)
		combined_storm_condensed = np.sum(combined_shap_values_check, axis=1)

		# Reshaping the arrays
		sw_storm_condensed = sw_storm_condensed.reshape(sw_storm_condensed.shape[0], sw_storm_condensed.shape[1])
		combined_storm_condensed = combined_storm_condensed.reshape(combined_storm_condensed.shape[0], combined_storm_condensed.shape[1])

		# Listing the parameters
		sw_features = ["sinMLT", "cosMLT", "B_Total", "BY_GSM",
					"BZ_GSM", "Vx", "Vy", "Vz", "proton_density", "T"]
		combined_features = ["N", "E", "sinMLT", "cosMLT", "B_Total", "BY_GSM",
								"BZ_GSM", "Vx", "Vy", "Vz", "proton_density", "T",
								"AE_INDEX", "SZA", "dBHt", "B"]


		# Turning the arrays into a dataframe for plotting
		sw_df = pd.DataFrame(sw_storm_condensed, columns=sw_features)
		combined_df = pd.DataFrame(combined_storm_condensed, columns=combined_features)

		# Adding in the date time index
		sw_df['Date_UTC'] = sw_test_dict[f'storm_{storm}']['real_df']['Date_UTC']
		combined_df['Date_UTC'] = combined_test_dict[f'storm_{storm}']['real_df']['Date_UTC']

		sw_df.set_index('Date_UTC', inplace=True)
		combined_df.set_index('Date_UTC', inplace=True)

		# Changing the SHAP values into percentage contributions
		perc_sw_df = (sw_df.div(sw_df.abs().sum(axis=1), axis=0))*100
		perc_combined_df = (combined_df.div(combined_df.abs().sum(axis=1), axis=0))*100

		prec_sw_x = perc_sw_df.index
		prec_combined_x = perc_combined_df.index

		perc_sw_df.reset_index(drop=True, inplace=True)
		perc_combined_df.reset_index(drop=True, inplace=True)

		# Calculates a rolling average
		perc_sw_rolling = perc_sw_df.rolling(10).mean()
		perc_combined_rolling = perc_combined_df.rolling(10).mean()

		# Seperating the positive contributions from the negative for plotting
		perc_sw_pos_df = perc_sw_df.mask(perc_sw_df < 0, other=0)
		perc_sw_neg_df = perc_sw_df.mask(perc_sw_df > 0, other=0)
		perc_combined_pos_df = perc_combined_df.mask(perc_combined_df < 0, other=0)
		perc_combined_neg_df = perc_combined_df.mask(perc_combined_df > 0, other=0)

		perc_sw_pos_dict, perc_sw_neg_dict, perc_combined_pos_dict, perc_combined_neg_dict = {}, {}, {}, {}

		# Creating numpy arrays for each parameter
		for pos, neg in zip(perc_sw_pos_df, perc_sw_neg_df):
			perc_sw_pos_dict[pos] = perc_sw_pos_df[pos].to_numpy()
			perc_sw_neg_dict[neg] = perc_sw_neg_df[neg].to_numpy()

		for pos, neg in zip(perc_combined_pos_df, perc_combined_neg_df):
			perc_combined_pos_dict[pos] = perc_combined_pos_df[pos].to_numpy()
			perc_combined_neg_dict[neg] = perc_combined_neg_df[neg].to_numpy()

		# Reordering the parameters in the combined model so the colors of
		# each parameter are the same in both plots

		reordered_combined_model_parameters = ["sinMLT", "cosMLT", "B_Total", "BY_GSM",
												"BZ_GSM", "Vx", "Vy", "Vz", "proton_density", "T",
												"AE_INDEX", "SZA", "N", "E", "B", "dBHt"]

		# Changing the order in the dataframe
		perc_combined_rolling = perc_combined_rolling[reordered_combined_model_parameters]

		perc_combined_rolling['Date_UTC'] = prec_combined_x
		perc_sw_rolling['Date_UTC'] = prec_sw_x

		perc_combined_rolling.set_index('Date_UTC', inplace=True)
		perc_sw_rolling.set_index('Date_UTC', inplace=True)

		# Reording the parameters in the dictonaries
		perc_combined_pos_dict = {k : perc_combined_pos_dict[k] for k in reordered_combined_model_parameters}

		perc_combined_neg_dict = {k : perc_combined_neg_dict[k] for k in reordered_combined_model_parameters}

		# Defining the color pallet for the stack plots
		sw_colors = sns.color_palette('tab20', len(perc_sw_pos_dict.keys()))
		combined_colors = sns.color_palette('tab20', len(perc_combined_pos_dict.keys())+4)

		# removing the grey-scale colors
		for i in range(4):
			combined_colors.pop(-3)


		# Getting the real data for plotting the ground truth
		results_df = combined_test_dict[f'storm_{storm}']['real_df']
		results_df.set_index('Date_UTC', inplace=True)

		# getting the higest and lowest values to set highlighted area limits
		sw_line_max = perc_sw_rolling.max().max()
		combined_line_max = perc_combined_rolling.max().max()

		sw_line_min = perc_sw_rolling.min().min()
		combined_line_min = perc_combined_rolling.min().min()

		# Creating df for highlighting and ground truth bars
		bar = pd.DataFrame({'stack_bottom':results_df['crossing']*101,
							'stack_top':results_df['crossing']*106,
							'sw_line_bottom':results_df['crossing']*(sw_line_max),
							'sw_line_top':results_df['crossing']*(sw_line_max+(0.05*sw_line_max)),
							'combined_line_bottom':results_df['crossing']*(combined_line_max),
							'combined_line_top':results_df['crossing']*(combined_line_max+(0.05*combined_line_max))},
							index=results_df.index)

		bar.index=pd.to_datetime(bar.index)

		# creating array for plotting the ground truth bar
		w1=np.array(bar['stack_bottom'])
		w2=np.array(bar['stack_top'])

		# creating array for plotting the ground truth bar
		sw_z1=np.array(bar['sw_line_bottom'])
		sw_z2=np.array(bar['sw_line_top'])

		# creating array for plotting the ground truth bar
		com_z1=np.array(bar['combined_line_bottom'])
		com_z2=np.array(bar['combined_line_top'])

		# defining the parameter labels
		combined_labels = ["sinMLT", "cosMLT", "$\mathregular{B_{total}^{GSM}}$", "$\mathregular{B_{y}^{GSM}}$", "$\mathregular{B_{z}^{GSM}}$",
					"$\mathregular{V_{x}}$", "$\mathregular{V_{y}}$", "$\mathregular{V_{z}}$", "$\mathregular{\u03C1_{SW}}$", "T", "AE Index",
					"SZA", "$\mathregular{B_{N}}$", "$\mathregular{B_{E}}$", "$\mathregular{B_{H}}$", "dB/dt"]

		sw_labels = ["sinMLT", "cosMLT", "$\mathregular{B_{total}^{GSM}}$", "$\mathregular{B_{y}^{GSM}}$", "$\mathregular{B_{z}^{GSM}}$",
					"$\mathregular{V_{x}}$", "$\mathregular{V_{y}}$", "$\mathregular{V_{z}}$", "$\mathregular{\u03C1_{SW}}$", "T"]


		# Plotting
		fig = plt.figure(figsize=(20,17))

		ax1 = plt.subplot(111)
		ax1.set_title('Solar Wind Model')

		# Plotting the ground truth bar
		ax1.fill_between(bar.index, bar['stack_bottom'], bar['stack_top'], where=w2>w1, alpha=1, label='ground truth', color='black')

		highlighting_color = 'black'

		# Highlihting area of interest
		for s, e, xpos, ypos, lab in zip(shade_stimes, shade_etimes, ann_xpositions, ann_ypositions, ann_labels):
			ax1.fill_between(bar[s:e].index, -100, 106, alpha=0.15, color=highlighting_color)
			ax1.text(xpos, ypos, s=lab, color='black', fontsize=22)

		# Stacking the positive and negative percent contributions
		plt.stackplot(prec_sw_x, perc_sw_pos_dict.values(), labels=sw_labels, colors=sw_colors, alpha=1)
		plt.stackplot(prec_sw_x, perc_sw_neg_dict.values(), colors=sw_colors, alpha=1)
		ax1.margins(x=0, y=0)				# Tightning the plot margins
		plt.ylabel('Percent Contribution')

		# Placing the legend outside of the plot
		plt.legend(bbox_to_anchor=(1,1), loc='upper left')
		plt.axhline(0, color='black')

		plt.savefig(f'plots/shap/sw_percent_contribution_{station}_storm_{storm}.png')

		# Repeating the process for the combined model
		fig = plt.figure(figsize=(20,17))

		ax2 = plt.subplot(111)
		ax2.set_title('Combined Model')
		ax2.fill_between(bar.index, bar['stack_bottom'], bar['stack_top'], where=w2>w1, alpha=1, label='ground truth', color='black')
		# Highlihting area of interest
		for s, e, xpos, ypos, lab in zip(shade_stimes, shade_etimes, ann_xpositions, ann_ypositions, ann_labels):
			ax2.fill_between(bar[s:e].index, -100, 106, alpha=0.15, color=highlighting_color)
			ax2.text(xpos, ypos, s=lab, color='black', fontsize=22)
		plt.stackplot(prec_combined_x, perc_combined_pos_dict.values(), labels=combined_labels, colors=combined_colors, alpha=1)
		plt.stackplot(prec_combined_x, perc_combined_neg_dict.values(), colors=combined_colors, alpha=1)
		ax2.margins(x=0, y=0)
		plt.ylabel('Percent Contribution')
		plt.axhline(0, color='black')
		plt.legend(bbox_to_anchor=(1,1), loc='upper left')
		plt.savefig(f'plots/shap/combined_percent_contribution_{station}_storm_{storm}.png')


		# Plotting the same two as above but in the same figure
		fig = plt.figure(figsize=(20,17))

		ax1 = plt.subplot(211)
		ax1.set_title(f'{station} Solar Wind Model', fontsize=20)
		ax1.fill_between(bar.index, bar['stack_bottom'], bar['stack_top'], where=w2>w1, alpha=1, label='ground truth', color='black')

		highlighting_color = 'black'
		# Highlihting area of interest
		for s, e, xpos, ypos, lab in zip(shade_stimes, shade_etimes, ann_xpositions, ann_ypositions, ann_labels):
			ax1.fill_between(bar[s:e].index, -100, 106, alpha=0.15, color=highlighting_color)
			ax1.text(xpos, ypos, s=lab, color='black', fontsize=22)
		plt.stackplot(prec_sw_x, perc_sw_pos_dict.values(), labels=sw_labels, colors=sw_colors, alpha=1)
		plt.stackplot(prec_sw_x, perc_sw_neg_dict.values(), colors=sw_colors, alpha=1)
		ax1.margins(x=0, y=0)
		plt.ylabel('Percent Contribution')
		plt.legend(bbox_to_anchor=(1,1), loc='upper left')
		plt.axhline(0, color='black')

		ax2 = plt.subplot(212, sharex=ax1)
		ax2.set_title(f'{station} Combined Model', fontsize=20)
		ax2.fill_between(bar.index, bar['stack_bottom'], bar['stack_top'], where=w2>w1, alpha=1, label='ground truth', color='black')
		# Highlihting area of interest
		for s, e, xpos, ypos, lab in zip(shade_stimes, shade_etimes, ann_xpositions, ann_ypositions, ann_labels):
			ax2.fill_between(bar[s:e].index, -100, 106, alpha=0.15, color=highlighting_color)
			ax2.text(xpos, ypos, s=lab, color='black', fontsize=22)
		plt.stackplot(prec_combined_x, perc_combined_pos_dict.values(), labels=combined_labels, colors=combined_colors, alpha=1)
		plt.stackplot(prec_combined_x, perc_combined_neg_dict.values(), colors=combined_colors, alpha=1)
		ax2.margins(x=0, y=0)
		plt.ylabel('Percent Contribution')
		plt.axhline(0, color='black')
		plt.legend(bbox_to_anchor=(1,1), loc='upper left')
		plt.savefig(f'plots/shap/stacked_{station}_storm_{storm}.png', bbox_inches='tight')


		# Same as above but with the rolling averages
		fig = plt.figure(figsize=(20,17))

		ax1 = plt.subplot(111)
		ax1.set_title('Solar Wind Model')
		ax1.fill_between(bar.index, bar['sw_line_bottom'], bar['sw_line_top'], where=sw_z2>sw_z1, alpha=1, label='ground truth', color='black')
		# Highlihting area of interest
		for s, e, xpos, ypos, lab in zip(shade_stimes, shade_etimes, ann_xpositions, ann_ypositions, ann_labels):
			ax1.fill_between(bar[s:e].index, sw_line_min, (sw_line_max+(0.05*sw_line_max)), alpha=0.15, color=highlighting_color)
			ax1.text(xpos, ypos, s=lab, color='black', fontsize=22)
		for param, label, color in zip(perc_sw_rolling.columns, sw_labels, sw_colors):
			plt.plot(perc_sw_rolling[param], label=label, color=color)
		ax1.margins(x=0, y=0)
		plt.ylabel('Percent Contribution')
		plt.legend(bbox_to_anchor=(1,1), loc='upper left')
		plt.axhline(0, color='black')

		plt.savefig(f'plots/shap/rolling_line_sw_percent_contribution_{station}_storm_{storm}.png')


		fig = plt.figure(figsize=(20,17))

		ax2 = plt.subplot(111)
		ax2.set_title('Combined Model')
		ax2.fill_between(bar.index, bar['combined_line_bottom'], bar['combined_line_top'], where=com_z2>com_z1, alpha=1, label='ground truth', color='black')
		# Highlihting area of interest
		for s, e, xpos, ypos, lab in zip(shade_stimes, shade_etimes, ann_xpositions, ann_ypositions, ann_labels):
			ax2.fill_between(bar[s:e].index, combined_line_min, (combined_line_max+(0.05*combined_line_max)), alpha=0.15, color=highlighting_color)
			ax2.text(xpos, ypos, s=lab, color='black', fontsize=22)
		for param, label, color in zip(perc_combined_rolling.columns, combined_labels, combined_colors):
			plt.plot(perc_combined_rolling[param], label=label, color=color)
		ax2.margins(x=0, y=0)
		plt.ylabel('Percent Contribution')
		plt.axhline(0, color='black')
		plt.legend(bbox_to_anchor=(1,1), loc='upper left')

		plt.savefig(f'plots/shap/rolling_line_combined_percent_contribution_{station}_storm_{storm}.png')


if __name__ == '__main__':

	# Lists the stations
	stations = ["OTT", "BFE", "WNG", "LER", "ESK", "STJ", "NEW", "VIC"]
	for station in stations:
		print(f'Starting {station}....')
		main(station)
		print(f'Finished {station}')

	print('It ran. Good job!')









