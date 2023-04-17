##########################################################################################
#
#	multi-station-dbdt-risk-assessment/shap_feature_contributions.py
#
#	Plots the SHAP percentage contributions against the input parameter values. Does
# 	so by loading the pre-calculated SHAP values and normalizing the values to a
# 	percentage. Does this for all of the time history, so each value is a single input
#	not the sum over the time history. Plots the results for all of the 10 randomly
# 	choosen models for all 8 stations and all 8 test storms. THis can cause outliers
# 	to appear more significant than they are. Plots the results in a  2D histogram for
# 	evaluation.
#
#
##########################################################################################

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

# Randomly choosen model split numbers
splits = [2, 13, 19, 20, 39, 43, 54, 65, 72, 97]

# Station names, order is not important here because all results are combined
stations = ["OTT", "BFE", "WNG", "LER", "ESK", "STJ", "NEW", "VIC"]

# storm numbers
storms = [0,1,2,3,4,5,6,7]

# listing the input parameters for the solar wind and combined models
sw_features = ["sinMLT", "cosMLT", "B_Total", "BY_GSM",
					"BZ_GSM", "Vx", "Vy", "Vz", "proton_density", "T"]
combined_features = ["N", "E", "sinMLT", "cosMLT", "B_Total", "BY_GSM",
						"BZ_GSM", "Vx", "Vy", "Vz", "proton_density", "T",
						"AE_INDEX", "SZA", "dBHt", "B"]

# Initializing dictonaries that will be used for storing results
sw_feats, combined_feats = {}, {}


'''creating a dataframe for each feature to store the parameter values as
"input" and the percent contributions as "cont"'''

for feat in sw_features:
	sw_feats[feat] = pd.DataFrame({'input':[], 'cont':[]})
for feat in combined_features:
	combined_feats[feat] = pd.DataFrame({'input':[], 'cont':[]})

for station in stations:

	# Loading the data
	with open(f'../data/prepared_data/SW_only_{station}_test_dict.pkl', 'rb') as d:
		sw_test_dict = pickle.load(d)

	with open(f'../data/prepared_data/combined_{station}_test_dict.pkl', 'rb') as b:
		combined_test_dict = pickle.load(b)

	# Iterates over the storms
	for storm in storms:

		# Concatenating the unscaled array from the solar wind and combined test dictonary
		combined_unscaled = np.concatenate(combined_test_dict[f'storm_{storm}']['unscaled_array'], axis=0)
		sw_unscaled = np.concatenate(sw_test_dict[f'storm_{storm}']['unscaled_array'], axis=0)

		# Initializing lists for storing arrays
		solar_shap, com_shap = [], []

		# Looping through the 10 randomly chosen models
		for split in tqdm(splits):

			# Loading the pre-calculated SHAP values
			with open(f'outputs/shap_values/sw_values_{station}_storm_{storm}_split_{split}.pkl', 'rb') as s:
				sw_shap_values = pickle.load(s)
			with open(f'outputs/shap_values/combined_values_{station}_storm_{storm}_split_{split}.pkl', 'rb') as c:
				combined_shap_values = pickle.load(c)

			# Grabbing the "crossing" array from the SHAP arrays and reshaping it to remove the channels dimension
			sw_shap_values = sw_shap_values[1].reshape(sw_shap_values[1].shape[0], sw_shap_values[1].shape[1], sw_shap_values[1].shape[2])
			combined_shap_values = combined_shap_values[1].reshape(combined_shap_values[1].shape[0], combined_shap_values[1].shape[1], combined_shap_values[1].shape[2])

			# Summing up over each input array for normalization
			sw_sum = np.sum(np.absolute(sw_shap_values), axis=(1,2))
			combined_sum = np.sum(np.absolute(combined_shap_values), axis=(1,2))

			# Using the summed values to normalize each input array to a percent contribution (100% total)
			sw_shap_values = np.array([((sw_shap_values[i,:,:]/sw_sum[i])*100) for i in range(sw_shap_values.shape[0])])
			combined_shap_values = np.array([((combined_shap_values[i,:,:]/combined_sum[i])*100) for i in range(combined_shap_values.shape[0])])

			# Concatinates all of the input arrays together
			sw_shap_values = np.concatenate(sw_shap_values, axis=0)
			combined_shap_values = np.concatenate(combined_shap_values, axis=0)

			# Adding each parameter array to a dataframe and combining it with the dataframes for the other model results
			for i in range(sw_shap_values.shape[1]):
				temp = pd.DataFrame()
				temp['input'] = sw_unscaled[:,i]
				temp['cont'] = sw_shap_values[:,i]

				# Concats the results to the current dataframe and stores it in the dictonary using the parameter name as the key
				sw_feats[sw_features[i]] = pd.concat([sw_feats[sw_features[i]], temp], axis=0, ignore_index=True)

			# Repeats the above for the combined models
			for i in range(combined_shap_values.shape[1]):
				temp = pd.DataFrame()
				temp['input'] = combined_unscaled[:,i]
				temp['cont'] = combined_shap_values[:,i]

				combined_feats[combined_features[i]] = pd.concat([combined_feats[combined_features[i]], temp], axis=0, ignore_index=True)

# Cutting off some of the outlier values with very few entries for the sake of plotting
combined_feats['dBHt'] = combined_feats['dBHt'][combined_feats['dBHt']['input']<300]
combined_feats['B'] = combined_feats['B'][combined_feats['B']['input']<1750]
combined_feats['proton_density'] = combined_feats['proton_density'][combined_feats['proton_density']['input']<45]
combined_feats['E'] = combined_feats['E'][combined_feats['E']['input']>-395]

# Listing the features to plot and the plotting labels for the combined models
plotting_feature = ["B_Total", "BY_GSM", "BZ_GSM", "Vx", "Vy", "proton_density", "AE_INDEX", "B", "dBHt"]
titles = ["$\mathregular{B_{total}^{GSM}}$", "$\mathregular{B_{y}^{GSM}}$", "$\mathregular{B_{z}^{GSM}}$",
			"$\mathregular{V_{x}}$", "$\mathregular{V_{y}}$", "$\mathregular{\u03C1_{SW}}$", "AE Index",
			"$\mathregular{B_{H}}$", "dB/dt"]

# Initilizing the figures
fig = plt.figure(figsize=(20,13))

# looping over the listed features and creating a subplot for each
for i, feat in enumerate(plotting_feature):

	combined_feats[feat].dropna(inplace=True)
	ax = plt.subplot(3,3,i+1)

	plt.title(titles[i], fontsize=17)
	plt.hist2d(x=combined_feats[feat]['input'], y=combined_feats[feat]['cont'], bins=100, norm=colors.LogNorm(), cmap='magma')
	plt.axhline(0, color='black', linestyle='--')
	plt.colorbar()
	plt.ylim(-10,25)
	plt.ylabel('% Contribution')

plt.savefig('plots/shap/combined_feature_contributions.png', bbox_inches='tight')


# Listing the features to plot and the plotting labels for the solar wind models
sw_plotting_feature = ["B_Total", "BY_GSM", "BZ_GSM", "Vx", "Vy", "proton_density"]
titles = ["$\mathregular{B_{total}^{GSM}}$", "$\mathregular{B_{y}^{GSM}}$", "$\mathregular{B_{z}^{GSM}}$",
			"$\mathregular{V_{x}}$", "$\mathregular{V_{y}}$", "$\mathregular{\u03C1_{SW}}$"]
sw_feats['proton_density'] = sw_feats['proton_density'][sw_feats['proton_density']['input']<45]

# Initilizing the figures
fig = plt.figure(figsize=(20,13))

# looping over the listed features and creating a subplot for each
for i, feat in enumerate(sw_plotting_feature):

	sw_feats[feat].dropna(inplace=True)
	ax = plt.subplot(2,3,i+1)

	plt.title(titles[i], fontsize=17)
	plt.hist2d(x=sw_feats[feat]['input'], y=sw_feats[feat]['cont'], bins=100, norm=colors.LogNorm(), cmap='magma')
	plt.axhline(0, color='black', linestyle='--')
	plt.colorbar()
	plt.ylim(-10,25)
	plt.ylabel('% Contribution')

plt.savefig('plots/shap/sw_feature_contributions.png', bbox_inches='tight')