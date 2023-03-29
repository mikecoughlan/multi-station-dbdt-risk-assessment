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

splits = [2, 13, 19, 20, 39, 43, 54, 65, 72, 97]

stations = ["OTT", "BFE", "WNG", "LER", "ESK", "STJ", "NEW", "VIC"]

storms = [0,1,2,3,4,5,6,7]

sw_features = ["sinMLT", "cosMLT", "B_Total", "BY_GSM",
					"BZ_GSM", "Vx", "Vy", "Vz", "proton_density", "T"]
combined_features = ["N", "E", "sinMLT", "cosMLT", "B_Total", "BY_GSM",
						"BZ_GSM", "Vx", "Vy", "Vz", "proton_density", "T",
						"AE_INDEX", "SZA", "dBHt", "B"]

sw_feats, combined_feats = {}, {}

for feat in sw_features:
	sw_feats[feat] = pd.DataFrame({'input':[], 'cont':[]})
for feat in combined_features:
	combined_feats[feat] = pd.DataFrame({'input':[], 'cont':[]})

for station in stations:
	print(station)

	# Loading the data

	with open(f'../data/prepared_data/SW_only_{station}_test_dict.pkl', 'rb') as d:
		sw_test_dict = pickle.load(d)

	with open(f'../data/prepared_data/combined_{station}_test_dict.pkl', 'rb') as b:
		combined_test_dict = pickle.load(b)

	for storm in storms:

		temp = sw_test_dict[f'storm_{storm}']['unscaled_array']

		combined_unscaled = np.concatenate(combined_test_dict[f'storm_{storm}']['unscaled_array'], axis=0)
		sw_unscaled = np.concatenate(sw_test_dict[f'storm_{storm}']['unscaled_array'], axis=0)

		# Loading in the pre-calculated SHAP values
		solar_shap, com_shap = [], []
		for split in tqdm(splits):
			with open(f'outputs/shap_values/sw_values_{station}_storm_{storm}_split_{split}.pkl', 'rb') as s:
				sw_shap_values = pickle.load(s)
			with open(f'outputs/shap_values/combined_values_{station}_storm_{storm}_split_{split}.pkl', 'rb') as c:
				combined_shap_values = pickle.load(c)

			sw_shap_values = sw_shap_values[1].reshape(sw_shap_values[1].shape[0], sw_shap_values[1].shape[1], sw_shap_values[1].shape[2])
			combined_shap_values = combined_shap_values[1].reshape(combined_shap_values[1].shape[0], combined_shap_values[1].shape[1], combined_shap_values[1].shape[2])

			sw_sum = np.sum(np.absolute(sw_shap_values), axis=(1,2))
			combined_sum = np.sum(np.absolute(combined_shap_values), axis=(1,2))
			# sw_sum = np.array([np.sum(sw_shap_values[i,:,:],axis=None) for i in range(sw_shap_values.shape[0])])
			# combined_sum = np.array([np.sum(combined_shap_values[i,:,:],axis=None) for i in range(combined_shap_values.shape[0])])
			sw_shap_values = np.array([((sw_shap_values[i,:,:]/sw_sum[i])*100) for i in range(sw_shap_values.shape[0])])
			# sw_shap_values = (sw_shap_values/np.sum(np.absolute(sw_shap_values), axis=0)[:,None,None])*100
			# print(np.max(sw_shap_values))
			combined_shap_values = np.array([((combined_shap_values[i,:,:]/combined_sum[i])*100) for i in range(combined_shap_values.shape[0])])
			# combined_shap_values = (combined_shap_values/np.sum(np.absolute(combined_shap_values), axis=0)[:,None,None])*100

			sw_shap_values = np.concatenate(sw_shap_values, axis=0)
			combined_shap_values = np.concatenate(combined_shap_values, axis=0)

			for i in range(sw_shap_values.shape[1]):
				temp = pd.DataFrame()
				temp['input'] = sw_unscaled[:,i]
				temp['cont'] = sw_shap_values[:,i]

				sw_feats[sw_features[i]] = pd.concat([sw_feats[sw_features[i]], temp], axis=0, ignore_index=True)

			for i in range(combined_shap_values.shape[1]):
				temp = pd.DataFrame()
				temp['input'] = combined_unscaled[:,i]
				temp['cont'] = combined_shap_values[:,i]

				combined_feats[combined_features[i]] = pd.concat([combined_feats[combined_features[i]], temp], axis=0, ignore_index=True)

combined_feats['dBHt'] = combined_feats['dBHt'][combined_feats['dBHt']['input']<300]
combined_feats['B'] = combined_feats['B'][combined_feats['B']['input']<1750]
combined_feats['proton_density'] = combined_feats['proton_density'][combined_feats['proton_density']['input']<45]
combined_feats['E'] = combined_feats['E'][combined_feats['E']['input']>-395]

plotting_feature = ["B_Total", "BY_GSM", "BZ_GSM", "Vx", "Vy", "proton_density", "AE_INDEX", "B", "dBHt"]
titles = ["$\mathregular{B_{total}^{GSM}}$", "$\mathregular{B_{y}^{GSM}}$", "$\mathregular{B_{z}^{GSM}}$",
			"$\mathregular{V_{x}}$", "$\mathregular{V_{y}}$", "$\mathregular{\u03C1_{SW}}$", "AE Index",
			"$\mathregular{B_{H}}$", "dB/dt"]

fig = plt.figure(figsize=(20,13))

for i, feat in enumerate(plotting_feature):

	combined_feats[feat].dropna(inplace=True)
	ax = plt.subplot(3,3,i+1)

	plt.title(titles[i], fontsize=17)
	plt.hist2d(x=combined_feats[feat]['input'], y=combined_feats[feat]['cont'], bins=100, norm=colors.LogNorm(), cmap='magma')
	plt.axhline(0, color='black', linestyle='--')
	# plt.scatter(x=combined_feats[feat]['input'], y=combined_feats[feat]['cont'], s=2, color='black')
	plt.colorbar()
	plt.ylim(-10,25)
	plt.ylabel('% Contribution')

plt.savefig('plots/shap/combined_feature_contributions.png', bbox_inches='tight')


sw_plotting_feature = ["B_Total", "BY_GSM", "BZ_GSM", "Vx", "Vy", "proton_density"]
titles = ["$\mathregular{B_{total}^{GSM}}$", "$\mathregular{B_{y}^{GSM}}$", "$\mathregular{B_{z}^{GSM}}$",
			"$\mathregular{V_{x}}$", "$\mathregular{V_{y}}$", "$\mathregular{\u03C1_{SW}}$"]
sw_feats['proton_density'] = sw_feats['proton_density'][sw_feats['proton_density']['input']<45]

fig = plt.figure(figsize=(20,13))

for i, feat in enumerate(sw_plotting_feature):

	sw_feats[feat].dropna(inplace=True)
	ax = plt.subplot(2,3,i+1)

	plt.title(titles[i], fontsize=17)
	plt.hist2d(x=sw_feats[feat]['input'], y=sw_feats[feat]['cont'], bins=100, norm=colors.LogNorm(), cmap='magma')
	plt.axhline(0, color='black', linestyle='--')
	# plt.scatter(x=sw_feats[feat]['input'], y=sw_feats[feat]['cont'], s=2, color='black')
	plt.colorbar()
	plt.ylim(-10,25)
	plt.ylabel('% Contribution')

plt.savefig('plots/shap/sw_feature_contributions.png', bbox_inches='tight')