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

# loading config and specific model config files. Using them as dictonaries
with open('config.json', 'r') as con:
	CONFIG = json.load(con)


def main(station):

	splits = [2, 13, 19, 20, 39, 43, 54, 65, 72, 97]

	storms = [7,4]

	with open(f'../data/prepared_data/SW_only_{station}_train_dict.pkl', 'rb') as f:
		sw_train_dict = pickle.load(f)

	with open(f'../data/prepared_data/SW_only_{station}_test_dict.pkl', 'rb') as d:
		sw_test_dict = pickle.load(d)

	with open(f'../data/prepared_data/combined_{station}_train_dict.pkl', 'rb') as c:
		combined_train_dict = pickle.load(c)

	with open(f'../data/prepared_data/combined_{station}_test_dict.pkl', 'rb') as b:
		combined_test_dict = pickle.load(b)

	for storm in storms:

		sw_storm = sw_test_dict[f'storm_{storm}']['Y']
		combined_storm = combined_test_dict[f'storm_{storm}']['Y']

		sw_storm = sw_storm.reshape(sw_storm.shape[0], sw_storm.shape[1], sw_storm.shape[2], 1)
		combined_storm = combined_storm.reshape(combined_storm.shape[0], combined_storm.shape[1], combined_storm.shape[2], 1)


		for model_split in splits:

			if os.path.exists(f'outputs/shap_values/sw_values_{station}_storm_{storm}_split_{model_split}.pkl') \
				and os.path.exists(f'outputs/shap_values/combined_values_{station}_storm_{storm}_split_{model_split}.pkl'):
				continue

			sw_model = load_model(f'models/{station}/CNN_SW_only_split_{model_split}.h5')
			combined_model = load_model(f'models/{station}/CNN_version_5_split_{model_split}.h5')


			# reducing the amount of the training dataset used to find the shap values
			combined_xtrain = combined_train_dict['X']
			combined_xtrain = combined_xtrain.reshape((combined_xtrain.shape[0], combined_xtrain.shape[1], combined_xtrain.shape[2], 1))
			combined_background = combined_xtrain[np.random.choice(combined_xtrain.shape[0], 1000, replace=False)]

			# attempting to use shap
			combined_explainer = shap.DeepExplainer(combined_model, combined_background)

			sw_xtrain = sw_train_dict['X']
			sw_xtrain = sw_xtrain.reshape((sw_xtrain.shape[0], sw_xtrain.shape[1], sw_xtrain.shape[2], 1))
			sw_background = sw_xtrain[np.random.choice(sw_xtrain.shape[0], 1000, replace=False)]

			# attempting to use shap
			sw_explainer = shap.DeepExplainer(sw_model, sw_background)


			combined_shap_values_check = combined_explainer.shap_values(combined_storm, check_additivity=False)
			print('Finished Combined Shap values. Onto solar wind ones....')
			sw_shap_values_check = sw_explainer.shap_values(sw_storm, check_additivity=False)


			with open(f'outputs/shap_values/sw_values_{station}_storm_{storm}_split_{model_split}.pkl', 'wb') as s:
				pickle.dump(sw_shap_values_check, s)
			with open(f'outputs/shap_values/combined_values_{station}_storm_{storm}_split_{model_split}.pkl', 'wb') as c:
				pickle.dump(combined_shap_values_check, c)


			gc.collect()


		solar, com = [], []
		for split in splits:
			with open(f'outputs/shap_values/sw_values_{station}_storm_{storm}_split_{split}.pkl', 'rb') as s:
				sw_shap_values_check = pickle.load(s)
			with open(f'outputs/shap_values/combined_values_{station}_storm_{storm}_split_{split}.pkl', 'rb') as c:
				combined_shap_values_check = pickle.load(c)

			solar.append(sw_shap_values_check[1])
			com.append(combined_shap_values_check[1])


		sw_shap_values = np.stack(solar, axis=3)
		combined_shap_values = np.stack(com, axis=3)

		sw_shap_values_check = np.mean(sw_shap_values, axis=3)
		combined_shap_values_check = np.mean(combined_shap_values, axis=3)

		sw_storm_condensed = np.sum(sw_shap_values_check, axis=1)
		combined_storm_condensed = np.sum(combined_shap_values_check, axis=1)

		sw_storm_condensed = sw_storm_condensed.reshape(sw_storm_condensed.shape[0], sw_storm_condensed.shape[1])
		combined_storm_condensed = combined_storm_condensed.reshape(combined_storm_condensed.shape[0], combined_storm_condensed.shape[1])


		sw_features = ["sinMLT", "cosMLT", "B_Total", "BY_GSM",
					"BZ_GSM", "Vx", "Vy", "Vz", "proton_density", "T"]
		combined_features = ["N", "E", "sinMLT", "cosMLT", "B_Total", "BY_GSM",
								"BZ_GSM", "Vx", "Vy", "Vz", "proton_density", "T",
								"AE_INDEX", "SZA", "dBHt", "B"]


		sw_df = pd.DataFrame(sw_storm_condensed, columns=sw_features)
		combined_df = pd.DataFrame(combined_storm_condensed, columns=combined_features)

		sw_df['Date_UTC'] = sw_test_dict[f'storm_{storm}']['real_df']['Date_UTC']
		combined_df['Date_UTC'] = combined_test_dict[f'storm_{storm}']['real_df']['Date_UTC']

		sw_df.set_index('Date_UTC', inplace=True)
		combined_df.set_index('Date_UTC', inplace=True)


		perc_sw_df = (sw_df.div(sw_df.abs().sum(axis=1), axis=0))*100
		perc_combined_df = (combined_df.div(combined_df.abs().sum(axis=1), axis=0))*100

		prec_sw_x = perc_sw_df.index
		prec_combined_x = perc_combined_df.index

		perc_sw_df.reset_index(drop=True, inplace=True)
		perc_combined_df.reset_index(drop=True, inplace=True)

		print(perc_sw_df.head())

		perc_sw_rolling = perc_sw_df.rolling(10).mean()
		perc_combined_rolling = perc_combined_df.rolling(10).mean()

		perc_sw_pos_df = perc_sw_df.mask(perc_sw_df < 0, other=0)
		print(perc_sw_pos_df.head())
		perc_sw_neg_df = perc_sw_df.mask(perc_sw_df > 0, other=0)
		perc_combined_pos_df = perc_combined_df.mask(perc_combined_df < 0, other=0)
		perc_combined_neg_df = perc_combined_df.mask(perc_combined_df > 0, other=0)

		perc_sw_pos_dict, perc_sw_neg_dict, perc_combined_pos_dict, perc_combined_neg_dict = {}, {}, {}, {}

		for pos, neg in zip(perc_sw_pos_df, perc_sw_neg_df):
			perc_sw_pos_dict[pos] = perc_sw_pos_df[pos].to_numpy()
			perc_sw_neg_dict[neg] = perc_sw_neg_df[neg].to_numpy()

		for pos, neg in zip(perc_combined_pos_df, perc_combined_neg_df):
			perc_combined_pos_dict[pos] = perc_combined_pos_df[pos].to_numpy()
			perc_combined_neg_dict[neg] = perc_combined_neg_df[neg].to_numpy()

		reordered_combined_model_parameters = ["sinMLT", "cosMLT", "B_Total", "BY_GSM",
												"BZ_GSM", "Vx", "Vy", "Vz", "proton_density", "T",
												"AE_INDEX", "SZA", "N", "E", "B", "dBHt"]

		perc_combined_pos_dict = {k : perc_combined_pos_dict[k] for k in reordered_combined_model_parameters}

		perc_combined_neg_dict = {k : perc_combined_neg_dict[k] for k in reordered_combined_model_parameters}

		sw_colors = sns.color_palette('tab20', len(perc_sw_pos_dict.keys()))
		combined_colors = sns.color_palette('tab20', len(perc_combined_pos_dict.keys())+4)

		for i in range(4):
			combined_colors.pop(-3)

		sw_greys = sns.color_palette('light:#C0C0C0', len(perc_sw_pos_dict.keys()))
		combined_greys = sns.color_palette('light:#C0C0C0', len(perc_combined_pos_dict.keys()))

		params = ['dBHt', 'Vx', 'B']

		for param in params:
			if param in sw_features:

				sw_param_location = sw_features.index(param)
				sw_greys[sw_param_location] = sw_colors[sw_param_location]

			combined_param_location = reordered_combined_model_parameters.index(param)
			combined_greys[combined_param_location] = combined_colors[combined_param_location]


		fig = plt.figure(figsize=(20,17))

		ax1 = plt.subplot(111)
		ax1.set_title('Solar Wind Model')
		plt.stackplot(prec_sw_x, perc_sw_pos_dict.values(), labels=perc_sw_pos_dict.keys(), colors=sw_greys)
		plt.stackplot(prec_sw_x, perc_sw_neg_dict.values(), colors=sw_greys)
		plt.ylabel('Percent Contribution')
		plt.legend(bbox_to_anchor=(1,1), loc='upper left')
		plt.axhline(0, color='black')

		plt.savefig(f'plots/shap/rolling_sw_percent_contribution_{station}_storm_{storm}_highlight_{params}.png')


		fig = plt.figure(figsize=(20,17))

		ax2 = plt.subplot(111)
		ax2.set_title('Combined Model')
		plt.stackplot(prec_combined_x, perc_combined_pos_dict.values(), labels=perc_combined_pos_dict.keys(), colors=combined_greys)
		plt.stackplot(prec_combined_x, perc_combined_neg_dict.values(), colors=combined_greys)
		plt.ylabel('Percent Contribution')
		plt.axhline(0, color='black')
		plt.legend(bbox_to_anchor=(1,1), loc='upper left')

		plt.savefig(f'plots/shap/rolling_combined_percent_contribution_{station}_storm_{storm}_highlight_{params}.png')



		fig = plt.figure(figsize=(20,17))

		ax1 = plt.subplot(111)
		ax1.set_title('Solar Wind Model')
		for param in params:
			if param in perc_sw_rolling.columns:
				plt.plot(perc_sw_rolling[param], label=param)
		plt.ylabel('Percent Contribution')
		plt.legend(bbox_to_anchor=(1,1), loc='upper left')
		plt.axhline(0, color='black')

		plt.savefig(f'plots/shap/rolling_line_sw_percent_contribution_{station}_storm_{storm}.png')


		fig = plt.figure(figsize=(20,17))

		ax2 = plt.subplot(111)
		ax2.set_title('Combined Model')
		for param in params:
			plt.plot(perc_combined_rolling[param], label=param)
		plt.ylabel('Percent Contribution')
		plt.axhline(0, color='black')
		plt.legend(bbox_to_anchor=(1,1), loc='upper left')

		plt.savefig(f'plots/shap/rolling_line_combined_percent_contribution_{station}_storm_{storm}.png')


if __name__ == '__main__':

	stations = ["OTT", "BFE", "WNG", "LER", "ESK", "STJ", "NEW", "VIC"]
	for station in stations:
		main(station)
		print(f'Finished {station}')

	print('It ran. Good job!')









