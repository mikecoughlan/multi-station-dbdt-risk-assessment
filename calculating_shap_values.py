import gc
import json
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from matplotlib import colors
from numba import cuda
from tensorflow.keras.models import Sequential, load_model
from tqdm import tqdm

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


def loading_data(station, split):

	with open(f'../data/prepared_data/SW_only_{station}_train_dict.pkl', 'rb') as f:
		sw_train_dict = pickle.load(f)

	with open(f'../data/prepared_data/SW_only_{station}_test_dict.pkl', 'rb') as d:
		sw_test_dict = pickle.load(d)

	with open(f'../data/prepared_data/combined_{station}_train_dict.pkl', 'rb') as c:
		combined_train_dict = pickle.load(c)

	with open(f'../data/prepared_data/combined_{station}_test_dict.pkl', 'rb') as b:
		combined_test_dict = pickle.load(b)

	sw_model = load_model(f'models/{station}/CNN_SW_only_split_{split}.h5')
	combined_model = load_model(f'models/{station}/CNN_version_5_split_{split}.h5')

	return sw_train_dict, sw_test_dict, combined_train_dict, combined_test_dict, sw_model, combined_model


def calculating_shap_values(train_dict, xtest, model):

	xtest = xtest.reshape((xtest.shape[0], xtest.shape[1], xtest.shape[2], 1))

	# reducing the amount of the training dataset used to find the shap values
	xtrain = train_dict['X']
	xtrain = xtrain.reshape((xtrain.shape[0], xtrain.shape[1], xtrain.shape[2], 1))
	background = xtrain[np.random.choice(xtrain.shape[0], 1000, replace=False)]

	# attempting to use shap
	explainer = shap.DeepExplainer(model, background)

	shap_values = explainer.shap_values(xtest, check_additivity=True)


	return shap_values


def combining_arrays(combined_test_dict, sw_test_dict):

	combined, solar = [], []

	for comb, sw in zip(combined_test_dict.keys(), sw_test_dict.keys()):
		combined.append(combined_test_dict[comb]['Y'])
		solar.append(sw_test_dict[sw]['Y'])

	combined_xtest = np.concatenate(combined, axis=0)
	sw_xtest = np.concatenate(solar, axis=0)

	return combined_xtest, sw_xtest


def main(station):

	random.seed(42)
	integers = []

	for i in range(10):
		random_int = random.randint(0,100)
		integers.append(random_int)

	for split in tqdm(integers):

		shap_dict = {}

		sw_train_dict, sw_test_dict, combined_train_dict, combined_test_dict, sw_model, combined_model = loading_data(station, split)

		combined_xtest, sw_xtest = combining_arrays(combined_test_dict, sw_test_dict)

		combined_shap_values = calculating_shap_values(combined_train_dict, combined_xtest, combined_model)
		sw_shap_values = calculating_shap_values(sw_train_dict, sw_xtest, sw_model)

		shap_dict[f'combined_split_{split}'] = combined_shap_values
		shap_dict[f'sw_split_{split}'] = sw_shap_values

		cuda.select_device(0)
		cuda.close()

		del combined_shap_values, sw_shap_values
		gc.collect()

		with open(f'outputs/shap_values/{station}_split_{split}.pkl', 'wb') as f:
			pickle.dump(shap_dict, f)



if __name__ == '__main__':

	for station in CONFIG['stations']:
		main(station)
		print('Finished {0}'.format(station))

	print('It ran. Good job!')









