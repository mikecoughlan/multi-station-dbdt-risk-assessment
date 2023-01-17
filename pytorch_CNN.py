import gc
import json
import os
import pickle
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# loading config and specific model config files. Using them as dictonaries
with open('config.json', 'r') as con:
	CONFIG = json.load(con)

with open('model_config.json', 'r') as mcon:
	MODEL_CONFIG = json.load(mcon)

# setting the random seeds for reproducibility
random.seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])

# Hyper-params
num_epochs = 100
batch_size = 32
learning_rate = 1e-6

def loading_data_and_indicies(station):
	'''
	Loading the preprepared traing and testing dictonaries for the station being examined.

	Args:
		station (str): 3 diget code indicating the data from which supermag station is being loaded

	Returns:
		train_dict, test_dict: dictonaries containing the training and target arrays (X, y)
		and testing data (X, y) for each of the testing storms

		train_indicies, val_indicies: dataframes containing the indicies for doing the boothstrapping
	'''

	with open('../data/prepared_data/SW_only_{0}_train_dict.pkl'.format(station), 'rb') as train:
		train_dict = pickle.load(train)
	with open('../data/prepared_data/SW_only_{0}_test_dict.pkl'.format(station), 'rb') as test:
		test_dict = pickle.load(test)
	train_indicies = pd.read_feather('../data/prepared_data/SW_only_{0}_train_indicies.feather'.format(station))
	val_indicies = pd.read_feather('../data/prepared_data/SW_only_{0}_val_indicies.feather'.format(station))

	return train_dict, test_dict, train_indicies, val_indicies


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def transform_data_for_modeling(X, y, batch_size=32, shuffle=True):

	# transofrming the numpy arrays into tensors
	X = torch.Tensor(X)
	y = torch.Tensor(y)

	# creating the tensor datasets from teh tensor objects
	dataset = torch.utils.data.TensorDataset(X, y)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

	return dataloader


class CNN(nn.Module):

	def __init__(self):
		super(CNN, self).__init__()
		self.conv1 = nn.Conv2d(out_channels=128, in_channels=1, kernel_size=(2,2), padding=1)
		self.relu = nn.ReLU()
		self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2,2))
		self.flatten = nn.Flatten()
		self.drop = nn.Dropout(p=0.2)
		self.dense1 = nn.Linear(128*15*5, 128)
		self.dense2 = nn.Linear(128, 64)
		self.dense3 = nn.Linear(64, 2)

	def forward(self, x):
		x1 = self.conv1(x)
		x2 = self.relu(x1)
		x3 = self.pool(x2)
		x4 = self.flatten(x3)
		x5 = self.drop(x4)
		x6 = self.dense1(x5)
		x7 = self.relu(x6)
		x8 = self.drop(x7)
		x9 = self.dense2(x8)
		x10 = self.relu(x9)
		x11 = self.dense3(x10)
		return x11


def model_training(model, train_data, val_data, early_stopping_patience=3):

	# loss, optimizer, and early stopping condition
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	early_stopper = EarlyStopper(patience=early_stopping_patience, min_delta=0)

	H = {
	"train_loss": [],
	"train_acc": [],
	"val_loss": [],
	"val_acc": []
	}

	# training loop
	for epoch in range(num_epochs):

		# start the clock
		start_time = time.time()

		# setting the model into training mode
		model.train()

		# initialize the total training and validation loss
		trainLoss = 0
		valLoss = 0
		train_samples = 0
		val_samples = 0

		for i, (X, y) in enumerate(train_data):

			# loading features and targets to device
			(X, y) = (X.to(device), y.to(device))
			X = X.reshape([X.size(0), 1, X.size(1), X.size(2)])

			# forward pass
			outputs = model(X)
			loss = criterion(outputs, y)
			train_samples += y.size(0)

			# backward pass
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# adding the losses
			trainLoss += loss.item() * y.size(0)

			if (i+1) % 100 == 0:
				print(f'epoch: {epoch} / {num_epochs}, step {i+1}/{len(train_data)}, loss = {trainLoss/train_samples}')


		# validation stage
		with torch.no_grad():
			model.eval()
			for (X_val, y_val) in val_data:

				X_val = X_val.reshape([X_val.size(0), 1, X_val.size(1), X_val.size(2)])

				(X_val, y_val) = (X_val.to(device), y_val.to(device))
				val_outputs = model(X_val)
				valLoss += criterion(val_outputs, y_val)
				val_samples += y_val.size(0)

			end_time = time.time()
			epoch_training_time = end_time - start_time

			print(f'epoch: {epoch} / {num_epochs}, time: {epoch_training_time:.4f}s, train loss: {trainLoss/train_samples: .4f}, val loss: {valLoss/val_samples: .4f}')


		if early_stopper.early_stop(valLoss/val_samples):
			print('Early Stopping conditions met. Exiting training loop before epoch limit....')
			break

	return model





def main(station):
	'''
	Pulls all the above functions together. Loops through the number of splits to create, fit ,
	and predict with a unique model for each train-val split. Outputs a saved file with the results.

	Args:
		station (str): 3 digit code for the station being examined. Passed to the script via arg parsing.
	'''

	# loading all data and indicies
	train_dict, test_dict, train_indicies, val_indicies = loading_data_and_indicies(station)

	# this is the bulk of the shuffeled k-fold splitting. We loop through the
	# list of indexes and train on the different train-val indices
	for split in range(MODEL_CONFIG['splits']):

		# the model path for this split and station
		PATH = f'models/{station}/CNN_pytorch_testing_split_{split}.pth'

		# getting the indicies for this random shuffeled split
		train_index = train_indicies['split_{0}'.format(split)].to_numpy()
		val_index = val_indicies['split_{0}'.format(split)].to_numpy()

		print('Split: '+ str(split))

		# pulling the data and catagorizing it into the train-val pairs
		xtrain = train_dict['X'][train_index]
		xval =  train_dict['X'][val_index]
		ytrain = train_dict['crossing'][train_index]
		yval = train_dict['crossing'][val_index]

		# transforming the data into tensors and dataloaders
		train_data = transform_data_for_modeling(xtrain, ytrain, batch_size=batch_size, shuffle=True)
		val_data = transform_data_for_modeling(xval, yval, batch_size=batch_size, shuffle=True)

		# if the saved model already exists, loads the pre-fit model
		if os.path.exists(PATH):
			model = CNN(*args, **kwargs)
			model.load_state_dict(torch.load(PATH))

		# if model has not been fit, fits the model
		else:
			model = CNN().to(device)
			model = model_training(model, train_data, val_data, early_stopping_patience=3)
			torch.save(model.state_dict(), PATH)

		# defines the test dictonary for storing results
		test_dict = making_predictions(model, test_dict, split)


	# for each storm we set the datetime index and saves the data in a CSV/feather file
	for i in range(len(test_dict)):
		real_df = test_dict['storm_{0}'.format(i)]['real_df']
		pd.to_datetime(real_df['Date_UTC'], format='%Y-%m-%d %H:%M:%S')
		real_df.reset_index(drop=True, inplace=True)

		if not os.path.exists('outputs/{0}'.format(station)):
			os.makedirs('outputs/{0}'.format(station))

		real_df.to_feather('outputs/{0}/SW_only_storm_{1}.feather'.format(station, i))



if __name__ == '__main__':
	# parser = argparse.ArgumentParser()
	# parser.add_argument('station',
	# 					action='store',
	# 					choices=['OTT', 'STJ', 'VIC', 'NEW', 'ESK', 'WNG', 'LER', 'BFE', 'NGK'],
	# 					type=str,
	# 					help='input station code for the SuperMAG station to be examined.')

	# args=parser.parse_args()

	main('OTT')

	print('It ran. God job!')