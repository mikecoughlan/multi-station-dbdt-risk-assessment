##############################################################################################################
#
# project/risk_classification/k_fold_analysis_ver2.py
#
# Calculates metric scores from predicted data. Prepares them for plotting. Also calculates the percentiles
# that will be used for plotting the uncertainties.
#
##############################################################################################################


import pickle
import random
from pickle import load as pkload
from statistics import mean

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import auc, mean_squared_error, precision_recall_curve
from tensorflow.keras.models import load_model

# stops this program from hogging the GPU
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

# Data directories
projectDir = '~/projects/risk_classification/'


CONFIG = {'thresholds': [7.15], # list of thresholds to be examined.
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
      'forecast': 30,
      'window': 30,                                 # time window over which the metrics will be calculated
      'splits': 1,                         # amount of k fold splits to be performed. Program will create this many models
      'stations':['OTT', 'STJ', 'VIC', 'NEW', 'ESK', 'WNG', 'LER', 'BFE', 'NGK'],
	  'version':3}    # list of stations being examined




def load_feather(station, i):

  df = pd.read_feather('outputs/{0}/version_{1}_storm_{2}.feather'.format(station, CONFIG['version'], i))
  # making the Date_UTC the index
  pd.to_datetime(df['Date_UTC'], format='%Y-%m-%d %H:%M:%S')
  df.reset_index(drop=True, inplace=True)
  df.set_index('Date_UTC', inplace=True, drop=True)
  df.index = pd.to_datetime(df.index)

  try:
    df.drop('Unnamed: 0', inplace=True, axis=1)
  except KeyError:
    print('No Unnamed column.')

  return df


def calculating_scores(df, splits):
  '''calculating the hss and rmse scores and then putting them in a list. Then getting a 95 percent confidence level for each metric.
    Inputs:
    df: results df from the testing predictions
    threshold: threshold being examined
    splits: number of splits
	'''

  index_column = ['mean', 'max' ,'min']

  prec_recall = pd.DataFrame()
  metrics = pd.DataFrame({'ind':index_column})

  bias, std_pred, rmse, area_uc, precision, recall, hss = [], [], [], [], [], [], []      # initalizing the lists for storing the individual scores

  for split in range(splits):
    pred = df['predicted_split_{0}'.format(split)]        # Grabbing the specific predicted data
    re = df['crossing']                   # and the real data for comparison


    prec, rec, ____ = precision_recall_curve(re, pred)
    area = auc(rec, prec)

    # segmenting the pd.series into the confusion matrix indicies
    A = df[(df['predicted_split_{0}'.format(split)] >= 0.5) & (df['crossing'] == 1)]
    B = df[(df['predicted_split_{0}'.format(split)] >= 0.5) & (df['crossing'] == 0)]
    C = df[(df['predicted_split_{0}'.format(split)] < 0.5) & (df['crossing'] == 1)]
    D = df[(df['predicted_split_{0}'.format(split)] < 0.5) & (df['crossing'] == 0)]

    a, b, c, d = len(A), len(B), len(C), len(D)           # getting the values from the length of each df

    # doing the calculations. Nan segments are just to avoid 1/0 errors
    if (a+c) > 0:
      prob_det = a/(a+c)
      freq_bias = (a+b)/(a+c)
    else:
      prob_det = 'NaN'
      freq_bias = 'NaN'
    if (b+d) > 0:
      prob_false = b/(b+d)
    else:
      prob_false = 'NaN'
    if ((a+c)*(c+d)+(a+b)*(b+d)) > 0:
      hs_score = (2*((a*d)-(b*c)))/((a+c)*(c+d)+(a+b)*(b+d))
    else:
      hs_score = 'NaN'

    hss.append(hs_score)

    # adding the data to lists
    bias.append((pred.mean()-re.mean()))
    std_pred.append(pred.std())
    area_uc.append(area.round(decimals=3))
    rmse.append(np.sqrt(mean_squared_error(re,pred)))             # calculating the root mean square error
    precision.append(prec)
    recall.append(rec)


  try:
    medIdx = area_uc.index(np.percentile(area_uc,50,interpolation='nearest'))  # type: ignore
    prec_recall['prec'] = precision[medIdx]
    prec_recall['rec'] = recall[medIdx]

  except ValueError:
    print('No AUC scores for this station and storm')


  hss = np.array(hss)
  bias = np.array(bias)
  std_pred = np.array(std_pred)
  rmse = np.array(rmse)
  area_uc = np.array(area_uc)

  max_perc = 97.5
  min_perc = 2.5

  metrics['HSS'] = [np.mean(hss), np.percentile(hss, max_perc), np.percentile(hss, min_perc)]
  metrics['BIAS'] = [np.mean(bias), np.percentile(bias, max_perc), np.percentile(bias, min_perc)]
  metrics['STD_PRED'] = [np.mean(std_pred), np.percentile(std_pred, max_perc), np.percentile(std_pred, min_perc)]
  metrics['RMSE'] = [np.mean(rmse), np.percentile(rmse, max_perc), np.percentile(rmse, min_perc)]
  metrics['AUC'] = [np.mean(area_uc), np.percentile(area_uc, max_perc), np.percentile(area_uc, min_perc)]

  metrics.set_index('ind', drop=True, inplace=True)
  print(metrics['AUC'])

  return prec_recall, metrics



def aggregate_results(length, splits, station):

  results_dict = {}
  for i in range(length):
    results_dict['storm_{0}'.format(i)] = {}
    results_dict['storm_{0}'.format(i)]['raw_results'] = load_feather(station, i)
    prec_recall, metrics = calculating_scores(results_dict['storm_{0}'.format(i)]['raw_results'], splits)
    results_dict['storm_{0}'.format(i)]['precision_recall'] = prec_recall
    results_dict['storm_{0}'.format(i)]['metrics'] = metrics
    results_dict['storm_{0}'.format(i)]['STD_real'] = results_dict['storm_{0}'.format(i)]['raw_results']['crossing'].std()

  return results_dict


def main():

  stations_dict = {}
  for station in CONFIG['stations']:
    results_dict = aggregate_results(len(CONFIG['test_storm_stime']), CONFIG['splits'], station)
    stations_dict[station] = results_dict


  with open('outputs/stations_results_dict.pkl', 'wb') as f:
    pickle.dump(stations_dict, f)




if __name__ == '__main__':

  main()    # calling the main function.

  print('It ran. Good job!')                    # if we get here we're doing alright.


