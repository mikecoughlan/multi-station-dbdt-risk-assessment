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

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import auc, mean_squared_error, precision_recall_curve

# stops this program from hogging the GPU
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

pd.options.mode.chained_assignment = None  # default='warn'

# Data directories
projectDir = '~/projects/risk_classification/'


CONFIG = {'thresholds': [7.15], # list of thresholds to be examined.
      'params': ['N', 'E', 'sinMLT', 'cosMLT', 'B_Total', 'BY_GSM',
	   					'BZ_GSM', 'Vx', 'Vy', 'Vz', 'proton_density', 'T',
	   					 'AE_INDEX', 'SZA', 'dBHt', 'B'],                  # List of parameters that will be used for training.
                                                  # Date_UTC will be removed, kept here for resons that will be evident below
      'test_storm_stime': ['2001-03-29 09:59:00', '2001-08-29 21:59:00', '2005-05-13 21:59:00',
                 '2005-08-30 07:59:00', '2006-12-13 09:59:00', '2010-04-03 21:59:00',
                 '2011-08-04 06:59:00', '2015-03-15 23:59:00'],           # These are the start times for testing storms
      'test_storm_etime': ['2001-04-02 12:00:00', '2001-09-02 00:00:00', '2005-05-17 00:00:00',
                  '2005-09-02 12:00:00', '2006-12-17 00:00:00', '2010-04-07 00:00:00',
                  '2011-08-07 09:00:00', '2015-03-19 14:00:00'],  # end times for testing storms. This will remove them from training
      'forecast': 30,
      'window': 30,                                 # time window over which the metrics will be calculated
      'splits': 100,                         # amount of k fold splits to be performed. Program will create this many models
      # 'stations':['OTT', 'STJ', 'VIC', 'NEW', 'ESK', 'WNG', 'LER', 'BFE', 'NGK'],
      'stations': ['OTT', 'STJ', 'WNG', 'BFE', 'NEW', 'VIC', 'ESK', 'LER'],
	    'version':5}    # list of stations being examined




def load_feather(station, i):

  df = pd.read_feather('outputs/{0}/version_{1}_storm_{2}.feather'.format(station, CONFIG['version'], i))
  # making the Date_UTC the index
  pd.to_datetime(df['Date_UTC'], format='%Y-%m-%d %H:%M:%S')
  df.reset_index(drop=True, inplace=True)
  df.set_index('Date_UTC', inplace=True, drop=True)
  df.index = pd.to_datetime(df.index)

  # try:
  #   df.drop('Unnamed: 0', inplace=True, axis=1)
  # except KeyError:
  #   print('No Unnamed column.')

  return df


def getting_model_input_data():

  mean_df, std_df, max_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
  for station in CONFIG['stations']:

    with open('../data/prepared_data/{0}_test_dict.pkl'.format(station), 'rb') as f:
      test_dict = pickle.load(f)

    X = [test_dict['storm_{0}'.format(i)]['Y'] for i in range(len(test_dict))] 		# loading the omni data
    X = np.concatenate(X, axis=0)
    X_mean = np.mean(X, axis=1)
    X_std = np.std(X, axis=1)
    X_max = np.amax(np.absolute(X), axis=1)

    sw_feats = ['B_Total', 'BY_GSM', 'BZ_GSM', 'Vx', 'Vy', 'Vz', 'proton_density', 'T','AE_INDEX', 'SZA']

    for col, feat in zip(range(X.shape[2]), CONFIG['params']):
      if (station != CONFIG['stations'][0]) and (feat in sw_feats):
        continue
      if feat in sw_feats:
        mean_df['{0}'.format(feat)] = X_mean[:,col]
        std_df['{0}'.format(feat)] = X_std[:,col]
        max_df['{0}'.format(feat)] = X_max[:,col]
      else:
        mean_df['{0}_{1}'.format(station, feat)] = X_mean[:,col]
        std_df['{0}_{1}'.format(station, feat)] = X_std[:,col]
        max_df['{0}_{1}'.format(station, feat)] = X_max[:,col]

  return mean_df, std_df, max_df


def calculating_scores(df, splits, station):
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
  diff_df, spread_df = pd.DataFrame(), pd.DataFrame()
  diff_df['Date_UTC'] = df.index
  spread_df['Date_UTC'] = df.index
  diff_df.set_index('Date_UTC', inplace=True)
  spread_df.set_index('Date_UTC', inplace=True)

  newdf = df[['predicted_split_{0}'.format(split) for split in range(splits)]]
  top_perc = newdf.quantile(0.975, axis=1)
  bottom_perc = newdf.quantile(0.025, axis=1)
  spread_df[station] = top_perc - bottom_perc

  for split in range(splits):
    temp_df = df[['crossing', 'predicted_split_{0}'.format(split)]]
    temp_df.dropna(inplace=True)

    pred = temp_df['predicted_split_{0}'.format(split)]        # Grabbing the specific predicted data
    re = temp_df['crossing']                   # and the real data for comparison

    diff_df['{0}_split_{1}'.format(station, split)] = abs(re-pred)

    prec, rec, ____ = precision_recall_curve(re, pred)
    area = auc(rec, prec)

    # segmenting the pd.series into the confusion matrix indicies
    A = temp_df[(temp_df['predicted_split_{0}'.format(split)] >= 0.5) & (temp_df['crossing'] == 1)]
    B = temp_df[(temp_df['predicted_split_{0}'.format(split)] >= 0.5) & (temp_df['crossing'] == 0)]
    C = temp_df[(temp_df['predicted_split_{0}'.format(split)] < 0.5) & (temp_df['crossing'] == 1)]
    D = temp_df[(temp_df['predicted_split_{0}'.format(split)] < 0.5) & (temp_df['crossing'] == 0)]

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

    if ((a>0)or(d>0)) and ((b==0)and(c==0)):
      hs_score = 1

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

  return prec_recall, metrics, diff_df, spread_df



def aggregate_results(length, splits, station):

  results_dict = {}
  for i in range(length):
    results_dict['storm_{0}'.format(i)] = {}
    results_dict['storm_{0}'.format(i)]['raw_results'] = load_feather(station, i)
    prec_recall, metrics, diff_df, spread_df = calculating_scores(results_dict['storm_{0}'.format(i)]['raw_results'], splits, station)
    results_dict['storm_{0}'.format(i)]['diff_df'] = diff_df
    results_dict['storm_{0}'.format(i)]['spread_df'] = spread_df
    results_dict['storm_{0}'.format(i)]['precision_recall'] = prec_recall
    results_dict['storm_{0}'.format(i)]['metrics'] = metrics
    results_dict['storm_{0}'.format(i)]['STD_real'] = results_dict['storm_{0}'.format(i)]['raw_results']['crossing'].std()

  return results_dict


def plotting_corrs(diff_df, spread_df, params, name):

  sw_feats = ['B_Total', 'BY_GSM', 'BZ_GSM', 'Vx', 'Vy', 'Vz', 'proton_density', 'T','AE_INDEX', 'SZA']
  for param in params:
    x, y = pd.Series(), pd.Series()
    fig = plt.figure()
    plt.title('{0} {1} vs. Difference'.format(param, name))
    for station in CONFIG['stations']:
      for i in range(CONFIG['splits']):
        x = pd.concat([x,diff_df['{0}_split_{1}'.format(station, i)]])
        if param in sw_feats:
          y = pd.concat([y,diff_df[param]])
        else:
          y = pd.concat([y,diff_df['{0}_{1}'.format(station, param)]])
    temp_df = pd.DataFrame({'x':x,
                            'y':y})
    temp_df.dropna(inplace=True)
    x = np.array(temp_df['x'])
    y = np.array(temp_df['y'])
    plt.hist2d(x,y, bins=(100,100), norm=mpl.colors.LogNorm(), range=((0.01,1), (y.min(),y.max())))
    plt.savefig('plots/{0}_{1}_vs_difference.png'.format(param, name))

    fig = plt.figure()
    x, y = pd.Series(), pd.Series()
    plt.title('{0} {1} vs. Spread'.format(param, name))
    for station in CONFIG['stations']:
      x = pd.concat([x,spread_df['{0}'.format(station)]])
      if param in sw_feats:
        y = pd.concat([y,spread_df[param]])
      else:
        y = pd.concat([y,spread_df['{0}_{1}'.format(station, param)]])

    temp_df = pd.DataFrame({'x':x,
                            'y':y})
    temp_df.dropna(inplace=True)
    x = np.array(temp_df['x'])
    y = np.array(temp_df['y'])
    plt.hist2d(x,y, bins=(100,100), norm=mpl.colors.LogNorm(), range=((0.01,1), (y.min(),y.max())))
    plt.savefig('plots/{0}_{1}_vs_spread.png'.format(param, name))


def main():

  stations_dict = {}
  for station in CONFIG['stations']:
    results_dict = aggregate_results(len(CONFIG['test_storm_stime']), CONFIG['splits'], station)
    stations_dict[station] = results_dict
  diff_df, spread_df = pd.DataFrame(), pd.DataFrame()
  for i in range(len(CONFIG['test_storm_stime'])):
    temp_diff, temp_spread = pd.DataFrame(), pd.DataFrame()
    for station in CONFIG['stations']:
      temp_diff = pd.concat([temp_diff, stations_dict[station]['storm_{0}'.format(i)]['diff_df']], axis=1, ignore_index=False)
      temp_spread = pd.concat([temp_spread, stations_dict[station]['storm_{0}'.format(i)]['spread_df']], axis=1, ignore_index=False)

    diff_df = pd.concat([diff_df, temp_diff], axis=0)
    spread_df = pd.concat([spread_df, temp_spread], axis=0)

  mean_df, std_df, max_df = getting_model_input_data()
  diff_df.reset_index(inplace=True, drop=True)
  spread_df.reset_index(inplace=True, drop=True)

  mean_diff_df = pd.concat([diff_df, mean_df], axis=1, ignore_index=False)
  std_diff_df = pd.concat([diff_df, std_df], axis=1, ignore_index=False)
  max_diff_df = pd.concat([diff_df, max_df], axis=1, ignore_index=False)

  mean_spread_df = pd.concat([spread_df, mean_df], axis=1, ignore_index=False)
  std_spread_df = pd.concat([spread_df, std_df], axis=1, ignore_index=False)
  max_spread_df = pd.concat([spread_df, max_df], axis=1, ignore_index=False)

  plotting_corrs(mean_diff_df, mean_spread_df, CONFIG['params'], 'mean')
  plotting_corrs(std_diff_df, std_spread_df, CONFIG['params'], 'std')
  plotting_corrs(max_diff_df, max_spread_df, CONFIG['params'], 'max')

  with open('outputs/stations_results_dict.pkl', 'wb') as f:
    pickle.dump(stations_dict, f)




if __name__ == '__main__':

  main()    # calling the main function.

  print('It ran. Good job!')                    # if we get here we're doing alright.


