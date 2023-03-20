##############################################################################################################
#
#	  multi-station-dbdt-risk-assessment/analyzing_results.py
#
#   Calculates metric scores from predicted data and defines the median, mean and 95th percentiles for plotting.
#   Saves results to a dictonary for plotting in the plotting.py script. Plots some supplimental information
#   such as correlations betwwen model input values and model outputs.
#
##############################################################################################################


import json
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import auc, mean_squared_error, precision_recall_curve

# # stops this program from hogging the GPU
# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#   tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#   # Invalid device or cannot modify virtual devices once initialized.
#   pass

pd.options.mode.chained_assignment = None  # default='warn'


# loading config and specific model config files. Using them as dictonaries
with open('config.json', 'r') as con:
	CONFIG = json.load(con)

with open('model_config.json', 'r') as mcon:
	MODEL_CONFIG = json.load(mcon)


def load_feather(station, i):

  '''
  Function for loading the feather files saved in the modeling.py script

  Args:
    station (str): three diget code indicating the station for which results are being loaded
    i (int): testing storm code from 0-7

  Returns:
    pd.dataframe: dataframe contining the model predictions for the station and storm
  '''

  df = pd.read_feather(f'outputs/{station}/version_5_storm_{i}.feather')

  # making the Date_UTC the index
  pd.to_datetime(df['Date_UTC'], format='%Y-%m-%d %H:%M:%S')
  df.reset_index(drop=True, inplace=True)
  df.set_index('Date_UTC', inplace=True, drop=True)
  if i == 4:
    df = df[CONFIG['test_storm_stime'][i]:CONFIG['test_storm_etime'][i]]
  df.index = pd.to_datetime(df.index)

  sw_df = pd.read_feather(f'outputs/{station}/SW_only_storm_{i}.feather')


  # making the Date_UTC the index
  pd.to_datetime(sw_df['Date_UTC'], format='%Y-%m-%d %H:%M:%S')
  sw_df.reset_index(drop=True, inplace=True)
  sw_df.set_index('Date_UTC', inplace=True, drop=True)
  if i == 4:
    sw_df = sw_df[CONFIG['test_storm_stime'][i]:CONFIG['test_storm_etime'][i]]
  sw_df.index = pd.to_datetime(sw_df.index)

  return df, sw_df


def getting_model_input_data():

  '''
  Combines the data input to the models for testing, calculates the mean, max,
  and standard deviation of the 100 models for each time step. These are calculated
  for each station data individually, and the results are paried with the input data
  at that time step. These are saved in unique dataframes for each statistic.

  Returns:
    pd.dataframes (3): dataframes for each statistical metric.
  '''

  mean_df, std_df, max_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
  # loops through all the stations
  for station in CONFIG['stations']:

    # loads the testing data for this station
    with open('../data/prepared_data/quiet_time_{0}_test_dict.pkl'.format(station), 'rb') as f:
      test_dict = pickle.load(f)

    # creating a lsit of all the storm dataframes
    X = [test_dict['storm_{0}'.format(i)]['Y'] for i in range(len(test_dict))]

    # combining them into one dataframe
    X = np.concatenate(X, axis=0)

    # calculating the mean, max, and std for each time step
    X_mean = np.mean(X, axis=1)
    X_std = np.std(X, axis=1)
    X_max = np.amax(np.absolute(X), axis=1)

    # defining the solar wind features so they are only added to the resulting dfs once instead of for each station
    sw_feats = ['B_Total', 'BY_GSM', 'BZ_GSM', 'Vx', 'Vy', 'Vz', 'proton_density', 'T','AE_INDEX', 'SZA']

    # looping through the columns in the np array and the features list
    for col, feat in zip(range(X.shape[2]), CONFIG['params']):
      # skipping over the sw conditions unless it's the first time the calculations are done.
      # SW conditions will be the dsame for all stations.
      if (station != CONFIG['stations'][0]) and (feat in sw_feats):
        continue
      if feat in sw_feats:
        mean_df['{0}'.format(feat)] = X_mean[:,col]
        std_df['{0}'.format(feat)] = X_std[:,col]
        max_df['{0}'.format(feat)] = X_max[:,col]
      else:
        # specifically labeling the station data
        mean_df['{0}_{1}'.format(station, feat)] = X_mean[:,col]
        std_df['{0}_{1}'.format(station, feat)] = X_std[:,col]
        max_df['{0}_{1}'.format(station, feat)] = X_max[:,col]

  return mean_df, std_df, max_df


def calculating_scores(df, splits, station):
  '''
    Calculating the hss, area under the precision recall curve, and rmse scores
    then putting them in a list. Then getting a 95 percent confidence level for
    each metric. Also creates a spread dataframe which measures the difference
    between the 97.5th percentile and the 2.5th percentile. THis is used to
    compare to the testing input data to see how the input data affects the
    differences in the models. The diffrence is also calculated and a dataframe
    created for the same purpose. Here the difference is the absoulute value
    of the crossing ground truth and the mean model prediction.

    Args:
      df (pd.dataframe): results df from the testing predictions
      splits (int): number of train-val splits
      station (str): 3 diget code for the station models being examined.

    Returns:
      pd.dataframes (4): dataframes containing the precision-recall values for
                          plotting, the metric scores, the spead and difference dfs
	'''


  # Definging the index column for saving the metric results
  index_column = ['mean', 'max' ,'min']

  prec_recall = pd.DataFrame()
  metrics = pd.DataFrame({'ind':index_column})

  bias, std_pred, rmse, area_uc, precision, recall, hss = [], [], [], [], [], [], []      # initalizing the lists for storing the individual scores
  diff_df, spread_df = pd.DataFrame(), pd.DataFrame()

  # setting the index for the spead and diff data frames
  diff_df['Date_UTC'] = df.index
  spread_df['Date_UTC'] = df.index
  diff_df.set_index('Date_UTC', inplace=True)
  spread_df.set_index('Date_UTC', inplace=True)


  # segmenting a df containing only the model predictions and calculating the percentiles and spread
  newdf = df[['predicted_split_{0}'.format(split) for split in range(splits)]]
  top_perc = newdf.quantile(0.975, axis=1)
  bottom_perc = newdf.quantile(0.025, axis=1)
  spread_df[station] = top_perc - bottom_perc


  # looping through the number of plots and calculating the values for each model's output
  for split in range(splits):

    # creating a temporary df for just the ground truth data and this model's prediction
    temp_df = df[['crossing', 'predicted_split_{0}'.format(split)]]

    # metric calcualtions cannot handle nan so they are dropped
    temp_df.dropna(inplace=True)

    pred = temp_df['predicted_split_{0}'.format(split)]        # Grabbing the specific predicted data
    re = temp_df['crossing']                   # and the real data for comparison

    # calculating the difference and adding it as a column to the diff df for this split
    diff_df['{0}_split_{1}'.format(station, split)] = abs(re-pred)

    # getting the precision and recall, then calculating the auc for this split
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

    # if model is perfect but not all elements are filled, hss is set to one
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


  # getting the median of the auc and getting the precision recall curves that correspond
  try:
    medIdx = area_uc.index(np.percentile(area_uc,50,interpolation='nearest'))  # type: ignore
    prec_recall['prec'] = precision[medIdx]
    prec_recall['rec'] = recall[medIdx]

  except ValueError:
    print('No AUC scores for this station and storm')

  # turning the matric lists into arrays so the mean and percentiles can be calculated
  hss = np.array(hss)
  bias = np.array(bias)
  std_pred = np.array(std_pred)
  rmse = np.array(rmse)
  area_uc = np.array(area_uc)

  # defining the top and bottom percentiles to be calculated. Corresponds to the 95th percentile
  max_perc = 97.5
  min_perc = 2.5

  metrics['HSS'] = [np.mean(hss), np.percentile(hss, max_perc), np.percentile(hss, min_perc)]
  metrics['BIAS'] = [np.mean(bias), np.percentile(bias, max_perc), np.percentile(bias, min_perc)]
  metrics['STD_PRED'] = [np.mean(std_pred), np.percentile(std_pred, max_perc), np.percentile(std_pred, min_perc)]
  metrics['RMSE'] = [np.mean(rmse), np.percentile(rmse, max_perc), np.percentile(rmse, min_perc)]
  metrics['AUC'] = [np.mean(area_uc), np.percentile(area_uc, max_perc), np.percentile(area_uc, min_perc)]

  # setting the index of the metrics to make the plotting easier
  metrics.set_index('ind', drop=True, inplace=True)

  return prec_recall, metrics, diff_df, spread_df


def aggregate_results(length, splits, station):
  '''
  Loads the data for each station/storm/split, calculates the metrics scores,
  combines them into one data frame, and gets the persistance scores for comparison.

  Args:
      length (int): number of testing storms
      splits (int): number of train-val splits and unique models for each station
      station (str): 3 diget code describing the station being examined.

  Returns:
      dict: contains all metric results for this station
  '''

  # initalizing the dict
  results_dict = {}
  for i in range(length):

    # creating a dict for each storm
    results_dict['storm_{0}'.format(i)] = {}

    # loading the model results for this storm
    results_dict['storm_{0}'.format(i)]['raw_results'], results_dict['storm_{0}'.format(i)]['sw_results'] = load_feather(station, i)

    # calculating the metric scores
    prec_recall, metrics, diff_df, spread_df = calculating_scores(results_dict['storm_{0}'.format(i)]['raw_results'], splits, station)
    sw_prec_recall, sw_metrics, ___, ___ = calculating_scores(results_dict['storm_{0}'.format(i)]['sw_results'], splits, station)

    # assigning the different dataframes to dict keys
    results_dict['storm_{0}'.format(i)]['diff_df'] = diff_df
    results_dict['storm_{0}'.format(i)]['spread_df'] = spread_df
    results_dict['storm_{0}'.format(i)]['precision_recall'] = prec_recall
    results_dict['storm_{0}'.format(i)]['metrics'] = metrics
    results_dict['storm_{0}'.format(i)]['sw_precision_recall'] = sw_prec_recall
    results_dict['storm_{0}'.format(i)]['sw_metrics'] = sw_metrics
    results_dict['storm_{0}'.format(i)]['STD_real'] = results_dict['storm_{0}'.format(i)]['raw_results']['crossing'].std()

  # combining all of the storms to get one metric result across all storms. Repeating above steps
  results_dict['all_storms_df'] = pd.concat([results_dict['storm_{0}'.format(i)]['raw_results'] for i in range(length)], axis=0, ignore_index=True)
  results_dict['all_sw_storms_df'] = pd.concat([results_dict['storm_{0}'.format(i)]['sw_results'] for i in range(length)], axis=0, ignore_index=True)

  prec_recall, metrics, ___, ___ = calculating_scores(results_dict['all_storms_df'], splits, station)
  sw_prec_recall, sw_metrics, ___, ___ = calculating_scores(results_dict['all_sw_storms_df'], splits, station)
  results_dict['total_metrics'] = metrics
  results_dict['total_precision_recall'] = prec_recall
  results_dict['total_sw_metrics'] = sw_metrics
  results_dict['total_sw_precision_recall'] = sw_prec_recall

  hss, auc, rmse, bias = getting_persistance_results(results_dict['all_storms_df'])
  results_dict['pers_HSS'] = hss
  results_dict['pers_AUC'] = auc
  results_dict['pers_RMSE'] = rmse
  results_dict['pers_BIAS'] = bias

  return results_dict


def getting_persistance_results(df):
  '''
  Calculates the results for the persistance models for comparison to the
  neural netwrok model outputs. Outputs just one value for each metric
  instead of a mean and percentile values.

  Args:
      df (pd.dataframe): dataframe containing the persistance model and the ground truth

  Returns:
      float (3): hss, auc, rmse, and bias for persistance model
  '''

  rmse, area_uc, hss, bias = [], [], [], []      # initalizing the lists for storing the individual scores

  # creating the temp df containing jsut the relevent columns and dropping the nan rows
  temp_df = df[['crossing', 'persistance']]
  temp_df.dropna(inplace=True)

  pred = temp_df['persistance']        # Grabbing the specific predicted data
  re = temp_df['crossing']             # and the real data for comparison
  bias.append(pred.mean()-re.mean())

  # getting the precision and recall arrays and calculating the area under the curve
  prec, rec, ____ = precision_recall_curve(re, pred)
  area = auc(rec, prec)

  # segmenting the pd.series into the confusion matrix indicies
  A = temp_df[(temp_df['persistance'] >= 0.5) & (temp_df['crossing'] == 1)]
  B = temp_df[(temp_df['persistance'] >= 0.5) & (temp_df['crossing'] == 0)]
  C = temp_df[(temp_df['persistance'] < 0.5) & (temp_df['crossing'] == 1)]
  D = temp_df[(temp_df['persistance'] < 0.5) & (temp_df['crossing'] == 0)]

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
  area_uc.append(area.round(decimals=3))
  rmse.append(np.sqrt(mean_squared_error(re,pred)))             # calculating the root mean square error

  return hss, area_uc, rmse, bias


def plotting_corrs(diff_df, spread_df, params, name):
  '''
  Plots the spread and differences from the models as a function of
  the std, mean, or max for various input features. Saves the resulting plots.

  Args:
      diff_df (pd.dataframe): dataframe containing the differences for each station and storm
      spread_df (pd.dataframe): dataframe containing the spreads for each station and storm
      params (str or list of strs): input param or list of params to plot against the spread
                                      and difference
      name (str): std, mean, or max or the input parameters to be compared.
  '''

  # defining which are the solar wind features
  sw_feats = ['B_Total', 'BY_GSM', 'BZ_GSM', 'Vx', 'Vy', 'Vz', 'proton_density', 'T','AE_INDEX', 'SZA']

  # looping through the parameters
  for param in params:
    x, y = pd.Series(), pd.Series()
    fig = plt.figure()
    plt.title('{0} {1} vs. Difference'.format(param, name))

    # looping through the stations and splits and concating all of the results together
    for station in CONFIG['stations']:
      for i in range(CONFIG['splits']):
        x = pd.concat([x,diff_df['{0}_split_{1}'.format(station, i)]])

        # all stations see same sw params, no need to loop over them all
        if param in sw_feats:
          y = pd.concat([y,diff_df[param]])
        else:
          y = pd.concat([y,diff_df['{0}_{1}'.format(station, param)]])
    temp_df = pd.DataFrame({'x':x,
                            'y':y})
    temp_df.dropna(inplace=True)
    x = np.array(temp_df['x'])
    y = np.array(temp_df['y'])

    # creates a 2d histogram for comparison and saving it
    plt.hist2d(x,y, bins=(100,100), norm=mpl.colors.LogNorm(), range=((0.01,1), (y.min(),y.max())))
    plt.savefig('plots/{0}_{1}_vs_difference.png'.format(param, name))

    # repeats the procedure for the spred comparison plots
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
  '''
  Pulling all the functions together
  '''

  # creating a dict for saving each station's metric results
  stations_dict = {}

  # looping through the stations and pulling together the metric results
  for station in CONFIG['stations']:
    results_dict = aggregate_results(len(CONFIG['test_storm_stime']), CONFIG['splits'], station)

    # saving the resulting metric df to the staion key
    stations_dict[station] = results_dict

  # # concatenating together all the diff and spread dfs
  # diff_df, spread_df = pd.DataFrame(), pd.DataFrame()
  # for i in range(len(CONFIG['test_storm_stime'])):
  #   temp_diff, temp_spread = pd.DataFrame(), pd.DataFrame()
  #   for station in CONFIG['stations']:
  #     temp_diff = pd.concat([temp_diff, stations_dict[station]['storm_{0}'.format(i)]['diff_df']], axis=1, ignore_index=False)
  #     temp_spread = pd.concat([temp_spread, stations_dict[station]['storm_{0}'.format(i)]['spread_df']], axis=1, ignore_index=False)

  #   diff_df = pd.concat([diff_df, temp_diff], axis=0)
  #   spread_df = pd.concat([spread_df, temp_spread], axis=0)

  # # getting the max std and mean dfs
  # mean_df, std_df, max_df = getting_model_input_data()
  # diff_df.reset_index(inplace=True, drop=True)
  # spread_df.reset_index(inplace=True, drop=True)

  # # creating dfs for all the stat and model result combinations to make the plotting easier
  # mean_diff_df = pd.concat([diff_df, mean_df], axis=1, ignore_index=False)
  # std_diff_df = pd.concat([diff_df, std_df], axis=1, ignore_index=False)
  # max_diff_df = pd.concat([diff_df, max_df], axis=1, ignore_index=False)

  # mean_spread_df = pd.concat([spread_df, mean_df], axis=1, ignore_index=False)
  # std_spread_df = pd.concat([spread_df, std_df], axis=1, ignore_index=False)
  # max_spread_df = pd.concat([spread_df, max_df], axis=1, ignore_index=False)

  # # putting all the correlation dfs into the plotting functions
  # plotting_corrs(mean_diff_df, mean_spread_df, CONFIG['params'], 'mean')
  # plotting_corrs(std_diff_df, std_spread_df, CONFIG['params'], 'std')
  # plotting_corrs(max_diff_df, max_spread_df, CONFIG['params'], 'max')

  print(stations_dict['OTT']['storm_4']['sw_results'].shape)
  print('THIS ONE!')
  # saving the dict with all the station metric results for plotting
  with open('outputs/stations_results_dict.pkl', 'wb') as f:
    pickle.dump(stations_dict, f)


if __name__ == '__main__':

  main()    # calling the main function.

  print('It ran. Good job!')       # if we get here we're doing alright.


