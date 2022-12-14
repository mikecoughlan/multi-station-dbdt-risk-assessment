############################################################################################
#
#	multi-station-dbdt-risk-assessment/preparing_mag_data.py
#
#	File for preparing the ground magnetometer data from the SuperMAG stations. Standardizes
# 	column names and interpolates up to 15 minutes of missing data. Saves the data as an
# 	external file for use in later scripts.
#
#	SCRIPT ADAPTED FROM SIMILAR SCRIPT WRITTEN BY VICTOR A. PINTO
############################################################################################

# importing relevent packages
import datetime as dt
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# defining reletive file paths
dataDir = '../../../../../data/supermag/baseline/'

# defining the beginning and ending years.
syear = 1995
eyear = 2019

start_time = str(pd.Timestamp(syear,1,1))
start_time = start_time.replace(' ', '').replace('-', '').replace(':', '')
end_time = str(pd.Timestamp(eyear,12,31,23,59,59))
end_time = end_time.replace(' ', '').replace('-', '').replace(':', '')

# listing stations being examined in this work
stations = ['VIC', 'NEW', 'OTT', 'STJ',
			'ESK', 'LER', 'WNG',
			'BFE']

method = 'linear'	# defining the interpolation method
limit = 15			# defining the limit of interploation

# defining the dir for saving the resulting data
dataDump = '../../data/supermag/'

# creating it if it doesn't already exist
if not os.path.exists(dataDump):
	os.makedirs(dataDump)

############################################################################################
####### Load and pre-process magnetometer data
############################################################################################
for station in stations:
	magFiles = sorted(glob.glob(dataDir+'{0}/{1}-*-supermag-baseline.csv'.format(station,station)))
	m = []
	# Load original mag data for training years
	# Convert original time to Pandas Datetime format, use datetime as DataFrame index
	# and then fill in the missing date values (filling with NaN for now)
	#entry = dataDir+'%s/%s-%s-supermag.csv' % (stations[0], stations[0], years[0])
	for entry in magFiles:
		df = pd.read_csv(entry)
		df.drop('IAGA', axis=1, inplace=True)
		df['Date_UTC'] = pd.to_datetime(df['Date_UTC'])
		df.set_index('Date_UTC', inplace=True, drop=True)
		df = df.reindex(pd.date_range(start=dt.datetime(df.index[0].year, 1, 1), end=dt.datetime(df.index[0].year, 12, 31, 23, 59), freq='1 Min'), copy=True, fill_value=np.NaN)  # type: ignore
		df['Date_UTC'] = df.index

		# Add magnitude and differential values
		df.rename(columns={'MAGLAT': 'MLAT'}, inplace=True)
		df.rename(columns={'dbn_nez': 'N'}, inplace=True)
		df.rename(columns={'dbe_nez': 'E'}, inplace=True)
		df.rename(columns={'dbz_nez': 'Z'}, inplace=True)
		df['MAGNITUDE'] = np.sqrt(df['N'] ** 2 + df['E'] ** 2 + df['Z'] ** 2)

		m.append(df)

	# Concatenate all DataFrames in a single DataFrame
	magData = pd.concat(m, axis = 0, ignore_index=True)
	magData.index = magData.Date_UTC
	magData = magData[start_time:end_time]

	# interpolating over missing data within a limit
	magData['Z'] = magData.Z.interpolate(method=method, limit=limit)
	magData['E'] = magData.E.interpolate(method=method, limit=limit)
	magData['N'] = magData.N.interpolate(method=method, limit=limit)
	magData['MAGNITUDE'] = magData.MAGNITUDE.interpolate(method=method, limit=limit)

	magData['SZA'] = magData.SZA.interpolate(method=method, limit=limit)
	magData['IGRF_DECL'] = magData.IGRF_DECL.interpolate(method=method, limit=limit)

	magData['MLAT'] = magData.MLAT.interpolate(method=method, limit=limit)
	magData['MLT'] = magData.MLT.interpolate(method=method, limit=limit)

	magData.reset_index(inplace=True, drop=True)

	# saving as feather to conserve memory and imporve perfromance
	magData.to_feather(dataDump+'{0}_{1}_interp.feather'.format(station, limit))

	print('{0} completed'.format(station))


