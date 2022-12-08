############################################################################################
#
#	multi-station-dbdt-risk-assessment/preparing_SW_data.py
#
#	File for preparing the raw solar wind data from both ACE and OMNI. Takes the source
# 	files, up or down samples the ACE data as necessary to the 1-minute resoultion. Changes
# 	missing data format from eg. 999.999 to np.nan. Interpolates up to 15 minutes of missing
# 	data. Saves the data as an external file for use in later scripts.
#
#	SCRIPT ADAPTED FROM SIMILAR SCRIPT WRITTEN BY VICTOR A. PINTO
############################################################################################

# importing relevent packages
import glob
import os

import numpy as np
import pandas as pd
from pyhdf.HDF import *
from pyhdf.VS import *

os.environ["CDF_LIB"] = "~/lib"

import cdflib

# defining reletive file paths
omni_dir = '../../../../../data/omni/hro_1min/'
plasmaDir = '../../../../../data/ace/swepam/'
magDir = '../../../../../data/ace/mag/'
dataDump = '../../data/SW/'

method = 'linear'	# defining the interpolation method
limit = 15			# defining the limit of interploation

# checking to see if the final data folder exists. If not, creates it.
if not os.path.exists(dataDump):
	os.makedirs(dataDump)


def break_dates(df, dateField, drop=False, errors="raise"):
	'''
	Break_dates expands a column of df from a datetime64 to many columns containing
	the information from the date. This applies changes inplace.

	Args:
		df (pd.dataframe): df gain several new columns.
		dateField (string): A string that is the name of the date column you wish to
							expand. If it is not a datetime64 series, it will be converted
							to one with pd.to_datetime.
		drop (bool, optional): If true then the original date column will be removed.
								Defaults to False.
		errors (str, optional): if raise, will raise an error if present during datatime
								conversion. Defaults to "raise".

	Modified from FastAI software by Victor Pinto.
	'''

	field = df[dateField]
	field_dtype = field.dtype
	if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
		field_dtype = np.datetime64

	if not np.issubdtype(field_dtype, np.datetime64):
		df[dateField] = field = pd.to_datetime(field, infer_datetime_format=True, errors=errors)

	attr = ['Year', 'Month', 'Day', 'Dayofyear', 'Hour', 'Minute']

	for n in attr: df[n] = getattr(field.dt, n.lower())
	if drop: df.drop(dateField, axis=1, inplace=True)

def omnicdf2dataframe(file):
	'''
	Load a CDF File and convert it in a Pandas DataFrame.

	WARNING: This will not return the CDF Attributes, just the variables.
	WARNING: Only works for CDFs of the same array length (OMNI)

	Args:
		file (cdf file): file input for conversion to a pd.dataframe

	Returns:
		pd.dataframe: cdf file converted to a pd.dataframe. Contains
						a datetime column named "Epoch".
	'''

	cdf = cdflib.CDF(file)
	cdfdict = {}

	for key in cdf.cdf_info()['zVariables']:
		cdfdict[key] = cdf[key]

	cdfdf = pd.DataFrame(cdfdict)

	if 'Epoch' in cdf.cdf_info()['zVariables']:
		cdfdf['Epoch'] = pd.to_datetime(cdflib.cdfepoch.encode(cdfdf['Epoch'].values))

	return cdfdf

def clean_omni(df):
	'''
	Remove filling numbers for missing data in OMNI data (1 min) and replace
	them with np.nan values.

	Args:
		df (pd.dataframe): dataframe containing OMNI data to be cleaned.

	Returns:
		pd.dataframe: cleaned dataframe.
	'''

	# Changing placeholder for missing data to np.nan
	df.loc[df['AE_INDEX'] >= 99999, 'AE_INDEX'] = np.nan
	df.loc[df['AL_INDEX'] >= 99999, 'AL_INDEX'] = np.nan
	df.loc[df['AU_INDEX'] >= 99999, 'AU_INDEX'] = np.nan
	df.loc[df['SYM_D'] >= 99999, 'SYM_D'] = np.nan
	df.loc[df['SYM_H'] >= 99999, 'ASY_D'] = np.nan
	df.loc[df['ASY_H'] >= 99999, 'ASY_H'] = np.nan
	df.loc[df['PC_N_INDEX'] >= 999, 'PC_N_INDEX'] = np.nan

	return df

def get_indicies_from_omni():
	'''
	Gets the AE_INDEX and the SYM_H indicies from the OMNI data as this is not contained in the ACE data.

	Returns:
		pd.Dataframe: pd.dataframe with the indicies and a column labeled Epoch containing teh datatime stamp
	'''

	# defining the beginning and ending years. Defined by the data available from ACE.
	syear = 1998
	eyear = 2017

	############################################################################################
	####### Load and pre-process solar wind data
	############################################################################################

	omniFiles = glob.glob(omni_dir+'*/*.cdf', recursive=True) # getting file names

	# creating list of dataframes
	o = []
	for fil in sorted(omniFiles):
		cdf = omnicdf2dataframe(fil)
		o.append(cdf)

	omni_start_time = str(pd.Timestamp(syear,1,1))
	omni_start_time = omni_start_time.replace(' ', '').replace('-', '').replace(':', '')
	omni_end_time = str(pd.Timestamp(eyear,12,31,23,59,59))
	omni_end_time = omni_end_time.replace(' ', '').replace('-', '').replace(':', '')

	# combining the yearly dataframes into one large dataframe
	omniData = pd.concat(o, axis = 0, ignore_index = True)
	# setting the index to a datetime index
	omniData.index = omniData.Epoch
	# trimming the dataframe to be in the time frame of interest
	omniData = omniData[omni_start_time:omni_end_time]

	to_drop = ['PLS', 'IMF_PTS', 'PLS_PTS', 'percent_interp',
			'Timeshift', 'RMS_Timeshift', 'RMS_phase', 'Time_btwn_obs',
			'RMS_SD_B', 'RMS_SD_fld_vec', 'Vx', 'Vy', 'Vz', 'proton_density',
			'BZ_GSM', 'BY_GSM', 'BX_GSE', 'BY_GSE', 'BZ_GSE', 'F',
			'flow_speed', 'T', 'Pressure', 'E', 'Beta', 'Mach_num',
			'Mgs_mach_num', 'Epoch', 'YR', 'Day', 'HR', 'Minute']

	# dropping unnecessary columns
	omniData = omniData.drop(to_drop, axis=1)
	clean_omni(omniData)

	# interpolating the columns relevent to this work
	omniData['AE_INDEX'].interpolate(method=method, limit=limit)
	omniData['SYM_H'].interpolate(method=method, limit=limit)

	return omniData


def ace_to_dataframe(file, dataType):
	'''
	Load ACE HDF4 SWESWI file and convert it to Pandas DataFrame

	Args:
		file (cdf file): cdf file for conversion to dataframe and resampling
		dataType (str): 'swepam' if converting plasma parameters,
						'mag' if converting magnetometer parameters.

	Returns:
		pd.dataframe: converted dataframe.
	'''

	if dataType == 'swepam':
		dType = 'SWEPAM_ion'
	if dataType == 'mag':
		dType = 'MAG_data_16sec'

	hdf = HDF(file)
	vs = hdf.vstart()
	vd = vs.attach(dType)

	# converting to pd.dataframe
	df = pd.DataFrame(vd[:], columns=vd._fields)

	vd.detach()
	vs.end()
	hdf.close()

	return df

def bad_ace_to_nan(df, dataType):
	'''
	Remove filling numbers for missing data in ACE data and replace
	them with np.nan values.

	Args:
		df (pd.dataframe): ACE data that needs cleaning
		dataType (str): 'swepam' if converting plasma parameters,
						'mag' if converting magnetometer parameters.

	Returns:
		pd.dataframe: cleaned ACE data
	'''

	if dataType == 'swepam':

		if 'proton_speed' in df.columns: df.loc[df['proton_speed'] <= -9999, 'proton_speed'] = np.nan
		if 'y_dot_GSM' in df.columns: df.loc[df['y_dot_GSM'] <= -9999, 'y_dot_GSM'] = np.nan
		if 'z_dot_GSM' in df.columns: df.loc[df['z_dot_GSM'] <= -9999, 'z_dot_GSM'] = np.nan

		if 'proton_density' in df.columns: df.loc[df['proton_density'] <= -9999, 'proton_density'] = np.nan
		if 'proton_density' in df.columns: df.loc[df['proton_density'] >=  999, 'proton_density'] = np.nan
		if 'proton_temp' in df.columns: df.loc[df['proton_temp'] <=  -9999, 'proton_temp'] = np.nan

	if dataType == 'mag':
		if 'Bt' in df.columns: df.loc[df['Bt'] <= -999, 'Bt'] = np.nan
		if 'Bgse_x' in df.columns: df.loc[df['Bgse_x'] <= -999, 'Bgse_x'] = np.nan
		if 'Bgsm_y' in df.columns: df.loc[df['Bgsm_y'] <= -999, 'Bgsm_y'] = np.nan
		if 'Bgsm_z' in df.columns: df.loc[df['Bgsm_z'] <= -999, 'Bgsm_z'] = np.nan

	return df


def ace_as_omni(plasmaData, magData):
	'''
	Resampling the ACE data to 1 minute resolution, interploates to the defined limit,
	and converting column names so they match the column names from the OMNI database.

	Args:
		plasmaData (pd.dataframe): ACE plasma data
		magData (pd.dataframe): ACE magnetometer data

	Returns:
		pd.dataframe: combined dataframe of ACE plasma and mag data
	'''

	# dropping duplicate plasma dates caused by upsampling
	plasmaData.drop_duplicates(subset='ACEepoch', inplace=True)
	plasmaData = plasmaData.resample('1 min').bfill()
	plasmaData = plasmaData.interpolate(method=method, limit=limit)

	magData = magData.resample('1 min').mean()
	magData = magData.interpolate(method=method, limit=limit)

	aceData = pd.DataFrame()

	aceData['B_Total'] = magData['Bt']
	aceData['BY_GSM'] = magData['Bgsm_y']
	aceData['BZ_GSM'] = magData['Bgsm_z']
	aceData['Vx'] = plasmaData['proton_speed']
	aceData['Vy'] = plasmaData['y_dot_GSM']
	aceData['Vz'] = plasmaData['z_dot_GSM']
	aceData['proton_density'] = plasmaData['proton_density']
	aceData['T'] = plasmaData['proton_temp']

	# creating derived data
	aceData['Pressure'] = (2*1e-6)*aceData['proton_density']*aceData['Vx']**2
	aceData['E_Field'] = -aceData['Vx'] * aceData['BZ_GSM'] * 1e-3

	return aceData


def processing_ACE():
	'''
	loads the ACE data and puts it through all of the preprocessing functions.

	Returns:
		pd.dataframe: processed, resampled, and interpolated ACE data
	'''

	# Getting the yearly files
	plasmaFiles = glob.glob(plasmaDir+'swepam_data_64sec_year*.hdf', recursive=True)
	magFiles = glob.glob(magDir+'mag_data_16sec_year*.hdf', recursive=True)

	p, m = [], []	# p and m lists for the resulting plasma and mag data

	for fil in sorted(plasmaFiles):
		acePlasma = ace_to_dataframe(fil, 'swepam')
		acePlasma.index = pd.to_datetime(acePlasma['ACEepoch'], unit='s', origin=pd.Timestamp('1996-01-01'))  # assigning the index to datatime
		acePlasma = bad_ace_to_nan(acePlasma, 'swepam')
		p.append(acePlasma)

	for fil in sorted(magFiles):
		aceMag = ace_to_dataframe(fil, 'mag')
		aceMag.index = pd.to_datetime(aceMag['ACEepoch'], unit='s', origin=pd.Timestamp('1996-01-01'))  # assigning the index to datatime
		aceMag = bad_ace_to_nan(aceMag, 'mag')
		m.append(aceMag)

	# concatinating the yearly data together
	acePlasma = pd.concat(p, axis = 0, ignore_index = False)
	aceMag = pd.concat(m, axis = 0, ignore_index = False)

	aceData = ace_as_omni(acePlasma, aceMag)

	return aceData


def combining_dfs(omniData, aceData):
	'''
	takes the indicies from OMNI and adds them to the ACE data

	Args:
		omniData (pd.dataframe): indicies data from OMNI
		aceData (pd.Dataframe): solar wind data from ACE at the L1 point

	Returns:
		pd.dataframe: combined dataframe
	'''
	df = pd.concat([aceData, omniData], axis=1, ignore_index=False)
	df.reset_index(drop=False, inplace=True)
	df.rename(columns={'index': 'Date_UTC'}, inplace=True)

	return df

def main():
	'''
	Main function calling both the indicies and the ACE data processing functions.
	Saves the individual ACE and OMNI data as well as the combined dataset.
	'''
	print('Entering main of preparing SW')

	aceData = processing_ACE()
	omniData = get_indicies_from_omni()
	df = combining_dfs(omniData=omniData, aceData=aceData)

	omniData.reset_index(drop=False, inplace=True)
	aceData.reset_index(drop=False, inplace=True)

	omniData.to_feather(dataDump+'indicies_data_{0}_interp.feather'.format(limit))
	aceData.to_feather(dataDump+'ace_data_{0}_interp.feather'.format(limit))
	df.to_feather(dataDump+'solarwind_and_indicies_{0}_interp.feather'.format(limit))



if __name__ == '__main__':

	main()

	print('It ran. Good job!')
