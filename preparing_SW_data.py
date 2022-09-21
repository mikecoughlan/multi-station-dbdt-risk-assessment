##########################################################################################
#
#
#
#
#
#
#
#
##########################################################################################

import datetime as dt
import glob
import os
from pickle import dump, load

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyhdf.HDF import *
from pyhdf.VS import *

os.environ["CDF_LIB"] = "~/lib"

import cdflib

omni_dir = '../../data/omni/'
plasmaDir = '../../data/ace/'
magDir = '../../data/ace/'
dataDump = '../data/SW/'

method = 'linear'
limit = 15

if not os.path.exists(dataDump):
	os.makedirs(dataDump)


def break_dates(df, dateField, drop=False, errors="raise"):
    """break_dates expands a column of df from a datetime64 to many columns containing
    the information from the date. This applies changes inplace.
    Parameters:
    -----------
    df: A pandas data frame. df gain several new columns.
    dateField: A string that is the name of the date column you wish to expand.
        If it is not a datetime64 series, it will be converted to one with pd.to_datetime.
    drop: If true then the original date column will be removed.

    Modified from FastAI software by Victor Pinto.
    """
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
    """
    Load a CDF File and convert it in a Pandas DataFrame.

    WARNING: This will not return the CDF Attributes, just the variables.
    WARNING: Only works for CDFs of the same array lenght (OMNI)
    """
    cdf = cdflib.CDF(file)
    cdfdict = {}

    for key in cdf.cdf_info()['zVariables']:
        cdfdict[key] = cdf[key]

    cdfdf = pd.DataFrame(cdfdict)

    if 'Epoch' in cdf.cdf_info()['zVariables']:
        cdfdf['Epoch'] = pd.to_datetime(cdflib.cdfepoch.encode(cdfdf['Epoch'].values))

    return cdfdf

def clean_omni(df):
    """
    Remove filling numbers for missing data in OMNI data (1 min) and replace
    them with np.nan values

    """

    # Indices
    df.loc[df['AE_INDEX'] >= 99999, 'AE_INDEX'] = np.nan
    df.loc[df['AL_INDEX'] >= 99999, 'AL_INDEX'] = np.nan
    df.loc[df['AU_INDEX'] >= 99999, 'AU_INDEX'] = np.nan
    df.loc[df['SYM_D'] >= 99999, 'SYM_D'] = np.nan
    df.loc[df['SYM_H'] >= 99999, 'ASY_D'] = np.nan
    df.loc[df['ASY_H'] >= 99999, 'ASY_H'] = np.nan
    df.loc[df['PC_N_INDEX'] >= 999, 'PC_N_INDEX'] = np.nan

    return(df)

def get_indicies_from_omni():
	'''
	Gets the AE_INDEX and the SYM_H indicies from the OMNI data as this is not contained in the ACE data.

	Returns:
		pd.Dataframe: pd.dataframe with the indicies and a column labeled Epoch containing teh datatime stamp
	'''

	syear = 1995
	eyear = 2019

	start_time = str(pd.Timestamp(syear,1,1))
	start_time = start_time.replace(' ', '').replace('-', '').replace(':', '')
	end_time = str(pd.Timestamp(eyear,12,31,23,59,59))
	end_time = end_time.replace(' ', '').replace('-', '').replace(':', '')

	############################################################################################
	####### Load and pre-process solar wind data
	############################################################################################

	omniFiles = glob.glob(omni_dir+'*/*.cdf', recursive=True)

	o = []
	for fil in sorted(omniFiles):
		cdf = omnicdf2dataframe(fil)
		o.append(cdf)

	omni_start_time = str(pd.Timestamp(syear,1,1))
	omni_start_time = omni_start_time.replace(' ', '').replace('-', '').replace(':', '')
	omni_end_time = str(pd.Timestamp(eyear,12,31,23,59,59))
	omni_end_time = omni_end_time.replace(' ', '').replace('-', '').replace(':', '')

	omniData = pd.concat(o, axis = 0, ignore_index = True)
	omniData.index = omniData.Epoch
	omniData = omniData[omni_start_time:omni_end_time]

	to_drop = ['PLS', 'IMF_PTS', 'PLS_PTS', 'percent_interp',
			'Timeshift', 'RMS_Timeshift', 'RMS_phase', 'Time_btwn_obs',
			'RMS_SD_B', 'RMS_SD_fld_vec', 'Vx', 'Vy', 'Vz', 'proton_density',
			'BZ_GSM', 'BY_GSM', 'BX_GSE', 'BY_GSE', 'BZ_GSE', 'F',
			'flow_speed', 'T', 'Pressure', 'E', 'Beta', 'Mach_num',
			'Mgs_mach_num', 'Epoch', 'YR', 'Day', 'HR', 'Minute']

	omniData = omniData.drop(to_drop, axis=1)
	clean_omni(omniData)
	omniData['AE_INDEX'].interpolate(method=method, limit=limit)
	omniData['SYM_H'].interpolate(method=method, limit=limit)

	return omniData


def ace_to_dataframe(file, dataType):
    """
    Load ACE HDF4 SWESWI file and convert it to Pandas DataFrame

    ** Will get only the 12 minutes data from file "SWESWI_data_12min"
    """

    if dataType == 'sweswi':
        dType = 'SWESWI_data_12min'
    if dataType == 'mag':
        dType = 'MAG_data_16sec'

    hdf = HDF(file)
    vs = hdf.vstart()
    vd = vs.attach(dType)

    df = pd.DataFrame(vd[:], columns=vd._fields)

    vd.detach()
    vs.end()
    hdf.close()

    return df

def bad_ace_to_nan(df, dataType):
    if dataType == 'sweswi':

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

    return(df)

def ace_as_omni(plasmaData, magData, delay=0):

    plasmaData = plasmaData.interpolate(method=method, limit=limit)
    plasmaData = plasmaData.resample('1 min').bfill()

    # magData = magData[sdate:edate]
    magData = magData.interpolate(method=method, limit=limit)
    magData = magData.resample('1 min').mean()

    aceData = pd.DataFrame()

    aceData['B_Total'] = magData['Bt']
    aceData['BY_GSM'] = magData['Bgsm_y']
    aceData['BZ_GSM'] = magData['Bgsm_z']
    aceData['Vx'] = plasmaData['proton_speed']
    aceData['Vy'] = plasmaData['y_dot_GSM']
    aceData['Vz'] = plasmaData['z_dot_GSM']
    aceData['proton_density'] = plasmaData['proton_density']
    aceData['T'] = plasmaData['proton_temp']
    aceData['Pressure'] = (2*1e-6)*aceData['proton_density']*aceData['Vx']**2
    aceData['E_Field'] = -aceData['Vx'] * aceData['BZ_GSM'] * 1e-3

    return aceData

def processing_ACE():

	plasmaFiles = glob.glob(plasmaDir+'sweswi_data_12min_year*.hdf', recursive=True)
	magFiles = glob.glob(magDir+'mag_data_16sec_year*.hdf', recursive=True)

	p, m = [], []
	for fil in sorted(plasmaFiles):
		acePlasma = ace_to_dataframe(fil, 'sweswi')
		acePlasma.index = pd.to_datetime(acePlasma['ACEepoch'], unit='s', origin=pd.Timestamp('1996-01-01'))  # type: ignore
		acePlasma = bad_ace_to_nan(acePlasma, 'sweswi')
		p.append(acePlasma)

	for fil in sorted(magFiles):
		aceMag = ace_to_dataframe(fil, 'mag')
		aceMag.index = pd.to_datetime(aceMag['ACEepoch'], unit='s', origin=pd.Timestamp('1996-01-01'))  # type: ignore
		aceMag = bad_ace_to_nan(aceMag, 'mag')
		m.append(aceMag)

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
	'''
	omniData = get_indicies_from_omni()
	aceData = processing_ACE()
	df = combining_dfs(omniData=omniData, aceData=aceData)

	omniData.reset_index(drop=False, inplace=True)
	aceData.reset_index(drop=False, inplace=True)

	omniData.to_feather(dataDump+'indicies_data.feather')
	aceData.to_feather(dataDump+'ace_data.feather')
	df.to_feather(dataDump+'solarwind_and_indicies.feather')



if __name__ == '__main__':

	main()

	print('It ran. Good job!')
