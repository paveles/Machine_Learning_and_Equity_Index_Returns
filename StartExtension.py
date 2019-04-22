#%% [markdown] #--------------------------------------------------
## Equity Premium and Machine Learning - Extension
#%% #-----------------------------------------------------------
#! Define Functions
#? Read Rapach 2013 Data
def data_read(path):
    df = pd.read_csv(path, na_values = ['NaN'])
    df.rename( index=str, columns={"date": "ym"}, inplace=True)
    df['date'] = pd.to_datetime(df['ym'],format='%Y%m') + MonthEnd(1)
    df['sp500_rf'] = df['sp500_rf'] * 100
    df['lnsp500_rf'] = df['lnsp500_rf'] * 100
    df.sort_values(by=['ym']);
    return df

#? Data Transformation
def data_transform(df):
    # Important! Lagging by 1
    df['recessionD_c'] = df['recessionD']
    vars = ['recessionD', 'dp', 'dy', 'ep', 'de', \
            'rvol', 'bm', 'ntis', 'tbl', 'lty', 'ltr', 'tms', 'dfy', 'dfr', 'infl', \
            'ma_1_9', 'ma_1_12', 'ma_2_9', 'ma_2_12', 'ma_3_9', 'ma_3_12', 'mom_9', \
            'mom_12', 'vol_1_9', 'vol_1_12', 'vol_2_9', 'vol_2_12', 'vol_3_9', \
            'vol_3_12', 'sento ', 'sent', 'dsento', 'dsent', 'ewsi']
    df[vars] = df[vars].shift(1)
    # Define variables
    other = ['ewsi']
    state = ['recessionD', 'sent']
    macro = [ 'dp', 'dy', 'ep', 'de', 'rvol', 'bm', 'ntis', 'tbl', 'lty', 'ltr', 'tms', 'dfy', 'dfr', 'infl'] 
    tech = ['ma_1_9', 'ma_1_12', 'ma_2_9', 'ma_2_12', 'ma_3_9', 'ma_3_12', 'mom_9', \
            'mom_12', 'vol_1_9', 'vol_1_12', 'vol_2_9', 'vol_2_12', 'vol_3_9', \
            'vol_3_12']
    return df, macro, tech, state, other
#? Cut Sample Period
def sample_cut(df, period = None):
"""
Sample Cut
"""
    if Period is None:
        predictors = macro + tech
    elif Period == 1974:
        df = df[(df['date'].dt.year >= 1974)&(df['date'].dt.year <= 2010)]
        predictors = macro + tech + other  + state
    elif Period == 1928:
        df = df[(df['date'].dt.year >= 1928)]
        predictors = macro
    elif Period == 1951:
        df = df[(df['date'].dt.year >= 1951)]
        predictors = macro+ tech
    else:
        sys.exit("Wrong Sample")
    return df, predictors
#%% #--------------------------------------------------
#! Import Libraries and Do Settings
import warnings
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]
import math
import time
import datetime
import sklearn
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.rcParams['patch.force_edgecolor'] = True
import seaborn as sns
sns.set()
from pandas.tseries.offsets import MonthEnd # To Determine the End of the Corresponding Month
import sys # To caclulate memory usage
import os
# dir = 'C:\Research\Google Search Volume'
dir = os.getcwd()
os.makedirs(dir + '/temp', exist_ok = True)
os.makedirs(dir + '/out/temp', exist_ok = True)
os.makedirs(dir + '/in', exist_ok = True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler, PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures

#%% #--------------------------------------------------
#! Global Parameters
#* Cross-validation Parameter
K = 10
#*  Share of Sample as Test
TsizeInv = 15
test_size= 1/TsizeInv
#* Add interactions or not
Poly = 1
#* Period
Period  = 1974

#%% #--------------------------------------------------
#! Analysis Sequence
#?Prepare
df = data_read('in/rapach_2013.csv')
df, macro, tech, state, other = data_transform(df)
df, predictors = sample_cut(df, period = Period)

#? Describe
df.describe().T 
#""" --> Data is the same is in the paper Rapach et al 2013"""
#%% #--------------------------------------------------
#?''' Train and Test Samples'''
from sklearn.model_selection import train_test_split
Xo= df[predictors]
yo = df['lnsp500_rf']
X, X_test, y, y_test = train_test_split(Xo, yo, test_size=test_size, shuffle = False )






#%%
