""" 
Predicting Equity Index Returns using Machine Learning Methods - Data Preparation File
"""
#%% #--------------------------------------------------
#* Load Data
import pickle
import pandas as pd
from pandas.tseries.offsets import MonthEnd # To Determine the End of the Corresponding Month
df = pd.read_csv('data/raw/neely_2014.csv', na_values = ['NaN'])
df.rename( index=str, columns={"date": "ym"}, inplace=True)
df['date'] = pd.to_datetime(df['ym'],format='%Y%m') + MonthEnd(1)
df['sp500_rf'] = df['sp500_rf'] * 100
df['lnsp500_rf'] = df['lnsp500_rf'] * 100
df = df.sort_values(by=['date'])
df.index = df.index.astype(int)
df.to_pickle("data/processed/df.pickle")
