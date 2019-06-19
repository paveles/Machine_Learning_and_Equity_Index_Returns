#%% #--------------------------------------------------

import warnings
import math
import time
import datetime
import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd # To Determine the End of the Corresponding Month
import sys # To caclulate memory usage
import os
import seaborn as sns
import matplotlib.pyplot as plt
#%% #--------------------------------------------------
#* Global Parameters *
# Add interactions or not
Poly = 1 # 1 - no polynomial features, 2 - first order interactions 

# Starting Year: 1928 - macro only, 1951 - macto + technical, 
# 1974 - add short interest    
Period  = 1951
# Number of Lags
LAGS = 1

# Estimate using Rolling Window or Exapnding
ROLLING = True
min_idx = 0
start_idx = 180

# Rolling or Exapnding Window
if ROLLING == True:
    Models_Folder = 'rolling'
else:
    Models_Folder = 'expanding'

VERBOSE = True
#%% #--------------------------------------------------

#* Load Data
df0 = pd.read_csv('in/rapach_2013.csv', na_values = ['NaN'])
df0.rename( index=str, columns={"date": "ym"}, inplace=True)
df0['date'] = pd.to_datetime(df0['ym'],format='%Y%m') + MonthEnd(1)
df0 = df0.sort_values(by=['date'])
df0.index = df0.index.astype(int)
dff = df0.iloc[y_pred.index]
t= dff['date']
rm_rf = dff['sp500_rf']

#%% #--------------------------------------------------
# Load Strategy
config = enet_config

df_pred = pd.read_csv('out/'+ Models_Folder +'/models/'+ config['name']+'_predictions.csv').set_index('index',drop = True)
y_pred = df_pred['y_pred']
scores_estimated = df_pred['scores_estimated']
y_moving_mean = yo.shift(1).expanding(1).mean().loc[y_pred.index]
y_true =rm_rf*100


#%% #--------------------------------------------------
#* Plot one-month ahead foreacst
data=pd.concat([y_true,y_pred,t],axis = 1)
#sns.scatterplot(x='sp500_rf',y ='y_pred',data=data)
#%% #--------------------------------------------------
data2 = data.melt(id_vars='date', var_name='model',  value_name='Monthly Return')
#sns.lineplot(x='date',y='leverage',data=Leverage_Data, palette="greys" )
plt.figure()
sns.lineplot(x='date',y='Monthly Return', hue ='model', data = data2, palette="deep6" )
#%% #--------------------------------------------------
#%% #--------------------------------------------------
# Strategy Performace
#%% #--------------------------------------------------

def calculate_cumulative_return(rm_rf):
    rm_rf_1 = (rm_rf+1)
    ln_rm_rf_1 = rm_rf_1.apply(lambda x: math.log(x))
    cumr=ln_rm_rf_1.cumsum(axis = 0).apply(lambda x: math.exp(x))
    return cumr

#%% #--------------------------------------------------


cumr = calculate_cumulative_return(rm_rf).rename('sp500')

# Constuct Enet Strategy

leverage = y_pred>0
#/abs(y_moving_mean)

leverage = np.maximum(leverage,0)
leverage = np.minimum(leverage, 1)


Leverage = pd.Series(leverage).rename('leverage')
Leverage_Data  =pd.concat([Leverage,t],axis = 1)
plt.figure()
sns.lineplot(x='date',y='leverage',data=Leverage_Data)
#%% #--------------------------------------------------

#Define Strategy based on ENET
strategy_return = leverage*rm_rf

strategy_cumreturn = calculate_cumulative_return(strategy_return)


strategy_cumreturn =strategy_cumreturn.rename(config['name'])

#%% #--------------------------------------------------

data=pd.concat([cumr,strategy_cumreturn,t],axis = 1)

data = data.melt(id_vars='date', var_name='model',  value_name='value of 1$')


#sns.lineplot(x='date',y='leverage',data=Leverage_Data, palette="greys" )
plt.figure()
sns.lineplot(x='date',y='value of 1$', hue ='model', data = data, palette="deep6" )
#%% #--------------------------------------------------


