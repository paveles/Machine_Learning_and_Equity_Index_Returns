#%% #--------------------------------------------------
#*Load Modules
from model_configs import*
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
sns.set()
sns.set(font_scale=1.5)
#sns.set(rc={'figure.facecolor':'white'})
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]
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
ROLLING = False
min_idx = 0
start_idx = 180

# Rolling or Expanding Window
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

#%% #--------------------------------------------------
#* Load Strategy
config = enet_config

df_pred = pd.read_csv('out/'+ Models_Folder +'/models/'+ config['name']+'_predictions.csv').set_index('index',drop = True)
y_pred = df_pred['y_pred']
scores_estimated = df_pred['scores_estimated']

dff = df0.iloc[y_pred.index]
t= dff['date']
rm_rf = dff['sp500_rf']
y_moving_mean = df0['sp500_rf'].shift(1).expanding(1).mean().loc[y_pred.index]
y_true =rm_rf*100


#%% #--------------------------------------------------
#* Plot one-month ahead foreacst
data=pd.concat([y_true.rename('Realized'),y_pred.rename('1-Month Forecast'),t.rename('Date')],axis = 1)
data2 = data.melt(id_vars='Date', var_name='Returns',  value_name='Monthly Returns')

plt.figure()
sns.lineplot(x='Date',y='Monthly Returns', hue ='Returns', data = data2, palette="deep6" )
plt.savefig('out/'+ Models_Folder +'/models/'+config['name']+'_returns.jpg')
#%% #--------------------------------------------------
#* Strategy Performace


def calculate_cumulative_return(rm_rf):
    rm_rf_1 = (rm_rf+1)
    ln_rm_rf_1 = rm_rf_1.apply(lambda x: math.log(x))
    cumr=ln_rm_rf_1.cumsum(axis = 0).apply(lambda x: math.exp(x))
    return cumr

cumr = calculate_cumulative_return(rm_rf).rename('sp500')
#%% #--------------------------------------------------

#** Constuct Strategy Postions
leverage = y_pred>0
Leverage = pd.Series(leverage).rename('Leverage')
Leverage_Data  =pd.concat([Leverage,t.rename('Date')],axis = 1)
plt.figure()
sns.lineplot(x='Date',y='Leverage',data=Leverage_Data)

plt.savefig('out/'+ Models_Folder +'/models/'+config['name']+'_leverage.jpg')

#%% #--------------------------------------------------
#** Define Strategy based on ENET
strategy_return = leverage*rm_rf
strategy_cumreturn = calculate_cumulative_return(strategy_return).rename(config['name'])
#%% #--------------------------------------------------
#** Plot Strategy Performance
data=pd.concat([cumr.rename('S&P 500'),strategy_cumreturn.rename('Elastic Net'),t.rename('Date')],axis = 1)
data = data.melt(id_vars='Date', var_name='Strategy',  value_name='Value of 1$')

plt.figure()
sns.lineplot(x='Date',y='Value of 1$', hue ='Strategy', data = data, palette="deep6" )
plt.savefig('out/'+ Models_Folder +'/models/'+config['name']+'_cumulative.jpg')
#%% #--------------------------------------------------


