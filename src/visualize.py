""" 
Predicting Equity Index Returns using Machine Learning Methods - Visualization
"""
#%% #--------------------------------------------------
#*Load Modules
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
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
plt.rcParams['figure.figsize'] = [15, 22.5]
#%% #--------------------------------------------------
#* Global Parameters *
# Add interactions or not
from src.settings import Period, ROLLING, start_idx, Models_Folder,\
VERBOSE, CONFIGS, training_window, validation_window



#%% #--------------------------------------------------
#* Load Data
df0 = pd.read_pickle("data/processed/df.pickle")


#%% #--------------------------------------------------
#* Aggregate Information into one file

df_config = pd.DataFrame()
for cname, config in CONFIGS.items():
    df_config = df_config.append(pd.read_csv('out/'+ Models_Folder +'/models/'+ cname +'.csv'),
     ignore_index =True)
print(df_config)
df_config.to_csv('out/'+ Models_Folder +'/models/'+'All_Models'+'.csv')
print('out/'+ Models_Folder +'/models/'+'All_Models'+'.csv'+' is produced')

#%% #--------------------------------------------------
#* Loop for Summary Graph
for cname, config in CONFIGS.items():
    #%% #------------------------------------------------
    #* Load Strategy
    df_pred = pd.read_csv('out/'+ Models_Folder +'/models/'+ config['name']+'_predictions.csv').set_index('index',drop = True)
    y_pred = df_pred['y_pred']/100
    scores_estimated = df_pred['scores_estimated']

    dff = df0.iloc[y_pred.index]
    t= dff['date']
    rm_rf = dff['sp500_rf']/100
    y_moving_mean = df0['sp500_rf'].shift(1).expanding(1).mean().loc[y_pred.index]
    y_true =rm_rf


    #%% #--------------------------------------------------
    #* Plot one-month ahead foreacst
    data=pd.concat([y_true.rename('Realized'),y_pred.rename('1-Month Forecast'),t.rename('Date')],axis = 1)
    data2 = data.melt(id_vars='Date', var_name='Returns',  value_name='Monthly Returns')

    fig, ax = plt.subplots( ncols=1, nrows=3)

    ax[0].set_title("A. Predicted vs. Realized Return")
    ax[1].set_title("B. Strategy Position")
    ax[2].set_title("C. Strategy Cumulative Returns")
    ax[2].set_xlabel('Date')

    sns.lineplot(x='Date',y='Monthly Returns', hue ='Returns', data = data2, palette="deep6", ax = ax[0] )
    #%% #--------------------------------------------------
    #* Strategy Performace
    #

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

    sns.lineplot(x='Date',y='Leverage',data=Leverage_Data, ax= ax[1])

    #%% #--------------------------------------------------
    #** Define Strategy based on ENET
    strategy_return = leverage*rm_rf
    strategy_cumreturn = calculate_cumulative_return(strategy_return).rename(config['name'])
    #%% #--------------------------------------------------
    #** Plot Strategy Performance
    data=pd.concat([cumr.rename('S&P 500'),strategy_cumreturn.rename('Strategy'),t.rename('Date')],axis = 1)
    data = data.melt(id_vars='Date', var_name='Strategy',  value_name='Value of 1$')

    sns.lineplot(x='Date',y='Value of 1$', hue ='Strategy', data = data, palette="deep6",ax = ax[2] )

    #%% #--------------------------------------------------
    #* Save Figure
    ax[0].set_xlabel('')
    ax[1].set_xlabel('')
    fig.tight_layout()
    plt.savefig('out/'+ Models_Folder +'/models/'+config['name']+'.png')
    print("Figure "+ 'out/'+ Models_Folder +'/models/'+config['name']+'.png'+" is produced")


