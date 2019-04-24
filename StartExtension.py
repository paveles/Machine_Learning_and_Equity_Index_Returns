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

#? Data Prepare
def data_prepare(df):
    # Important! Lagging by 1
    df['recessionD_c'] = df['recessionD']
    vars = ['recessionD', 'dp', 'dy', 'ep', 'de', \
            'rvol', 'bm', 'ntis', 'tbl', 'lty', 'ltr', 'tms', 'dfy', 'dfr', 'infl', \
            'ma_1_9', 'ma_1_12', 'ma_2_9', 'ma_2_12', 'ma_3_9', 'ma_3_12', 'mom_9', \
            'mom_12', 'vol_1_9', 'vol_1_12', 'vol_2_9', 'vol_2_12', 'vol_3_9', \
            'vol_3_12', 'sento ', 'sent', 'dsento', 'dsent', 'ewsi']
    df[vars] = df[vars].shift(1) #!  Shifting
    # Define variables
    other = ['ewsi']
    state = ['recessionD', 'sent']
    macro = [ 'dp', 'dy', 'ep', 'de', 'rvol', 'bm', 'ntis', 'tbl', 'lty', 'ltr', 'tms', 'dfy', 'dfr', 'infl'] 
    tech = ['ma_1_9', 'ma_1_12', 'ma_2_9', 'ma_2_12', 'ma_3_9', 'ma_3_12', 'mom_9', \
            'mom_12', 'vol_1_9', 'vol_1_12', 'vol_2_9', 'vol_2_12', 'vol_3_9', \
            'vol_3_12']
    return df, macro, tech, state, other
#?''' Train and Test Samples'''
from sklearn.model_selection import train_test_split
def train_test(df):
    Xo= df[predictors]
    yo = df['lnsp500_rf']
    X, X_test, y, y_test = train_test_split(Xo, yo, test_size=test_size, shuffle = False )
    return X, X_test, y, y_test


#? Cut Sample Period
def sample_cut(df, period = None):
# """
# Sample Cut
# """
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

from sklearn.preprocessing import PolynomialFeatures
#?''' Interaction Terms'''
def generate_interactions(X, X_test):
    if Poly == 1:
        poly = PolynomialFeatures(interaction_only=True,include_bias = False)
        Xp = poly.fit_transform(X)
        Xp_test = poly.fit_transform(X_test)
    elif Poly == 2:
        poly = PolynomialFeatures(degree = 2,include_bias = False)
        Xp = poly.fit_transform(X)
        Xp_test = poly.fit_transform(X_test)
    else:
        Xp = X
        Xp_test = X_test
    return Xp, Xp_test
#? '''Standardize Data'''
from sklearn.preprocessing import StandardScaler, MinMaxScaler
def data_scale(X, X_test):
    scaler = StandardScaler().fit(X)
    X = pd.DataFrame(scaler.transform(X),  index=X.index, columns=X.columns )
    X_test = pd.DataFrame(scaler.transform(X_test),  index=X_test.index, columns=X_test.columns )
    return X, X_test
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

from tsml_framework import * # Framework for Time Series Analysis
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
#! Define Models 
#?''' OLS model'''
from sklearn import linear_model
ols = linear_model.LinearRegression()

#? Constant Need to change! as  Scikit- Learn Model
#c = linear_model.LinearRegression(fit_intercept = False)

#? PCA model - Also as a scikit-learn model + Optimal number of Comp
from sklearn.decomposition import PCA
# reg = linear_model.LinearRegression()
# pca = PCA(n_components=4)
# pca.fit(X)
# X_pca = pca.transform(X)
# model_pca = reg.fit(X_pca,y)
# X_test_pca = pca.transform(X_test)

#? LassoCV
from sklearn.linear_model import  LassoCV
lasso = LassoCV(cv=K)

#? RidgeCV
from sklearn.linear_model import ElasticNetCV
ridge = ElasticNetCV( l1_ratio = 0, cv=K)

#? ElasticNetCV
from sklearn.linear_model import ElasticNetCV
enet = ElasticNetCV(cv=K)

#? XGBoost
from xgboost import XGBRegressor
#? lightgbm
from lightgbm import LGBMRegressor
#? Random Forest
from sklearn.ensemble import RandomForestRegressor

#? Keras - FFN
#? Keras - LSTM
#? AutoML
#? AutoSklearn
#? TPOT
#? Autofeat
#? Talos

models = {"ols": ols}
#%% #--------------------------------------------------
#! Analysis Sequence
#?Prepare
df = data_read('in/rapach_2013.csv')
df, macro, tech, state, other = data_prepare(df)
df, predictors = sample_cut(df, period = Period)

#? Describe
#df.describe().T 
#""" --> Data is the same is in the paper Rapach et al 2013"""
#%% #--------------------------------------------------
#? Split and Prepare
X, X_test, y, y_test = train_test(df)
X, X_test = data_scale(X,X_test)
Xp, Xp_test = generate_interactions(X, X_test)
X = X[macro]
dft = pd.concat([X, y, df['date'].dt.to_period('M').astype(int)], join = 'inner', axis=1)
dft.columns
#%% #--------------------------------------------------
#? Simplest Experiment
super_1_p = ols.fit(X,y)
#X_1 = X.shift(1).dropna(axis = 0, how ='all')
Model_1_Error = super_1_p.score(X,y)
print(super_1_p,Model_1_Error)
#%% #--------------------------------------------------
#? One Step Further
from tsml_framework import * # Framework for Time Series Analysis
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
#import numpy as np
model = ols
kf = Kfold_time(target='lnsp500_rf',date_col = 'date', 
                date_init=450, date_final=dft['date'].max())
                #108
steps_1 = [('1_step', ToSupervised(X,y,0,dropna = True)), 
('predic_1', TimeSeriesRegressor(model=model,cv=kf, scoring = mean_squared_error))]

super_1_p = Pipeline(steps_1).fit(dft)
#Model_1_Error = super_1_p.score(dft)
#%% #--------------------------------------------------
Model_1_Error


#%%
