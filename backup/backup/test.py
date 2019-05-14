#%% [markdown] #--------------------------------------------------
## Equity Premium and Machine Learning
#%% #--------------------------------------------------

import warnings
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]
import math
import time
import datetime
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.rcParams['patch.force_edgecolor'] = True
import seaborn as sns
sns.set()
from pandas.tseries.offsets import MonthEnd # To Determine the End of the Corresponding Month
import sys # To caclulate memory usage
import os

dir = os.getcwd()
os.chdir(dir)
os.makedirs(dir + '/temp', exist_ok = True)
os.makedirs(dir + '/out/temp', exist_ok = True)
os.makedirs(dir + '/in', exist_ok = True)

# Cross-validation Parameter
K = 10
#  Share of Sample as Test
TsizeInv = 15
test_size= 1/TsizeInv
# Add interactions or not
Poly = 1
# Starting Year
Period  = 1951

#%% #--------------------------------------------------

df = pd.read_csv('in/rapach_2013.csv', na_values = ['NaN'])
df.rename( index=str, columns={"date": "ym"}, inplace=True)
df['date'] = pd.to_datetime(df['ym'],format='%Y%m') + MonthEnd(1)
df['sp500_rf'] = df['sp500_rf'] * 100
df['lnsp500_rf'] = df['lnsp500_rf'] * 100
df = df.sort_values(by=['ym'])

#%% #--------------------------------------------------
#"""Lagging predictive  variables"""

df['recessionD_c'] = df['recessionD']
vars = ['recessionD', 'dp', 'dy', 'ep', 'de', \
       'rvol', 'bm', 'ntis', 'tbl', 'lty', 'ltr', 'tms', 'dfy', 'dfr', 'infl', \
       'ma_1_9', 'ma_1_12', 'ma_2_9', 'ma_2_12', 'ma_3_9', 'ma_3_12', 'mom_9', \
       'mom_12', 'vol_1_9', 'vol_1_12', 'vol_2_9', 'vol_2_12', 'vol_3_9', \
       'vol_3_12', 'sento ', 'sent', 'dsento', 'dsent', 'ewsi']
# Important! Lagging by 1
df[vars] = df[vars].shift(1)
#%% #--------------------------------------------------
"""
Define variables
"""
other = ['ewsi']
state = ['recessionD', 'sent']
macro = [ 'dp', 'dy', 'ep', 'de', 'rvol', 'bm', 'ntis', 'tbl', 'lty', 'ltr', 'tms', 'dfy', 'dfr', 'infl'] 
tech = ['ma_1_9', 'ma_1_12', 'ma_2_9', 'ma_2_12', 'ma_3_9', 'ma_3_12', 'mom_9', \
       'mom_12', 'vol_1_9', 'vol_1_12', 'vol_2_9', 'vol_2_12', 'vol_3_9', \
       'vol_3_12']
# predictors = macro+ tech # + other  + state
#%% #--------------------------------------------------
"""
Sample Cut
"""

df_full = df
if Period == 1974:
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
# df[pd.isnull(df["ewsi"])!= 1]['date'].describe()
#%% #--------------------------------------------------
#"""Provide a Description of the Data"""
df.describe().T.to_csv("out/temp/descriptive.csv")
#""" --> Data is the same is in the paper Rapach et al 2013"""
# df.describe().T


#%% #--------------------------------------------------
#''' Train and Test Samples'''
from sklearn.model_selection import train_test_split
Xo= df[predictors]
yo = df['lnsp500_rf']

X, X_test, y, y_test = train_test_split(Xo, yo, test_size=test_size, shuffle = False )
#%% #--------------------------------------------------
#'''Standardize Data'''
from sklearn.preprocessing import StandardScaler,MinMaxScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

pipline = Pipeline(steps=[
    ('minmax', StandardScaler()),
    ('pca', PCA(n_components=1)),
])
scaler = pipline.fit(X)
#scaler = StandardScaler().fit(X)

X_train = pd.DataFrame(scaler.transform(X),  index=X.index)


#%% #--------------------------------------------------
#* Prepare data for the PCA


from sklearn.decomposition import PCA
#pca = PCA().fit(X)

# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance');
X_stand = StandardScaler().fit_transform(X)
pca = PCA(n_components=1)
pca.fit(X_stand)
X_pca = pca.transform(X_stand)

# #############################################################################
#%% #--------------------------------------------------
X_diff = X_train - X_pca
X_diff
#%% #--------------------------------------------------
