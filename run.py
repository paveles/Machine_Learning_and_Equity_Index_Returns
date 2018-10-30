#%matplotlib inline 
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]
import math
import time
import datetime
import pandas as pd
import numpy as np
import matplotlib as mpl
from sklearn import linear_model
mpl.rcParams['patch.force_edgecolor'] = True
# %matplotlib inline

import seaborn as sns
#plt.style.use('ggplot')
sns.set()
#sns.set_style("whitegrid")



from pandas.tseries.offsets import MonthEnd # To Determine the End of the Corresponding Month
#pd.set_option('display.max_rows', 500)
#pd.set_option('display.max_columns', 100)

import sys # To caclulate memory usaage

import os
# dir = 'C:\Research\Google Search Volume'
dir = 'E:\Research\Equity Premium and Machine Learning'
dir = "C:/Users/vonNe/Google Drive/Data Science/Projects/Equity Premium and Machine Learning"
#dir = 'D:/Ravenpack'
os.chdir(dir)
os.makedirs(dir + '/temp', exist_ok = True)
os.makedirs(dir + '/out/temp', exist_ok = True)
os.makedirs(dir + '/in', exist_ok = True)
#%%

df = pd.read_csv('in/rapach_2013.csv', na_values = ['NaN'])
df.rename( index=str, columns={"date": "ym"}, inplace=True)
df['date'] = pd.to_datetime(df['ym'],format='%Y%m') + MonthEnd(1)
df['sp500_rf'] = df['sp500_rf'] * 100
df['lnsp500_rf'] = df['lnsp500_rf'] * 100


#%%
"""Lagging predictive  variables"""

df['recessionD_c'] = df['recessionD']
vars = ['recessionD', 'dp', 'dy', 'ep', 'de', \
       'rvol', 'bm', 'ntis', 'tbl', 'lty', 'ltr', 'tms', 'dfy', 'dfr', 'infl', \
       'ma_1_9', 'ma_1_12', 'ma_2_9', 'ma_2_12', 'ma_3_9', 'ma_3_12', 'mom_9', \
       'mom_12', 'vol_1_9', 'vol_1_12', 'vol_2_9', 'vol_2_12', 'vol_3_9', \
       'vol_3_12', 'sento ', 'sent', 'dsento', 'dsent', 'ewsi']
df[vars] = df[vars].shift(1)

#%%
"""Sample Cut"""

df_full = df
df = df[(df['date'].dt.year >= 1951)]

#%%
"""Provide a Description of the Data"""
df.describe().T.to_csv("out/temp/descriptive.csv")
"""
--> Data is the same is in the paper Rapach et al 2013
"""


#%%
"""
Define variables
"""
state = ['recessionD', 'sent']
macro = [ 'dp', 'dy', 'ep', 'de', 'rvol', 'bm', 'ntis', 'tbl', 'lty', 'ltr', 'tms', 'dfy', 'dfr', 'infl'] 
tech = ['ma_1_9', 'ma_1_12', 'ma_2_9', 'ma_2_12', 'ma_3_9', 'ma_3_12', 'mom_9', \
       'mom_12', 'vol_1_9', 'vol_1_12', 'vol_2_9', 'vol_2_12', 'vol_3_9', \
       'vol_3_12']
predictors = macro + tech
#%%
'''Standardize Data'''
X= df[predictors]
y = df['lnsp500_rf']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X2 = X
X[X.columns]= scaler.fit_transform(X[X.columns])

#%% 
''' Lasso model selection: Cross-Validation / AIC / BIC'''

"""
X /= np.sqrt(np.sum(X ** 2, axis=0))
x = pd.DataFrame([2,3,2,1,4,5])
x /= np.sqrt(np.sum(x ** 2, axis=0))
"""
'''
from sklearn.model_selection import KFold
kf = KFold(n_splits=10)
for train, test in kf.split(X):
    print("%s %s" % (train, test))
'''

# LassoCV: coordinate descent
# Compute paths

from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC, ElasticNetCV
# #############################################################################
# LassoCV: coordinate descent

print("Computing regularization path using the coordinate descent lasso...")
t1 = time.time()
model = LassoCV(cv=10).fit(X, y)
model_lasso = model
t_lasso_cv = time.time() - t1
alpha_lasso = -np.log10(model.alpha_)


# Display results
m_log_alphas = -np.log10(model.alphas_)

plt.figure()

plt.plot(m_log_alphas, model.mse_path_, ':')
plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha = %f: CV estimate' % alpha_lasso)

plt.legend()

plt.xlabel('-log(alpha)')
plt.ylabel('Mean square error')
plt.title('Mean square error on each fold: coordinate descent '
          '(train time: %.2fs)' % t_lasso_cv)
plt.axis('tight')
print(alpha_lasso)
#ymin, ymax = 2300, 3800
#plt.ylim(ymin, ymax)
#%%
# #############################################################################
# .ElasticNetCV: coordinate descent

print("Computing regularization path using the coordinate descent lasso...")
t1 = time.time()
model = ElasticNetCV(cv=10).fit(X, y)
model_enet = model
t_enet_cv = time.time() - t1

# Display results
m_log_alphas = -np.log10(model.alphas_)

alpha_enet = -np.log10(model.alpha_)


plt.figure()

plt.plot(m_log_alphas, model.mse_path_, ':')
plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha =%f : CV estimate' % alpha_enet ) 

plt.legend()

plt.xlabel('-log(alpha)')
plt.ylabel('Mean square error')
plt.title('Mean square error on each fold: coordinate descent '
          '(train time: %.2fs)' % t_enet_cv)
plt.axis('tight')
print('\n')
print(alpha_enet)


'''
# #############################################################################
# LassoLarsCV: least angle regression
print("Computing regularization path using the Lars lasso...")
t1 = time.time()
model = LassoLarsCV(cv=20).fit(X, y)
t_lasso_lars_cv = time.time() - t1

# Display results
m_log_alphas = -np.log10(model.cv_alphas_)

plt.figure()
plt.plot(m_log_alphas, model.mse_path_, ':')
plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.legend()

plt.xlabel('-log(alpha)')
plt.ylabel('Mean square error')
plt.title('Mean square error on each fold: Lars (train time: %.2fs)'
          % t_lasso_lars_cv)
plt.axis('tight')
#plt.ylim(ymin, ymax)



print(-np.log10(model.alpha_))

# #############################################################################
# LassoLarsIC: least angle regression with BIC/AIC criterion

model_bic = LassoLarsIC(criterion='bic')
t1 = time.time()
model_bic.fit(X, y)
t_bic = time.time() - t1
alpha_bic_ = model_bic.alpha_

model_aic = LassoLarsIC(criterion='aic')
model_aic.fit(X, y)
alpha_aic_ = model_aic.alpha_

plt.show()

def plot_ic_criterion(model, name, color):
    alpha_ = model.alpha_
    alphas_ = model.alphas_
    criterion_ = model.criterion_
    plt.plot(-np.log10(alphas_), criterion_, '--', color=color,
             linewidth=3, label='%s criterion' % name)
    plt.axvline(-np.log10(alpha_), color=color, linewidth=3,
                label='alpha: %s estimate' % name)
    plt.xlabel('-log(alpha)')
    plt.ylabel('criterion')

plt.figure()
plot_ic_criterion(model_aic, 'AIC', 'b')
plot_ic_criterion(model_bic, 'BIC', 'r')
plt.legend()
plt.title('Information-criterion for model selection (training time %.3fs)'
          % t_bic)
'''
#%%
''' Lasso and Elastic Net - Paths'''

from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import lasso_path, enet_path
from sklearn import datasets

# Compute paths

eps = 5e-2  # the smaller it is the longer is the path

print("Computing regularization path using the lasso...")
alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps, fit_intercept=False)


print("Computing regularization path using the elastic net...")
alphas_enet, coefs_enet, _ = enet_path(
    X, y, eps=eps, l1_ratio=0.8, fit_intercept=False)



# Display results

plt.figure()
colors = cycle(['b', 'r', 'g', 'c', 'k','m','y'])
neg_log_alphas_lasso = -np.log10(alphas_lasso)
neg_log_alphas_enet = -np.log10(alphas_enet)
for coef_l, coef_e, c in zip(coefs_lasso, coefs_enet, colors):
    l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)
    l2 = plt.plot(neg_log_alphas_enet, coef_e, linestyle='--', c=c)

plt.xlabel('-Log(alpha)')
plt.ylabel('coefficients')
plt.title('Lasso and Elastic-Net Paths')
plt.legend((l1[-1], l2[-1]), ('Lasso', 'Elastic-Net'), loc='lower left')
plt.axis('tight')
plt.savefig(dir+"/out/lasso_enet_paths")
#%%
''' Lasso Path - Labels'''
sns.set_palette("husl", 28)
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import lasso_path, enet_path
from sklearn import datasets

# Compute paths

eps = 5e-2  # the smaller it is the longer is the path

print("Computing regularization path using the lasso...")
alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps, fit_intercept=False)



# Display results
labels = X.columns
plt.figure()
#colors = cycle(['b', 'r', 'g', 'c', 'k','m','y'])
neg_log_alphas_lasso = -np.log10(alphas_lasso)
K = coefs_lasso.shape[0]
for k in range(K):
    l1 = plt.plot(neg_log_alphas_lasso, coefs_lasso[k], label = labels[k])

plt.axvline(x=alpha_lasso, color='k', linestyle='--')
plt.xlabel('-Log(alpha)')
plt.ylabel('Coefficients')
plt.title('Lasso Path')
plt.legend()
plt.axis('tight')
plt.savefig(dir+"/out/lassopath")
#%%
#A helper method for pretty-printing linear models
def pretty_print_linear(coefs, names = None, sort = False):
    if np.sum(names == None):
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst, key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name) for coef, name in lst)

print("Linear model:", pretty_print_linear(model_lasso.coef_[model_lasso.coef_>0], names =  X.columns[model_lasso.coef_>0] ))

#print("Linear model:", pretty_print_linear(model_enet.coef_, names =  X.columns ))