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
scaler = StandardScaler().fit(X)
X = pd.DataFrame(scaler.transform(X),  index=X.index, columns=X.columns )
X_test = pd.DataFrame(scaler.transform(X_test),  index=X_test.index, columns=X_test.columns )
#%% #--------------------------------------------------
#''' Interaction Terms'''
from sklearn.preprocessing import PolynomialFeatures
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
#%% #--------------------------------------------------
#* Ones for Constant Model
Ones = pd.DataFrame(np.ones(y.shape[0]))
Ones_test = pd.DataFrame(np.ones(y_test.shape[0]))
#%% #--------------------------------------------------
#* Prepare data for the PCA
from sklearn import linear_model
from sklearn.decomposition import PCA
#pca = PCA().fit(X)

# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance');

pca = PCA(n_components=4)
pca.fit(X)
X_pca = pca.transform(X)
X_test_pca = pca.transform(X_test)
# #############################################################################
#%% #-------------------------------------------------- 
from sklearn import linear_model
#''' OLS model'''
reg = linear_model.LinearRegression()
model_ols = reg.fit(X,y)

#'''Constant model'''
reg = linear_model.LinearRegression(fit_intercept = False)
model_c = reg.fit(Ones,y)
#''' PCA '''
reg = linear_model.LinearRegression()
model_pca = reg.fit(X_pca,y)


#%% #--------------------------------------------------
#''' Lasso model selection: Cross-Validation'''
# LassoCV: coordinate descent
# Compute paths

from sklearn.linear_model import  RidgeCV, LassoCV, ElasticNetCV

# LassoCV: coordinate descent
model_lasso = LassoCV(cv=K).fit(X, y)
lambda_lasso = model_lasso.alpha_

#Ridge
ridge_alphas = np.logspace(-4, 2, 50)
model_ridge = ElasticNetCV(alphas = ridge_alphas, l1_ratio = 0, cv=K).fit(X, y)
lambda_ridge = model_ridge.alpha_

# .ElasticNetCV: coordinate descent
model_enet = ElasticNetCV(cv=K).fit(X, y)
lambda_enet = model_enet.alpha_

#%% #--------------------------------------------------
#? lightgbm
import lightgbm as lgb
from sklearn.metrics import  make_scorer, mean_squared_error, r2_score
lgtrain = lgb.Dataset(X,y ,feature_name = "auto")
lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    "learning_rate": 0.01,
    "num_leaves": 180,
    "feature_fraction": 0.50,
    "bagging_fraction": 0.50,
    'bagging_freq': 4,
    "max_depth": -1,
    "reg_alpha": 0.3,
    "reg_lambda": 0.1,
    #"min_split_gain":0.2,
    "min_child_weight":10,
    'zero_as_missing': False
                }
# Find Optimal Parameters / Boosting Rounds
lgb_cv = lgb.cv(
    params = lgbm_params,
    train_set = lgtrain,
    num_boost_round=2000,
    stratified=False,
    nfold = 10,
    verbose_eval=50,
    seed = 23,
    early_stopping_rounds=75)
    
optimal_rounds = np.argmin(lgb_cv['l2-mean'])
best_cv_score = min(lgb_cv['l2-mean'])

print("\nOptimal Round: {}\nOptimal Score: {} + {}".format(
    optimal_rounds,best_cv_score,lgb_cv['l2-stdv'][optimal_rounds]))
results = pd.DataFrame(columns = ["Rounds","Score","STDV", "LB", "Parameters"])
results = results.append({"Rounds": optimal_rounds,
                          "Score": best_cv_score,
                          "STDV": lgb_cv['l2-stdv'][optimal_rounds],
                          "LB": None,
                          "Parameters": lgbm_params}, ignore_index=True)
print(results)

final_model_params = results.iloc[results["Score"].idxmin(),:]["Parameters"]
optimal_rounds = results.iloc[results["Score"].idxmin(),:]["Rounds"]
print("Parameters for Final Models:\n",final_model_params)
print("Score: {} +/- {}".format(results.iloc[results["Score"].idxmin(),:]["Score"],results.iloc[results["Score"].idxmin(),:]["STDV"]))
print("Rounds: ", optimal_rounds)

model_lgb = lgb.train(
    final_model_params,
    lgtrain,
    num_boost_round = optimal_rounds + 1,
    verbose_eval=200)

#%% #--------------------------------------------------
#? XGBoost
import xgboost as xgb
from xgboost import XGBRegressor
import scipy.stats as stats
from sklearn.model_selection import RandomizedSearchCV
from sklearn import model_selection
from sklearn.metrics import  make_scorer, mean_squared_error, r2_score

# model = XGBRegressor()
# model_xgb = model.fit(X,y)



clf_xgb = xgb.XGBRegressor()
param_dist = {'n_estimators': stats.randint(150, 500),
              'learning_rate': stats.uniform(0.01, 0.07),
              'subsample': stats.uniform(0.3, 0.7),
              'max_depth': [3, 4, 5, 6, 7, 8, 9],
              'colsample_bytree': stats.uniform(0.5, 0.45),
              'min_child_weight': [1, 2, 3]
            }
model_xgb = RandomizedSearchCV(clf_xgb, param_distributions = param_dist, n_iter = 25,
                         scoring = 'neg_mean_squared_error', error_score = 0, verbose = 3, n_jobs = -1, cv = 5).fit(X,y)

# numFolds = 5
# folds = model_selection.KFold(shuffle = False, n_splits = numFolds)

# estimators = []
# results = np.zeros(len(X))
# score = 0.0
# for train_index, test_index in folds.split(X.index):
#     print(test_index)
#     print(train_index)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     clf.fit(X_train, y_train)

#     estimators.append(clf.best_estimator_)
#     results[test_index] = clf.predict(X_test)
#     score += mean_squared_error(y_test, results[test_index])
# score /= numFolds
#%% #--------------------------------------------------
#? Random Forest
from sklearn.ensemble import RandomForestRegressor
model_rf= RandomForestRegressor(random_state=2).fit(X,y)
#%% #--------------------------------------------------
#? 
from sklearn.ensemble import AdaBoostRegressor
model_adab= AdaBoostRegressor(random_state=2).fit(X,y)
#%% #--------------------------------------------------
#? 
from sklearn.ensemble import GradientBoostingRegressor
model_gbr= GradientBoostingRegressor(random_state=2).fit(X,y)
#%% #--------------------------------------------------

#? Gradient Boosting 
#? Keras - FFN
#? Keras - LSTM
#? AutoML
#? AutoSklearn
#? TPOT
#? Autofeat
#? Talos

#%% #--------------------------------------------------
#? TPOT
# #! Takes around 3 min
# from tpot import TPOTRegressor
# from sklearn.metrics import  make_scorer, mean_squared_error, r2_score
# mse_scorer = make_scorer(mean_squared_error)
# tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2,
#                      random_state = 1, template = 'Regressor') #, scoring=mse_scorer
# model_tpot = tpot.fit(X, y).fitted_pipeline_
#     # tpot.export('out/tpot_pipeline.py')
#%% #--------------------------------------------------
#? Used Models
models ={
    'c' : model_c,
    'ols' : model_ols,
    'pca' : model_pca,
    'ridge' : model_ridge,
    'lasso' : model_lasso,
    'enet' : model_enet,
    'adab' : model_adab,
    'rf': model_rf,
    'gbr':model_gbr,
    'lgb' : model_lgb,
    'xgb': model_xgb,
#    'tpot': model_tpot,
}


#%% #--------------------------------------------------
# #* Pickle Models
import pickle

with open("models.pickle","wb") as f:
    pickle.dump(models, f)

#%% #--------------------------------------------------
#* Load Pickled Models
import pickle
if 'models' not in globals():
    with open("models.pickle", "rb") as f:
        models = pickle.load(f)

#%% #--------------------------------------------------
#'''Train-Validation-Test Prepare '''


print("Train-Validation-Test Performance")
from sklearn.model_selection import cross_validate
### Define Scorer
from sklearn.metrics import  make_scorer, mean_squared_error, r2_score

mean_squared_error_scorer = make_scorer(mean_squared_error)
scoring = {'MSE': mean_squared_error_scorer, 'r2': make_scorer(r2_score)}
#%% #--------------------------------------------------
### Cross-Validation + Test Sample

yhats = pd.DataFrame()
df_results = pd.DataFrame()
for m_name, m in models.items():
    if m_name == 'c':
        XX = Ones
        XX_test = Ones_test
    elif m_name == 'ols':
        XX = X
        XX_test = X_test
    elif m_name == 'pca':
       XX = X_pca 
       XX_test = X_test_pca
    else:
        XX = X
        XX_test = X_test
    # Cross-Validation 
    yhat_train = m.predict(XX)
    df_avg = pd.Series()
    train_r2 = r2_score(y, yhat_train)
    train_MSE = mean_squared_error(y, yhat_train)
    df_avg = df_avg.append(pd.Series(train_r2)).rename({0 : "train_r2"}, axis = 'index')
    df_avg = df_avg.append(pd.Series(train_MSE)).rename({0 : "train_MSE"}, axis = 'index')
    
    # cv_results = cross_validate(m, XX, y, cv  = K , return_train_score=True, scoring = scoring )
    # df_cv = pd.DataFrame.from_dict(cv_results)
    # df_avg = df_cv.drop(columns=['fit_time', 'score_time']).mean(axis = 0)
    # df_avg = df_avg.rename(index={ "test_MSE" : "valid_MSE", "test_r2" : "valid_r2"})

    
    # Test Sample
    yhat = m.predict(XX_test)
    yhats[m_name] = yhat
    test_r2 = r2_score(y_test, yhat)
    test_MSE = mean_squared_error(y_test, yhat)
    df_avg = df_avg.append(pd.Series(test_r2)).rename({0 : "test_r2"}, axis = 'index')
    df_avg = df_avg.append(pd.Series(test_MSE)).rename({0 : "test_MSE"}, axis = 'index')
    
    df_name = pd.Series(m_name) 
    df_avg = df_avg.append(df_name).rename({0 : "model"}, axis = 'index')
    
    df_results = df_results.append(df_avg, ignore_index=True)
    

print(df_results)
df_results.to_csv("out/insample_results.csv")
'''
--> ENET and LASSO  perform better our-of-sample but R2 negative
--> OLS performs the best in-sample
'''
#%% #--------------------------------------------------
#'''Prediction Plot''' 
sns.set_palette("deep")
plt.figure(figsize=(15,7.5))
ym_test= pd.DataFrame(df['date'].loc[y_test.index]).reset_index(drop=True)
y_test_fig = y_test.reset_index(drop = True)
plotdata = pd.concat([ym_test,y_test_fig,yhats],axis = 1)
plotdata = plotdata.melt(id_vars='date', var_name='model',  value_name='return')
sns.lineplot(x='date',y='return', hue ='model', data = plotdata )
plt.savefig(dir+"/out/lineplot_predict")
plt.show()

#%% #--------------------------------------------------

# sns.reset_defaults()
# g = sns.FacetGrid(plotdata, hue='model', height = 5, aspect= 2 )
# g = g.map(plt.bar, 'date','return' , alpha=0.7)
# g = g.add_legend()
# plt.show()
# plt.savefig(dir+"/out/barplot_predict")

#%% #--------------------------------------------------
#* Define Function to Draw Cross-Validation for Optimal Lambda
# def display_optimal_alpha(model,model_name):
#     # Display results
#     m_log_alphas = -np.log10(model.alphas_)


#     plt.figure()

#     plt.plot(model.alphas_, model.mse_path_, ':')
#     plt.plot(model.alphas_, model.mse_path_.mean(axis=-1), 'k',
#             label='Average across the folds', linewidth=2)
#     plt.axvline(model.alpha_, linestyle='--', color='k',
#                 label='alpha = %f: CV estimate' % model.alpha_)

#     plt.legend()

#     plt.xlabel('$\lambda$')
#     plt.ylabel('Mean square error')
#     plt.title(model_name + ' - Mean square error on each fold: coordinate descent ')
#     plt.axis('tight')
#     print(-np.log10(model.alpha_))
#     #ymin, ymax = 2300, 3800
#     #plt.ylim(ymin, ymax)
#     plt.show()
#     return plt


# fig = display_optimal_alpha(model_lasso, 'lasso')
# plt.savefig(dir+"/out/lasso_cv")
# fig = display_optimal_alpha(model_ridge, 'ridge')
# plt.savefig(dir+"/out/ridge_cv")
# fig = display_optimal_alpha(model_enet, 'enet')
# plt.savefig(dir+"/out/enet_cv")


# #%% #--------------------------------------------------
# #''' Lasso and Enet Path - Labels'''

# from itertools import cycle
# import numpy as np
# import matplotlib.pyplot as plt

# from sklearn.linear_model import lasso_path, enet_path
# from sklearn import datasets

# def lambda_path(model_name, lmbda):

#     if model_name == "enet": 
#         l1 = 0.5
#     elif model_name == "lasso":  
#         l1 = 1
#     elif model_name == "ridge":  
#         l1 = 0
#     # Compute paths
#     eps = 5e-2  # the smaller it is the longer is the path

#     print("Computing regularization path using the lasso...")
#     alphas, coefs, _ =  enet_path(X, y, eps=eps, l1_ratio=l1, fit_intercept=True)

#     # Display results
#     labels = X.columns
#     plt.figure()
#     #colors = cycle(['b', 'r', 'g', 'c', 'k','m','y'])
#     log_alphas = np.log10(alphas)
#     C = coefs.shape[0]
#     for k in range(C):
#         l1 = plt.plot(alphas, coefs[k], label = labels[k],)

#     plt.axvline(x=lmbda, color='k', linestyle='--')
#     plt.xlabel('$\lambda$')
#     plt.ylabel('Coefficients')
#     plt.title('Lasso Path')
#     plt.legend()
#     plt.axis('tight')
#     plt.show()
#     return plt

# sns.set_palette("husl", 28)
# fig = lambda_path("lasso", lambda_lasso)
# fig.savefig(dir+"/out/lasso_path")
# fig = lambda_path("enet", lambda_enet)
# fig.savefig(dir+"/out/enet_path")
# sns.set_palette("deep")


# #%% #--------------------------------------------------
# #'''Coefficients Plot''' 
# plt.figure()
# #labels.insert(0,'cons')
# if Poly ==0:
#     labels = list(X.columns)

#     plt.plot(np.arange(len(labels)),model_ridge.coef_, color='g', linewidth=2,
#              label='Ridge coefficients', alpha = 0.6)
#     plt.plot(np.arange(len(labels)), model_lasso.coef_, color='r', linewidth=2,
#              label='Lasso coefficients', alpha = 0.6)
#     plt.plot(np.arange(len(labels)), model_enet.coef_, color='b', linewidth=2,
#              label='Elastic net coefficients', alpha = 0.6)
#     plt.xticks(range(len(labels)), labels, rotation=45)
#     #ax[1].set_xticklabels(labels)

# else:
#  #   plt.plot(model_ols.coef_, '--', color='navy', label='OLS coefficients')
#     plt.plot(model_ridge.coef_, color='g', linewidth=2,
#              label='Ridge coefficients', alpha = 0.6)
#     plt.plot(model_lasso.coef_, color='r', linewidth=2,
#              label='Lasso coefficients', alpha = 0.6)
#     plt.plot(model_enet.coef_, color='b', linewidth=2,
#              label='Elastic net coefficients', alpha = 0.6)
    
# plt.axhline(y=0,linestyle = '--', color='k')
# plt.legend(loc='best')
# plt.title("Ridge, Lasso, Elastic Net Coefficients")
# plt.xlabel('Variables')
# plt.ylabel('Coefficients')
# #plt.show()
# plt.savefig(dir+"/out/Coefficients")

#%% #--------------------------------------------------
# Multicollinearity

# corr=np.corrcoef(X,rowvar=0)
# corr
# W,V=np.linalg.eig(corr)
# print(W)
# list(X)
# Xcorr = X.corr()
# plt.figure(figsize=(30,15))
# sns.heatmap(Xcorr, annot=True)
# plt.savefig(dir+"/out/Coefficients")
#--> Multicollinearity in data
#%% #--------------------------------------------------

# # Performance Metrics - Out_of-Sample Comparison
# print(" Performance Metrics - Out_of-Sample Comparison")
# from sklearn.metrics import mean_squared_error, r2_score
# def r2_adj_score(y, yhat, n, p):
#     r2 =  r2_score(y, yhat)
#     return 1 - (1-r2)*(n-1)/(n-p-1)

# yhat_c = model_c.predict(Ones_test)
# yhat_ols = model_ols.predict(X_test)
# yhat_pca = model_pca.predict(X_test_pca)
# yhat_ridge = model_ridge.predict(X_test)
# yhat_lasso = model_lasso.predict(X_test)
# yhat_enet = model_enet.predict(X_test)

# yhats_old = [yhat_c,  yhat_pca, yhat_ols,  yhat_ridge, yhat_lasso, yhat_enet]
# yhats_names = ['c_model','pca_model', 'ols_model','ridge_model', 'lasso_model', 'enet_model']
# print("MSE:")
# for yh,yh_n in zip(yhats_old,yhats_names):
#     print("mean_squared_error(y_test, {})".format(yh_n))
#     print(mean_squared_error(y_test, yh))
#     print("r2_score(y_test, {})".format(yh_n))
#     print(r2_score(y_test, yh))


# ## Print the Results
# os.makedirs(dir+"out/oos/",exist_ok = True)
# f = open(dir+"/out/oos/K{}_TsizeInv{}_Poly{}_Period{}.txt".format(K,TsizeInv,Poly,Period), 'w')
# f.write("MSE:")
# for yh,yh_n in zip(yhats_old,yhats_names):
#     f.write("mean_squared_error(y_test, {})\n".format(yh_n))
#     f.write("{}\n".format(mean_squared_error(y_test, yh)))
# f.close()

#%% #--------------------------------------------------
# #'''Prediction Plot''' 
# plt.figure(figsize=(15,7.5))
# x= np.array(df['ym'].loc[y_test.index]).astype(int)
# #plt.plot(model_ols.coef_, '--', color='navy', label='OLS coefficients')
# colors = cycle(['c','m','y', 'g', 'r','b', 'k'])
# yhats_old = [yhat_c,  yhat_pca, yhat_ols,  yhat_ridge, yhat_lasso, yhat_enet]
# yhats_names = ['Const', 'PCA',	'OLS',	'Ridge', 'Lasso', 'Enet']
# plt.plot(np.arange(len(x)),y_test,'--', color='k', linewidth=1,
#              label='Realized Return')
# for yh,yh_n,color in zip(yhats_old,yhats_names,colors):
#     plt.plot(np.arange(len(x)),yh, linewidth=2, color = color)
# #label="{} prediction".format(yh_n)
# plt.xticks(range(len(x)), x, rotation=45)
# yhats_names.insert(0,'Realized Return')
# plt.xlabel('Date')
# plt.ylabel('Monthly Return in %')
# plt.legend(yhats_names)
# plt.savefig(dir+"/out/Prediction")

