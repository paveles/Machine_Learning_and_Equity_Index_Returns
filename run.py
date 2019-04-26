

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
# dir = 'C:\Research\Google Search Volume'
dir = os.getcwd()
#'E:/Research/Equity Premium and Machine Learning'
#dir = "C:/Users/vonNe/Google Drive/Data Science/Projects/Equity Premium and Machine Learning"
#dir = 'D:/Ravenpack'
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
# Period
Period  = 1974
#Test New Line
#%% #--------------------------------------------------

df = pd.read_csv('in/rapach_2013.csv', na_values = ['NaN'])
df.rename( index=str, columns={"date": "ym"}, inplace=True)
df['date'] = pd.to_datetime(df['ym'],format='%Y%m') + MonthEnd(1)
df['sp500_rf'] = df['sp500_rf'] * 100
df['lnsp500_rf'] = df['lnsp500_rf'] * 100
df.sort_values(by=['ym']);

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
# predictors = macro+ tech + other  + state
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
df.describe().T


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
# #############################################################################
#%% #-------------------------------------------------- 
#''' OLS model'''
from sklearn import linear_model
reg = linear_model.LinearRegression()
model_ols = reg.fit(X,y)
#'''Constant model'''
from sklearn import linear_model
reg = linear_model.LinearRegression(fit_intercept = False)
Ones = pd.DataFrame(np.ones(y.shape[0]))
Ones_test = pd.DataFrame(np.ones(y_test.shape[0]))
model_c = reg.fit(Ones,y)
#%% #--------------------------------------------------
#''' PCA'''
from sklearn.decomposition import PCA
#pca = PCA().fit(X)

# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance');

reg = linear_model.LinearRegression()
pca = PCA(n_components=4)
pca.fit(X)
X_pca = pca.transform(X)
model_pca = reg.fit(X_pca,y)
X_test_pca = pca.transform(X_test)
#%% #--------------------------------------------------

#''' Lasso model selection: Cross-Validation'''
# LassoCV: coordinate descent
# Compute paths

from sklearn.linear_model import  RidgeCV, LassoCV, LassoLarsCV, LassoLarsIC, ElasticNetCV

# LassoCV: coordinate descent

print("Computing regularization path using the coordinate descent lasso...")
t1 = time.time()
model = LassoCV(cv=K).fit(X, y)
#print(model.alphas_)
model_lasso = model
t_lasso_cv = time.time() - t1
alpha_lasso = -np.log10(model.alpha_)
lambda_lasso = model.alpha_

# Display results
m_log_alphas = -np.log10(model.alphas_)


plt.figure()

plt.plot(model.alphas_, model.mse_path_, ':')
plt.plot(model.alphas_, model.mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(model.alpha_, linestyle='--', color='k',
            label='alpha = %f: CV estimate' % lambda_lasso)

plt.legend()

plt.xlabel('$\lambda$')
plt.ylabel('Mean square error')
plt.title('LASSO - Mean square error on each fold: coordinate descent ')
plt.axis('tight')
print(alpha_lasso)
#ymin, ymax = 2300, 3800
#plt.ylim(ymin, ymax)

plt.savefig(dir+"/out/lasso_cv")

#%% #--------------------------------------------------
print("Computing regularization path using the coordinate descent ridge...")
t1 = time.time()
ridge_alphas = np.logspace(-4, 2, 50)
model = ElasticNetCV(alphas = ridge_alphas, l1_ratio = 0, cv=K).fit(X, y)

model_ridge = model
t_ridge_cv = time.time() - t1
alpha_ridge = -np.log10(model.alpha_)
lambda_ridge = model.alpha_

# Display results
m_log_alphas = -np.log10(model.alphas_)

plt.figure()

plt.plot(m_log_alphas, model.mse_path_, ':')
plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha = %f: CV estimate' % alpha_ridge)

plt.legend()

plt.xlabel('-log(alpha)')
plt.ylabel('Mean square error')
plt.title('Ridge - Mean square error on each fold: coordinate descent '
          '(train time: %.2fs)' % t_ridge_cv)
plt.axis('tight')
print(alpha_ridge)
#ymin, ymax = 2300, 3800
#plt.ylim(ymin, ymax)

#%% #--------------------------------------------------
# ###
# .ElasticNetCV: coordinate descent

print("Computing regularization path using the coordinate descent lasso...")
t1 = time.time()
model = ElasticNetCV(cv=K).fit(X, y)

model_enet = model
t_enet_cv = time.time() - t1

# Display results
m_log_alphas = -np.log10(model.alphas_)

alpha_enet = -np.log10(model.alpha_)
lambda_enet = (model.alpha_)
#print("model.alpha_ = {}".format(model.alpha_))
#print("model.l1_ratio_ = {}".format(model.l1_ratio_))


plt.figure()

plt.plot(model.alphas_, model.mse_path_, ':')
plt.plot(model.alphas_, model.mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(model.alpha_, linestyle='--', color='k',
            label='alpha =%f : CV estimate' % lambda_enet ) 

plt.legend()

plt.xlabel('-log(alpha)')
plt.ylabel('Mean square error')
plt.title('ENET - Mean square error on each fold: coordinate descent '
          '(train time: %.2fs)' % t_enet_cv)
plt.axis('tight')
print('\n')
print(alpha_enet)


# #############################################################################
#%% #--------------------------------------------------
#''' Lasso and Elastic Net - Paths'''
from sklearn.linear_model import ElasticNetCV
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import lasso_path, enet_path
from sklearn import datasets

# Compute paths

eps = 5e-2  # the smaller it is the longer is the path

print("Computing regularization path using the lasso...")
alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps, fit_intercept=True)


print("Computing regularization path using the elastic net...")
alphas_enet, coefs_enet, _ = enet_path(
    X, y, eps=eps, l1_ratio=0.5, fit_intercept=True)



# Display results

plt.figure()
colors = cycle(['b', 'r', 'g', 'c', 'k','m','y'])
neg_log_alphas_lasso = -np.log10(alphas_lasso)
neg_log_alphas_enet = -np.log10(alphas_enet)
for coef_l, coef_e, c in zip(coefs_lasso, coefs_enet, colors):
    l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)
    l2 = plt.plot(neg_log_alphas_enet, coef_e, linestyle='--', c=c)

plt.axvline(x=alpha_lasso, color='k', linestyle='-')
plt.axvline(x=alpha_enet, color='k', linestyle='--')
plt.xlabel('-Log(alpha)')
plt.ylabel('coefficients')
plt.title('Lasso and Elastic-Net Paths')
plt.legend((l1[-1], l2[-1]), ('Lasso', 'Elastic-Net'), loc='lower left')
plt.axis('tight')
plt.savefig(dir+"/out/lasso_enet_paths")
#%% #--------------------------------------------------
#''' Lasso Path - Labels'''
sns.set_palette("husl", 28)
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import lasso_path, enet_path
from sklearn import datasets

# Compute paths

eps = 5e-2  # the smaller it is the longer is the path

print("Computing regularization path using the lasso...")
alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps, fit_intercept=True)



# Display results
labels = X.columns
plt.figure()
#colors = cycle(['b', 'r', 'g', 'c', 'k','m','y'])
log_alphas_lasso = np.log10(alphas_lasso)
C = coefs_lasso.shape[0]
for k in range(C):
    l1 = plt.plot(alphas_lasso, coefs_lasso[k], label = labels[k],)

plt.axvline(x=lambda_lasso, color='k', linestyle='--')
plt.xlabel('$\lambda$')
plt.ylabel('Coefficients')
plt.title('Lasso Path')
plt.legend()
plt.axis('tight')
plt.savefig(dir+"/out/lassopath")
#%% #--------------------------------------------------
#################################################
#''' Interpret the model output'''
#A helper method for pretty-printing linear models
def pretty_print_linear(coefs, names = None, sort = False):
    if np.sum(names == None):
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst, key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name) for coef, name in lst)
if Poly ==0:
    print("")
    print("OLS model:", pretty_print_linear(model_ols.coef_[abs(model_ols.coef_)>0], names =  X.columns[abs(model_ols.coef_)>0] ))
    print("")
    print("Ridge model:", pretty_print_linear(model_ridge.coef_[abs(model_ols.coef_)>0], names =  X.columns[abs(model_ols.coef_)>0] ))
    print("")
    print("Lasso model:", pretty_print_linear(model_lasso.coef_[abs(model_lasso.coef_)>0], names =  X.columns[abs(model_lasso.coef_)>0] ))
    print("")
    print("Enet model:", pretty_print_linear(model_enet.coef_[abs(model_enet.coef_)>0], names =  X.columns[abs(model_enet.coef_)>0] ))

#%% #--------------------------------------------------
#'''Coefficients Plot''' 
plt.figure()
#labels.insert(0,'cons')
if Poly ==0:
    labels = list(X.columns)

    plt.plot(np.arange(len(labels)),model_ridge.coef_, color='g', linewidth=2,
             label='Ridge coefficients', alpha = 0.6)
    plt.plot(np.arange(len(labels)), model_lasso.coef_, color='r', linewidth=2,
             label='Lasso coefficients', alpha = 0.6)
    plt.plot(np.arange(len(labels)), model_enet.coef_, color='b', linewidth=2,
             label='Elastic net coefficients', alpha = 0.6)
    plt.xticks(range(len(labels)), labels, rotation=45)
    #ax[1].set_xticklabels(labels)

else:
 #   plt.plot(model_ols.coef_, '--', color='navy', label='OLS coefficients')
    plt.plot(model_ridge.coef_, color='g', linewidth=2,
             label='Ridge coefficients', alpha = 0.6)
    plt.plot(model_lasso.coef_, color='r', linewidth=2,
             label='Lasso coefficients', alpha = 0.6)
    plt.plot(model_enet.coef_, color='b', linewidth=2,
             label='Elastic net coefficients', alpha = 0.6)
    
plt.axhline(y=0,linestyle = '--', color='k')
plt.legend(loc='best')
plt.title("Ridge, Lasso, Elastic Net Coefficients")
plt.xlabel('Variables')
plt.ylabel('Coefficients')
#plt.show()
plt.savefig(dir+"/out/Coefficients")

 #%% #--------------------------------------------------
#''' Performance Metrics - In-Sample Comparison'''
print("''' Performance Metrics - In-Sample Comparison'''")
from sklearn.metrics import mean_squared_error, r2_score

def r2_adj_score(y, yhat, n, p):
    r2 =  r2_score(y, yhat)
    return 1 - (1-r2)*(n-1)/(n-p-1)

yhat_c = model_c.predict(Ones)
yhat_ols = model_ols.predict(X)
yhat_pca = model_pca.predict(X_pca)
yhat_ridge = model_ridge.predict(Xp)
yhat_lasso = model_lasso.predict(Xp)
yhat_enet = model_enet.predict(Xp)


print("R2:")
print(r2_score(y, yhat_c))
print(r2_score(y, yhat_ols))
print(r2_score(y, yhat_pca))
print(r2_score(y, yhat_lasso))
print(r2_score(y, yhat_enet))
print("R2_adj:")
print(r2_adj_score(y, yhat_c,n = y.shape[0],  p = X.shape[1]))
print(r2_adj_score(y, yhat_ols,n = y.shape[0],  p = X.shape[1]))
print(r2_adj_score(y, yhat_lasso,n = y.shape[0],  p = Xp.shape[1]))
print(r2_adj_score(y, yhat_enet,n = y.shape[0],  p = Xp.shape[1]))
print("MSE:")
print(mean_squared_error(y, yhat_c))
print(mean_squared_error(y, yhat_ols))
print(mean_squared_error(y, yhat_lasso))
print(mean_squared_error(y, yhat_enet))
'''
--> OLS performs the best in-sample
'''

#%% #--------------------------------------------------

# Performance Metrics - Cross-Validated Comparison - CV = 10
print("Performance Metrics - Cross-Validated  Comparison - CV = {}".format(K))
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics

models ={
    # 'model_c' : linear_model.LinearRegression(fit_intercept=False),
    # 'model_ols' : linear_model.LinearRegression(fit_intercept=True),
    # 'model_pca' : linear_model.LinearRegression(fit_intercept=True),
    'model_ridge' : linear_model.Ridge(alpha = lambda_ridge,fit_intercept=True),
    # 'model_lasso' : linear_model.Lasso(alpha = lambda_lasso, fit_intercept=True),
    # 'model_enet' : linear_model.ElasticNet(alpha = lambda_enet, l1_ratio=0.5, fit_intercept=True),
}
#K = 10
def test_model(model, model_name, K):
    #model = ols_model
    #model_name = 'OLS'
    if model_name == "model_c":
        XX = Ones
    elif model_name == "model_pca":
       XX = X_pca 
    else:
        XX = X
    scores = cross_val_score(model, XX, y, cv=K)
    predictions = cross_val_predict(model, XX, y, cv=K)   
    accuracy = metrics.r2_score(y, predictions)
    accuracy_adj = r2_adj_score(y, predictions,n = y.shape[0],  p = X.shape[1])
    MSE = metrics.mean_squared_error(y,predictions)

    print("Model:{}".format(model_name))
    print("Cross-Predicted R2:{}".format(accuracy))
    print("Cross-Predicted Adj R2:{}".format(accuracy_adj))
    print("Cross-Predicted MSE:{}".format(MSE))


for m_name,m in models.items():
    test_model(m,m_name, K)

'''
--> ENET and LASSO  perform better our-of-sample but R2 negative
'''
#%% #--------------------------------------------------
#'''Potential Alternative Approach '''
models ={
    'model_c' : linear_model.LinearRegression(fit_intercept=False),
    'model_ols' : linear_model.LinearRegression(fit_intercept=True),
    'model_pca' : linear_model.LinearRegression(fit_intercept=True),
    'model_ridge' : linear_model.Ridge(alpha = lambda_ridge,fit_intercept=True),
    'model_lasso' : linear_model.Lasso(alpha = lambda_lasso, fit_intercept=True),
    'model_enet' : linear_model.ElasticNet(alpha = lambda_enet, l1_ratio=0.5, fit_intercept=True),
}
print("Potential Alternative Approach")
from sklearn.model_selection import cross_validate
### Define Scorer
from sklearn.metrics import  make_scorer, mean_squared_error, r2_score

mean_squared_error_scorer = make_scorer(mean_squared_error)
scoring = {'MSE': mean_squared_error_scorer, 'r2': make_scorer(r2_score)}

### Cross-Validation + Test Sample
yhats = pd.DataFrame()
df_results = pd.DataFrame()
for m_name, m in models.items():
    if m_name == 'model_c':
        XX = Ones
        XX_test = Ones_test
    elif m_name == 'model_ols':
        XX = X
        XX_test = X_test
    elif m_name == 'model_pca':
       XX = X_pca 
       XX_test = X_test_pca
    else:
        XX = X
        XX_test = X_test
    # Cross-Validation 
    cv_results = cross_validate(m, XX, y, cv  = K , return_train_score=True, scoring = scoring )
    df_cv = pd.DataFrame.from_dict(cv_results)
    df_avg = df_cv.drop(columns=['fit_time', 'score_time']).mean(axis = 0)
    
    df_avg = df_avg.rename(index={ "test_MSE" : "valid_MSE", "test_r2" : "valid_r2"})

    
    # Test Sample
    yhat = m.fit(XX,y).predict(XX_test)
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

 #%% #--------------------------------------------------
# Multicollinearity
 
#corr=np.corrcoef(X,rowvar=0)
#corr
#W,V=np.linalg.eig(corr)
#print(W)
#list(X)
#Xcorr = X.corr()
#sns.heatmap(Xcorr, annot=True)

#--> Multicollinearity in data

 #%% #--------------------------------------------------

# Performance Metrics - Out_of-Sample Comparison
print(" Performance Metrics - Out_of-Sample Comparison")
from sklearn.metrics import mean_squared_error, r2_score
def r2_adj_score(y, yhat, n, p):
    r2 =  r2_score(y, yhat)
    return 1 - (1-r2)*(n-1)/(n-p-1)

yhat_c = model_c.predict(Ones_test)
yhat_ols = model_ols.predict(X_test)
yhat_pca = model_pca.predict(X_test_pca)
yhat_ridge = model_ridge.predict(Xp_test)
yhat_lasso = model_lasso.predict(Xp_test)
yhat_enet = model_enet.predict(Xp_test)

yhats_old = [yhat_c,  yhat_pca, yhat_ols,  yhat_ridge, yhat_lasso, yhat_enet]
yhats_names = ['c_model','pca_model', 'ols_model','ridge_model', 'lasso_model', 'enet_model']
print("MSE:")
for yh,yh_n in zip(yhats_old,yhats_names):
    print("mean_squared_error(y_test, {})".format(yh_n))
    print(mean_squared_error(y_test, yh))
    print("r2_score(y_test, {})".format(yh_n))
    print(r2_score(y_test, yh))


## Print the Results
os.makedirs(dir+"out/oos/",exist_ok = True)
f = open(dir+"/out/oos/K{}_TsizeInv{}_Poly{}_Period{}.txt".format(K,TsizeInv,Poly,Period), 'w')
f.write("MSE:")
for yh,yh_n in zip(yhats_old,yhats_names):
    f.write("mean_squared_error(y_test, {})\n".format(yh_n))
    f.write("{}\n".format(mean_squared_error(y_test, yh)))
f.close()

#%% #--------------------------------------------------
#'''Prediction Plot''' 
plt.figure(figsize=(15,7.5))
x= np.array(df['ym'].loc[y_test.index]).astype(int)
#plt.plot(model_ols.coef_, '--', color='navy', label='OLS coefficients')
colors = cycle(['c','m','y', 'g', 'r','b', 'k'])
yhats_old = [yhat_c,  yhat_pca, yhat_ols,  yhat_ridge, yhat_lasso, yhat_enet]
yhats_names = ['Const', 'PCA',	'OLS',	'Ridge', 'Lasso', 'Enet']
plt.plot(np.arange(len(x)),y_test,'--', color='k', linewidth=1,
             label='Realized Return')
for yh,yh_n,color in zip(yhats_old,yhats_names,colors):
    plt.plot(np.arange(len(x)),yh, linewidth=2, color = color)
#label="{} prediction".format(yh_n)
#    
#    plt.plot(x,yhat_ols,'--', color='r', linewidth=2,
#             label='OLS prediction')
#    plt.plot(x,yhat_c, '--',color='b', linewidth=2,
#             label='Constant prediction')
#    plt.plot(x,yhat_lasso,'--', color='y', linewidth=2,
#             label='Lasso prediction')
#    plt.plot(x,yhat_pca, '--', linewidth=2,
#             label='PCA prediction')
#    plt.plot(x,yhat_ridge,'--', linewidth=2,
#             label='Ridge prediction')
plt.xticks(range(len(x)), x, rotation=45)
yhats_names.insert(0,'Realized Return')
plt.xlabel('Date')
plt.ylabel('Monthly Return in %')
plt.legend(yhats_names)
plt.savefig(dir+"/out/Prediction")
#%% #--------------------------------------------------
plt.figure(figsize=(15,7.5))
ym_test= pd.DataFrame(df['date'].loc[y_test.index]).reset_index(drop=True)
y_test_fig = y_test.reset_index(drop = True)
plotdata = pd.concat([ym_test,y_test_fig,yhats],axis = 1)
plotdata = plotdata.melt(id_vars='date', var_name='model',  value_name='return')
sns.set_palette("deep")
sns.lineplot(x='date',y='return', hue ='model', data = plotdata )
plt.savefig(dir+"/out/lineplot_predict")
plt.show()

#%%
# sns.reset_defaults()
# g = sns.FacetGrid(plotdata, hue='model', height = 5, aspect= 2 )
# g = g.map(plt.bar, 'date','return' , alpha=0.7)
# g = g.add_legend()
# plt.show()
# plt.savefig(dir+"/out/barplot_predict")
#%%
%qtconsole


#%%
sns.set()

#%%
