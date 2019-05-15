

#%% #--------------------------------------------------

sns.reset_defaults()
g = sns.FacetGrid(plotdata, hue='model', height = 5, aspect= 2 )
g = g.map(plt.bar, 'date','return' , alpha=0.7)
g = g.add_legend()
plt.show()
plt.savefig(dir+"/out/barplot_predict")

#%% #--------------------------------------------------
* Define Function to Draw Cross-Validation for Optimal Lambda
def display_optimal_alpha(model,model_name):
    # Display results
    m_log_alphas = -np.log10(model.alphas_)


    plt.figure()

    plt.plot(model.alphas_, model.mse_path_, ':')
    plt.plot(model.alphas_, model.mse_path_.mean(axis=-1), 'k',
            label='Average across the folds', linewidth=2)
    plt.axvline(model.alpha_, linestyle='--', color='k',
                label='alpha = %f: CV estimate' % model.alpha_)

    plt.legend()

    plt.xlabel('$\lambda$')
    plt.ylabel('Mean square error')
    plt.title(model_name + ' - Mean square error on each fold: coordinate descent ')
    plt.axis('tight')
    print(-np.log10(model.alpha_))
    #ymin, ymax = 2300, 3800
    #plt.ylim(ymin, ymax)
    plt.show()
    return plt


fig = display_optimal_alpha(model_lasso, 'lasso')
plt.savefig(dir+"/out/lasso_cv")
fig = display_optimal_alpha(model_ridge, 'ridge')
plt.savefig(dir+"/out/ridge_cv")
fig = display_optimal_alpha(model_enet, 'enet')
plt.savefig(dir+"/out/enet_cv")


#%% #--------------------------------------------------
#''' Lasso and Enet Path - Labels'''

from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import lasso_path, enet_path
from sklearn import datasets

def lambda_path(model_name, lmbda):

    if model_name == "enet": 
        l1 = 0.5
    elif model_name == "lasso":  
        l1 = 1
    elif model_name == "ridge":  
        l1 = 0
    # Compute paths
    eps = 5e-2  # the smaller it is the longer is the path

    print("Computing regularization path using the lasso...")
    alphas, coefs, _ =  enet_path(X, y, eps=eps, l1_ratio=l1, fit_intercept=True)

    # Display results
    labels = X.columns
    plt.figure()
    #colors = cycle(['b', 'r', 'g', 'c', 'k','m','y'])
    log_alphas = np.log10(alphas)
    C = coefs.shape[0]
    for k in range(C):
        l1 = plt.plot(alphas, coefs[k], label = labels[k],)

    plt.axvline(x=lmbda, color='k', linestyle='--')
    plt.xlabel('$\lambda$')
    plt.ylabel('Coefficients')
    plt.title('Lasso Path')
    plt.legend()
    plt.axis('tight')
    plt.show()
    return plt

sns.set_palette("husl", 28)
fig = lambda_path("lasso", lambda_lasso)
fig.savefig(dir+"/out/lasso_path")
fig = lambda_path("enet", lambda_enet)
fig.savefig(dir+"/out/enet_path")
sns.set_palette("deep")


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
Multicollinearity

corr=np.corrcoef(X,rowvar=0)
corr
W,V=np.linalg.eig(corr)
print(W)
list(X)
Xcorr = X.corr()
plt.figure(figsize=(30,15))
sns.heatmap(Xcorr, annot=True)
plt.savefig(dir+"/out/Coefficients")
--> Multicollinearity in data
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
yhat_ridge = model_ridge.predict(X_test)
yhat_lasso = model_lasso.predict(X_test)
yhat_enet = model_enet.predict(X_test)

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
plt.xticks(range(len(x)), x, rotation=45)
yhats_names.insert(0,'Realized Return')
plt.xlabel('Date')
plt.ylabel('Monthly Return in %')
plt.legend(yhats_names)
plt.savefig(dir+"/out/Prediction")
