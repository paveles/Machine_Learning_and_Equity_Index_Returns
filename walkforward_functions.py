'''
Module containing walk-forward functions
'''
#* Walk-Forward Modeling
from sklearn.linear_model import  LinearRegression
from scipy import stats
from sklearn.metrics import  make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from transform_cv import TimeSeriesSplitMod
from transform_cv import DisabledCV, ToConstantTransformer, ToNumpyTransformer
import pandas as pd

def calculate_r2_wf(y_true, y_pred, y_moving_mean):
    '''
    Calculate out-of-sample R^2 for the walk-forward procedure
    '''
    mse_urestricted = ((y_true - y_pred)**2).sum()
    mse_restricted = ((y_true - y_moving_mean)**2).sum()
    return 1 - mse_urestricted/mse_restricted

def calculate_msfe_adjusted(y_true, y_pred, y_moving_mean):
    '''
    Calculate t-statistic for the test on significant imporvement in predictions
    '''
    f = (y_true - y_moving_mean)**2 - ((y_true - y_pred)**2 - (y_moving_mean - y_pred)**2)
    t_stat,pval_two_sided = stats.ttest_1samp(f, 0, axis=0)
    pval_one_sided = stats.t.sf(t_stat, f.count() - 1)
    return t_stat, pval_one_sided

def r2_adj_score(y_true,y_pred,N,K):
'''
Calculate in-sample R^2 that is adjusted for the number of predictors (ols model only)
'''
    r2 = r2_score(y_true,y_pred)
    return 1-(1-r2)*(N-1)/(N-K-1)

def estimate_walk_forward(config, X, y, start_idx, max_idx, rolling = False, verbose = True):
'''
Function that esimate walk-forward using expanding or rolling window.
Cross-validation procedure, and type of grid-search are provided in the config file.
Please see "model_configs.py" for the model config structure.
Outputs are pandas timeseries of:
    models_estimated - best model estimated for given month using past info
    scores_estimated - scores of the best models
    predictions - predictions of the best models
'''
    if verbose == True:
        print(config['param_grid'])

    # Define outputs
    models_estimated = pd.Series(index=X.index[start_idx:])
    scores_estimated = pd.Series(index=X.index[start_idx:])
    predictions = pd.Series(index=X.index[start_idx:])
    
    # Pipeline to Estimate
    model_to_estimate = config['pipeline'] 
    
    # Scorer to Use
    scorer = config['scorer']
    
    # Grid Search Function and Grid of Parameters
    grid_search = config['grid_search']
    param_grid = config['param_grid']


    # Different Cross-Validation Procedures
    for idx in range(start_idx,max_idx,1):    
        if rolling == True:
            X_tr = X.iloc[idx - 240 : idx]
            y_tr = y.iloc[idx - 240 : idx]
            if config['cv'] == TimeSeriesSplitMod:
                cv = config['cv']( n_splits =240 - 1, start_test_split = 240-24 ).split(X_tr,y_tr)
            elif config['cv'] == DisabledCV:
                cv = config['cv']().split(X_tr,y_tr)
        else:
            X_tr = X.iloc[0 : idx]
            y_tr = y.iloc[0 : idx]
            if config['cv'] == TimeSeriesSplitMod:
                cv = config['cv']( n_splits =idx - 1, start_test_split = start_idx - 1 ).split(X_tr,y_tr)
            elif config['cv'] == DisabledCV:
                cv = config['cv']().split(X_tr,y_tr)
        
        grid = grid_search(estimator=model_to_estimate, param_grid=param_grid, cv=cv \
            , scoring = scorer, n_jobs=-1)        
        model = grid.fit(X_tr,y_tr)
        best_model = model.best_estimator_
        best_score = model.best_score_
        models_estimated.loc[X.index[idx]] = best_model # save the model
        if verbose == True:        
            if ((idx-start_idx) % 10) == 0:
                print(str(idx)+" / "+str(max_idx) )
                print(best_model)
                print(best_score)

        scores_estimated.loc[X.index[idx]] = best_score # save the score
        predictions.loc[X.index[idx]] = model.predict([X.iloc[idx]]) # predict next month 
    return models_estimated,scores_estimated, predictions
    
#%% #--------------------------------------------------
# #* Check How Indexes and Cross-Validation are Calculated
# idx = len(yo)-1
# tscv = TimeSeriesSplitMod( n_splits = 180 - 1, start_test_split = 156 ) #( n_splits =idx - 1, start_test_split = start_idx - 1 )

# # print(tscv)  
# X_tr = Xo.iloc[idx - 180 : idx] # Xo[0:idx]
# y_tr = yo.iloc[idx - 180 : idx] # yo[0:idx]
# i = 0
# for train_index, test_index in tscv.split(X_tr, y_tr):
#     i+=1
#     print("TRAIN:", y_tr.index[train_index], "TEST:", y_tr.index[test_index])
#     print( len(y_tr.index[train_index]), len(y_tr.index[test_index]))
# #print(Xo.index[idx])
# print(i)
# print( idx)
# ##* Correct!
   