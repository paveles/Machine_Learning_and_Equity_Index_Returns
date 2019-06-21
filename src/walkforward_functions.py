"""
Module containing walk-forward functions
"""
#* Walk-Forward Modeling
from sklearn.linear_model import  LinearRegression
from scipy import stats
from sklearn.metrics import  make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from src.transform_cv import TimeSeriesSplitMod
from src.transform_cv import DisabledCV, ToConstantTransformer, ToNumpyTransformer
import pandas as pd
from sklearn.preprocessing import  PolynomialFeatures

def calculate_r2_wf(y_true, y_pred, y_moving_mean):
    """
    Calculate out-of-sample R^2 for the walk-forward procedure
    """
    mse_urestricted = ((y_true - y_pred)**2).sum()
    mse_restricted = ((y_true - y_moving_mean)**2).sum()
    return 1 - mse_urestricted/mse_restricted

def calculate_msfe_adjusted(y_true, y_pred, y_moving_mean):
    """
    Calculate t-statistic for the test on significant imporvement in predictions
    """
    f = (y_true - y_moving_mean)**2 - ((y_true - y_pred)**2 - (y_moving_mean - y_pred)**2)
    t_stat,pval_two_sided = stats.ttest_1samp(f, 0, axis=0)
    pval_one_sided = stats.t.sf(t_stat, f.count() - 1)
    return t_stat, pval_one_sided

def r2_adj_score(y_true,y_pred,N,K):
    """
    Calculate in-sample R^2 that is adjusted for the number of predictors (ols model only)
    """
    r2 = r2_score(y_true,y_pred)
    return 1-(1-r2)*(N-1)/(N-K-1)

def estimate_walk_forward(config, X, y, start_idx, rolling = False,
 tr_win = None, val_win = None, verbose = True):
    """
    Function that esimates walk-forward using expanding or rolling window.
    Cross-validation procedure, and the type of grid-search are determined in the config file.
    Please see "model_configs.py" for the model config structure.

    Yields
    ---------
    Outputs are pandas dataseries:
        - models_estimated - best model estimated for given month using past info
        - scores_estimated - scores of the best models
        - predictions - predictions of the best models
    """
    if verbose == True:
        print(config['param_grid'])

    max_idx = y.shape[0]
    # Generate Interaction Terms
    if 'interactions' in config:
        if config['interactions'] == True:
            X = pd.DataFrame(PolynomialFeatures(interaction_only=True,include_bias = False).fit_transform(X),index = X.index)

    # Generate Lags
    if 'addlags' in config:
        LAGS = config['addlags']
        if (type(LAGS) == int) & (LAGS > 0):
            for lag in range(1,LAGS+1,1):
                X = pd.concat([X, X.shift(lag).add_suffix('_L{}'.format(lag))], axis = 1)
                # temp.iloc[0,(X.shape[1]):] = X.iloc[0,:].values)
                # X = temp
    # Define outputs
    models_estimated = pd.Series(index=X.index[start_idx:])
    scores_estimated = pd.Series(index=X.index[start_idx:])
    predictions = pd.Series(index=X.index[start_idx:])
    
    # Pipeline to Estimate
    model_to_estimate = config['pipeline'] 
    
    # Scorer to Use
    scorer = config['scorer']
    
    for idx in range(start_idx,max_idx,1):
        # Different Cross-Validation Procedures

        # For Fixed Window Rolling Regression, 
        # the Window Size is 240 Months,
        # the Validation Period Size is 24 Months,
        # a prediction is made for the following month
        if rolling == True:
            X_tr = X.iloc[idx - tr_win : idx]
            y_tr = y.iloc[idx - tr_win : idx]
            if config['cv'] == TimeSeriesSplitMod:
                cv = TimeSeriesSplitMod( n_splits =tr_win - 1,
                 start_test_split = tr_win-val_win ).split(X_tr,y_tr)
            elif config['cv'] == DisabledCV:
                cv = DisabledCV().split(X_tr,y_tr)
        # For Exapnding Window Regression,
        # the starting period 'start_idx' is extended by 1 month 
        # a new model is estimated, using the cross-validation procedure
        # and a prediction is made for the next month
        else:
            X_tr = X.iloc[0 : idx]
            y_tr = y.iloc[0 : idx]
            if config['cv'] == TimeSeriesSplitMod:
                cv = TimeSeriesSplitMod( n_splits =idx - 1, start_test_split = start_idx - 1 ).split(X_tr,y_tr)
            elif config['cv'] == DisabledCV:
                cv = DisabledCV().split(X_tr,y_tr)
        # Grid Search Function and Grid of Parameters
        grid_search = config['grid_search']
        param_grid = config['param_grid'] 
    
        grid = grid_search(estimator=model_to_estimate, param_grid=param_grid, cv=cv \
            , scoring = scorer, n_jobs=-1)
        
        # Best model, best score and the respective prediction    
        model = grid.fit(X_tr,y_tr)
        best_model = model.best_estimator_
        best_score = model.best_score_
        models_estimated.loc[X.index[idx]] = best_model # save the model
        scores_estimated.loc[X.index[idx]] = best_score # save the score
        predictions.loc[X.index[idx]] = model.predict([X.iloc[idx]]) # predict next month 
        
        if verbose == True:        
            if ((idx-start_idx) % 10) == 0:
                print(str(idx)+" / "+str(max_idx) )
                print(best_model)
                print(best_score)

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
   