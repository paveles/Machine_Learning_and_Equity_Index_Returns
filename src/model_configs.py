"""
Define Configurations of the Model - dictionaries with the following entries.

Parameters
----------
- name - name of the model
- cv - cross-validation procedure
- pipeline - pipeline to use 
- grid_search - grid search methods, e.g. GridSearchCV 
- param_grid - parameters of the pipline that should be tuned by the grid-search method 
- scorer - score criteria, according to which the best model will be chosen
For more details see the descriptions of the respective models .

"""
#* Basic Modules
import time
from scipy import stats
import pandas as pd
import numpy as np

#* Load Models to Estimate
from sklearn.linear_model import  LinearRegression, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

#* Transformations, Pipelines and CV Methods
from sklearn.preprocessing import StandardScaler,MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import  TimeSeriesSplit
from transform_cv import TimeSeriesSplitMod
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import  make_scorer, mean_squared_error, r2_score
from transform_cv import DisabledCV, ToConstantTransformer, ToNumpyTransformer

#? OLS Models
ols_config = {}
ols_config['name'] = "ols"
ols_config['cv'] = DisabledCV

ols_config['pipeline'] = Pipeline(steps=[
    ('ols', LinearRegression())
])

# list(range(1, X.shape[1] + 1))
ols_config['param_grid'] = {'ols__fit_intercept':[True]}
ols_config['scorer'] = make_scorer(mean_squared_error, greater_is_better=False)
ols_config['grid_search'] = GridSearchCV


#? CONST Models
const_config = {}

const_config['name'] = "const"
const_config['cv'] = DisabledCV
const_config['pipeline'] = Pipeline(steps=[
    ('const', ToConstantTransformer()),
    ('ols', LinearRegression())
])

# list(range(1, X.shape[1] + 1))
const_config['param_grid'] = {'ols__fit_intercept':[False]}
const_config['scorer'] = make_scorer(mean_squared_error, greater_is_better=False)
const_config['grid_search'] = GridSearchCV



#? PCA Models
pca_config = {}
pca_config['name'] = "pca"
pca_config['cv'] = TimeSeriesSplitMod # DisabledCV

pca_config['pipeline'] = Pipeline(steps=[
   ('pca', PCA()),
    ('ols', LinearRegression())
])

# list(range(1, X.shape[1] + 1))
pca_config['param_grid'] = {'pca__n_components': [1,2,3,4,5,6,7,8,9,10]  }
pca_config['scorer'] = make_scorer(mean_squared_error, greater_is_better=False)
pca_config['grid_search'] = GridSearchCV

#? Enet  Model
enet_config = {}
enet_config['name'] = "enet"
enet_config['cv'] = TimeSeriesSplitMod # DisabledCV

enet_config['pipeline'] = Pipeline(steps=[
    ('standard', StandardScaler()),
    ('enet', ElasticNet())
])

# list(range(1, X.shape[1] + 1))
enet_config['param_grid'] = {'enet__alpha': [0.1,  0.5 , 0.7, 0.9,  0.97, 0.99],
                            'enet__l1_ratio': [0, 0.25 , 0.5, 0.75, 1],
                            'enet__random_state' : [0],
                            }
enet_config['scorer'] = make_scorer(mean_squared_error, greater_is_better=False)
enet_config['grid_search'] = GridSearchCV

#? PCA + Enet  Model
pca_enet_config = {}
pca_enet_config['name'] = "pca_enet"
pca_enet_config['cv'] = TimeSeriesSplitMod # DisabledCV

pca_enet_config['pipeline'] = Pipeline(steps=[
    ('standard', StandardScaler()),
    ('pca', PCA()),
    ('enet', ElasticNet())
])

# list(range(1, X.shape[1] + 1))
pca_enet_config['param_grid'] = {'enet__alpha': [0.1,  0.5 , 0.7, 0.9,  0.97, 0.99],
                            'enet__l1_ratio': [0, 0.25 , 0.5, 0.75, 1],
                            'enet__random_state' : [0],
                            }
pca_enet_config['scorer'] = make_scorer(mean_squared_error, greater_is_better=False)
pca_enet_config['grid_search'] = GridSearchCV


#? Enet + Lag  Model
lag_enet_config = {}
lag_enet_config['name'] = "lag_enet"
lag_enet_config['cv'] = TimeSeriesSplitMod # DisabledCV
lag_enet_config['addlags'] = 1

lag_enet_config['pipeline'] = Pipeline(steps=[
    ('standard', StandardScaler()),
    ('enet', ElasticNet())
])

# list(range(1, X.shape[1] + 1))
lag_enet_config['param_grid'] = {'enet__alpha': [0.1,  0.5 , 0.7, 0.9,  0.97, 0.99],
                            'enet__l1_ratio': [0, 0.25 , 0.5, 0.75, 1],
                            'enet__random_state' : [0],
                            }
lag_enet_config['scorer'] = make_scorer(mean_squared_error, greater_is_better=False)
lag_enet_config['grid_search'] = GridSearchCV

#? Enet + Interact  Model
poly_enet_config = {}
poly_enet_config['name'] = "poly_enet"
poly_enet_config['cv'] = DisabledCV # DisabledCV
poly_enet_config['interactions']=True

poly_enet_config['pipeline'] = Pipeline(steps=[
    ('standard', StandardScaler()),
    ('enet', ElasticNet())
])

# list(range(1, X.shape[1] + 1))
poly_enet_config['param_grid'] = {'enet__alpha': [0.1,  0.5 , 0.7, 0.9,  0.97, 0.99],
                            'enet__l1_ratio': [0, 0.25 , 0.5, 0.75, 1],
                            'enet__random_state' : [0],
                            }
poly_enet_config['scorer'] = make_scorer(mean_squared_error, greater_is_better=False)
poly_enet_config['grid_search'] = GridSearchCV



#? Enet + Interact + Lag  Model
poly_lag_enet_config = {}
poly_lag_enet_config['name'] = "poly_lag_enet"
poly_lag_enet_config['cv'] = TimeSeriesSplitMod # DisabledCV
poly_lag_enet_config['interactions'] = True
poly_lag_enet_config['addlags'] = 1

poly_lag_enet_config['pipeline'] = Pipeline(steps=[
    ('standard', StandardScaler()),
    ('enet', ElasticNet())
])

# list(range(1, X.shape[1] + 1))
poly_lag_enet_config['param_grid'] = {'enet__alpha': [0.1,  0.5 , 0.7, 0.9,  0.97, 0.99],
                            'enet__l1_ratio': [0, 0.25 , 0.5, 0.75, 1],
                            'enet__random_state' : [0],
                            }
poly_lag_enet_config['scorer'] = make_scorer(mean_squared_error, greater_is_better=False)
poly_lag_enet_config['grid_search'] = GridSearchCV

#! No Cross-Validation!!!
#? Random Forest Model + No CV
rf_nocv_config = {}
rf_nocv_config['name'] = "rf_nocv"
rf_nocv_config['cv'] = DisabledCV # DisabledCV TimeSeriesSplitMod

rf_nocv_config['pipeline'] = Pipeline(steps=[
    ('rf', RandomForestRegressor())
])

# list(range(1, X.shape[1] + 1))
rf_nocv_config['param_grid'] = {
                            'rf__random_state' : [0],
                            'rf__n_estimators': [100],
                            'rf__max_depth':[15]
                            }
rf_nocv_config['scorer'] = make_scorer(mean_squared_error, greater_is_better=False)
rf_nocv_config['grid_search'] = GridSearchCV

#? AdaBoostRegressor Model + No CV
adab_nocv_config = {}
adab_nocv_config['name'] = "adab_nocv"
adab_nocv_config['cv'] = DisabledCV # DisabledCV TimeSeriesSplitMod

adab_nocv_config['pipeline'] = Pipeline(steps=[
    ('adab', AdaBoostRegressor())
])

# list(range(1, X.shape[1] + 1))
adab_nocv_config['param_grid'] = {
                            'adab__random_state' : [0],
                            'adab__n_estimators': [100],
                            }
adab_nocv_config['scorer'] = make_scorer(mean_squared_error, greater_is_better=False)
adab_nocv_config['grid_search'] = GridSearchCV

#? GradientBoostingRegressor  Model + No CV
gbr_nocv_config = {}
gbr_nocv_config['name'] = "gbr_nocv"
gbr_nocv_config['cv'] = DisabledCV # DisabledCV TimeSeriesSplitMod

gbr_nocv_config['pipeline'] = Pipeline(steps=[
    ('gbr', GradientBoostingRegressor())
])

# list(range(1, X.shape[1] + 1))
gbr_nocv_config['param_grid'] = {
                            'gbr__random_state' : [0],
                            'gbr__n_estimators':[100],
                            # 'gbr__random_state' : [0],
                            # 'gbr__n_estimators':[ 200],
                            # 'gbr__max_depth':[ 10],
                            # 'gbr__learning_rate':[0.05, ],
                            }
gbr_nocv_config['scorer'] = make_scorer(mean_squared_error, greater_is_better=False)
gbr_nocv_config['grid_search'] = GridSearchCV

#? XGB  Model + No CV
xgb_nocv_config = {}
xgb_nocv_config['name'] = "xgb_nocv"
xgb_nocv_config['cv'] = DisabledCV # DisabledCV TimeSeriesSplitMod

xgb_nocv_config['pipeline'] = Pipeline(steps=[
    ('to_numpy', ToNumpyTransformer()),
    ('xgb', XGBRegressor())
])

# list(range(1, X.shape[1] + 1))
xgb_nocv_config['param_grid'] = {
                            'xgb__random_state' : [0],
                            'xgb__n_estimators':[100],
                            }
xgb_nocv_config['scorer'] = make_scorer(mean_squared_error, greater_is_better=False)
xgb_nocv_config['grid_search'] = GridSearchCV
#! With Cross-Validation

#? Random Forest Model 
rf_config = {}
rf_config['name'] = "rf"
rf_config['cv'] = TimeSeriesSplitMod # DisabledCV TimeSeriesSplitMod

rf_config['pipeline'] = Pipeline(steps=[
    ('rf', RandomForestRegressor())
])

# list(range(1, X.shape[1] + 1))
rf_config['param_grid'] = {
                            'rf__random_state' : [0],
                            'rf__n_estimators': [25, 100],
                            'rf__max_depth':[5, 20],
                            'rf__min_samples_leaf' : [1, 3],
                            'rf__max_features' : [9, 'sqrt']
                            }
rf_config['scorer'] = make_scorer(mean_squared_error, greater_is_better=False)
rf_config['grid_search'] = GridSearchCV

#? AdaBoostRegressor Model 
adab_config = {}
adab_config['name'] = "adab"
adab_config['cv'] = TimeSeriesSplitMod # DisabledCV TimeSeriesSplitMod

adab_config['pipeline'] = Pipeline(steps=[
    ('adab', AdaBoostRegressor())
])

# list(range(1, X.shape[1] + 1))
adab_config['param_grid'] = {
                            'adab__random_state' : [0],
                            'adab__n_estimators': [25, 100, 200],
                            'adab__base_estimator':[DecisionTreeRegressor(max_depth=3), 
                            DecisionTreeRegressor(max_depth=5), DecisionTreeRegressor(max_depth=10)],
                            'adab__learning_rate':[0.05, 0.1, 0.2],
                            }
adab_config['scorer'] = make_scorer(mean_squared_error, greater_is_better=False)
adab_config['grid_search'] = GridSearchCV

#? GradientBoostingRegressor  Model 
gbr_config = {}
gbr_config['name'] = "gbr"
gbr_config['cv'] = TimeSeriesSplitMod # DisabledCV TimeSeriesSplitMod

gbr_config['pipeline'] = Pipeline(steps=[
    ('gbr', GradientBoostingRegressor())
])

# list(range(1, X.shape[1] + 1))
gbr_config['param_grid'] = {
                            'gbr__random_state' : [0],
                            'gbr__n_estimators':[25, 100, 200],
                            'gbr__max_depth':[3, 5, 10],
                            'gbr__learning_rate':[0.05, 0.1, 0.2],
                            }
gbr_config['scorer'] = make_scorer(mean_squared_error, greater_is_better=False)
gbr_config['grid_search'] = GridSearchCV

#? XGB  Model 
xgb_config = {}
xgb_config['name'] = "xgb"
xgb_config['cv'] = TimeSeriesSplitMod # DisabledCV TimeSeriesSplitMod

xgb_config['pipeline'] = Pipeline(steps=[
    ('to_numpy', ToNumpyTransformer()),
    ('xgb', XGBRegressor())
])

# list(range(1, X.shape[1] + 1))
xgb_config['param_grid'] = {
                            'xgb__random_state' : [0],
                            'xgb__n_estimators':[25, 100],
                            'xgb__max_depth':[ 5, 10],
                            'xgb__eta':[0.05, 0.1],
                            'xgb__alpha':[1, 0.5],
                            'xgb__lambda':[0, 0.5],
                            }
xgb_config['scorer'] = make_scorer(mean_squared_error, greater_is_better=False)
xgb_config['grid_search'] = GridSearchCV