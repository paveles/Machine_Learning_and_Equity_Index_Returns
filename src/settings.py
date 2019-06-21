#%% #--------------------------------------------------
#* Global Parameters *
from src.model_configs import *
# Starting Year: 1928 - macro only, 1951 - macto + technical, 
# 1974 - add short interest    

Period = 1951


# Estimate using Rolling Window or Expanding

ROLLING = False

min_idx = 0

start_idx = 180

# Rolling or Expanding Window

if ROLLING == True:
    Models_Folder = 'rolling'
else:
    Models_Folder = 'expanding'

# Print Intermediary Results
VERBOSE = True

# Different Models to Use at different steps
Configs_Estimated ={
    'const' : const_config,
    'ols' : ols_config,
    # 'pca' : pca_config, #~ 23 minutes
    # 'enet' : enet_config, #~ 2.5 hours
    # 'pca_enet' : pca_enet_config, #~ 3 hours
    # 'adab_nocv' : adab_nocv_config,
    # 'gbr_nocv': gbr_nocv_config,
    # 'rf_nocv': rf_nocv_config,
    # 'xgb_nocv': xgb_nocv_config,
    # 'adab': adab_config,
    # 'gbr': gbr_config,
    # 'rf': rf_config,
    # 'xgb' : xgb_config
}
Configs_Aggregate ={
    'const' : const_config,
    'ols' : ols_config,
    'pca' : pca_config, #~ 23 minutes
    'enet' : enet_config, #~ 2.5 hours
    'pca_enet' : pca_enet_config, #~ 3 hours
    'adab_nocv' : adab_nocv_config,
    'gbr_nocv': gbr_nocv_config,
    'rf_nocv': rf_nocv_config,
    'xgb_nocv': xgb_nocv_config,
    'adab': adab_config,
    'gbr': gbr_config,
    'rf': rf_config,
    'xgb' : xgb_config
}

Configs_Analysis ={
     'const' : const_config,
    #'ols' : ols_config,
    # 'pca' : pca_config, #~ 23 minutes
    # 'enet' : enet_config, #~ 2.5 hours
    # 'pca_enet' : pca_enet_config, #~ 3 hours
    # 'adab_nocv' : adab_nocv_config,
    # 'gbr_nocv': gbr_nocv_config,
    # 'rf_nocv': rf_nocv_config,
    # 'xgb_nocv': xgb_nocv_config,
    # 'adab': adab_config,
    # 'gbr': gbr_config,
    # 'rf': rf_config,
    # 'xgb' : xgb_config
}

Configs_Visualize ={
    'enet' : enet_config,
    'const' : const_config,
    'ols' : ols_config,
    'rf' : rf_config,
}


#%%
