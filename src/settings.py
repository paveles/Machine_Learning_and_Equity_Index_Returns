""" 
Predicting Equity Index Returns using Machine Learning Methods - Settings File
"""

#%% #--------------------------------------------------
#* Global Parameters *
# Import Possible Model Configs
from src.model_configs import *
# Starting Year: 1928 - macro only, 1951 - macto + technical, 
# 1974 - add short interest    

Period = 1951


# Estimate using Rolling Window or Expanding
ROLLING = False

# How many months in a fixed-window to use in a rolling regression
training_window = 240
# How many months to use for validation in a rolling regression
validation_window = 24 
# How many months used to train till the first prediction
start_idx = 180


# Rolling or Expanding Window Parameters
if ROLLING == True:
    Models_Folder = 'rolling'
    if start_idx < training_window:
        # To make sure that we have  at least training window for the first est.
        start_idx = training_window 
else:
    Models_Folder = 'expanding'

# Print Intermediary Results
VERBOSE = True

CONFIGS={
    'const' : const_config,
}
