#%% #--------------------------------------------------
#* Global Parameters *

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


VERBOSE = True



#%%
