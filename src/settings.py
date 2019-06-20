#%% #--------------------------------------------------
#* Global Parameters *
def init():
    # Starting Year: 1928 - macro only, 1951 - macto + technical, 
    # 1974 - add short interest    
    global Period
    Period = 1951


    # Estimate using Rolling Window or Expanding
    global ROLLING
    ROLLING = False

    global min_idx
    min_idx = 0
    global start_idx 
    start_idx = 180

    # Rolling or Expanding Window
    global Model_Folder
    if ROLLING == True:
        Models_Folder = 'rolling'
    else:
        Models_Folder = 'expanding'

    global VERBOSE 
    VERBOSE = True



#%%
