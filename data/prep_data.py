import sys
sys.path.append('../')
from utils import paths
from utils import utils

import os
import warnings
from tqdm import tqdm
import pandas as pd
import numpy as np
import datetime
import xarray as xr
import time
from pyTMD import compute_tide_corrections


# Define function to prepare data
def GetFormattedData(location,SAVE_DATA=False):

    start = time.time()
    warnings.filterwarnings("ignore")
    ps_data, u_data, v_data, t_data, pr_data = utils.Step1()
    ps_data, u_data, v_data, t_data, pr_data = utils.Step2(ps_data, u_data, v_data, t_data, pr_data)
    ps_window, u_window, v_window, t_window, pr_window = utils.Step3(ps_data, u_data, v_data, t_data, pr_data, location)
    df_ps, df_u, df_v, df_t, df_pr = utils.Step4(ps_window, u_window, v_window, t_window, pr_window)
    df_ps, df_u, df_v, df_t, df_pr = utils.Step5(df_ps, df_u, df_v, df_t, df_pr)
    df_ps, df_u, df_v, df_t, df_pr = utils.Step6(df_ps, df_u, df_v, df_t, df_pr)
    df_merged = utils.Step7(df_ps, df_u, df_v, df_t, df_pr)
    df_merged = utils.Step8(df_merged)
    df_merged = utils.Step9(df_merged)
    df_merged = utils.Step10(df_merged)
    df_merged = utils.Step11(df_merged)
    df_merged = utils.Step12(df_merged, location)
    df_merged = utils.Step13(df_merged, location)
    end = time.time()
    print("Time taken to get the data: ", (end-start)/60, ' mins.')
    if SAVE_DATA:
        df_merged.to_csv('./final_df_' + str(location) + '.csv')
    print('Data for ', str(location), 'is saved.')
    return df_merged

def get_location():
    # Check if a command-line argument is provided
    if len(sys.argv) > 1:
        return sys.argv[1]
    else:
        # Prompt the user for input if no command-line argument is provided
        return input("Enter the city name: ")



# Main function
if __name__ == "__main__":
    # Specify the location in the UK
    location = get_location()
    
    SAVE_DATA = True

    GetFormattedData(location, SAVE_DATA)
