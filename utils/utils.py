
from utils import paths
import pandas as pd
import warnings
from tqdm import tqdm
import datetime
import xarray as xr
import time
import numpy as np
from pyTMD import compute_tide_corrections
import os 

def date_to_fractional_years_resample3hrs(x):
    year = x.year
    month = x.month
    day = x.day
    hour = x.hour - x.hour%3
    day_in_year = x.timetuple().tm_yday

    # Determine if it's a leap year
    is_leap_year = (year % 4 == 0) and (year % 100 != 0 or year % 400 == 0)

    # Calculate the number of days in the current year
    days_in_year = 365 + is_leap_year

    # Calculate the fractional part of the year
    if is_leap_year==0:
        fractional_years = year + (day_in_year-1)/365 + (hour / (24 * days_in_year))
    else:
        if day_in_year > 60:
            fractional_years = year + (day_in_year-2)/365 + (hour / (24 * days_in_year))
        else:
            fractional_years = year + (day_in_year-1)/365 + (hour / (24 * days_in_year))

    return fractional_years


def GetCoordinates(location):
    '''
    location: name of location, in lowercase, e.g. 'aberdeen'.
    '''
    # Retrieve data from specified location in the UK
    # The metadata file is the paths.Haigh_meta_path
    df = pd.read_csv(paths.Haigh_meta_path)
    lat = df[df['location'] == location]['latitude'].values[0]
    lon = df[df['location'] == location]['longitude'].values[0]
    return lat, lon

def ExtractFloods(location):
    '''
    water_level_data: the water_level data
    location: name of location, in lowercase, e.g. 'aberdeen'.
    ---
    ! This function has warnings on being inneficient with pandas, but they are supressed.
    '''
    warnings.filterwarnings("ignore")

    water_level= pd.read_csv(paths.water_levels_path,skiprows = 2, header =3)
    water = water_level[water_level['Tide gauge site'] == location.capitalize()]
    hours_list = []
    month_list = []
    days_list = []
    years_list = []
    t_list = []

    for x in water['Date and time']:
        dt = datetime.datetime.strptime(x,'%d/%m/%Y %H:%M')
        hours_list.append(dt.hour)
        month_list.append(dt.month)
        days_list.append(dt.day)
        years_list.append(dt.year)
        t_list.append(date_to_fractional_years_resample3hrs(dt))
    water['hour'] = hours_list
    water['month'] = month_list
    water['day'] = days_list
    water['year'] = years_list
    water['t'] = t_list

    return water

def ExtractWindow(data,geo_coord_tuple,data_type = 'CMIP6'):
    """
    data_type (string): 'CMIP6' or 'ERA5'
    THIS FUNCTION FAILS FOR NEGATIVE LONGITUDES ---- FIX!
    """


    if data_type == 'CMIP6':

        lat_lw = geo_coord_tuple[0][0]
        lat_up = geo_coord_tuple[0][1]
        lon_lw = geo_coord_tuple[1][0]
        lon_up = geo_coord_tuple[1][1]

        data = data.sel(lat = slice(lat_lw,lat_up))

        ### Check if the window crosses the meridian:
        if ((lon_lw >180)&(lon_lw<360))&((lon_up>0)&(lon_up<180)):
            crop1 = data.sel(lon = slice(lon_lw,360))
            crop2 = data.sel(lon = slice(0,lon_up))
            data = xr.concat([crop1,crop2],dim = 'lon')
        else:
            data = data.sel(lon = slice(lon_lw,lon_up))
    elif data_type == 'ERA5':
        ### ERA5 data has longitude in [-180,180]
        lat_lw = geo_coord_tuple[0][1]
        lat_up = geo_coord_tuple[0][0]
        lon_lw = geo_coord_tuple[1][0]
        lon_up = geo_coord_tuple[1][1]

        data = data.sel(latitude = slice(lat_lw,lat_up))

        ### Check if the window crosses the meridian:
        if ((lon_lw >180)&(lon_lw<360))&((lon_up>0)&(lon_up<180)):
            crop1 = data.sel(longitude = slice(lon_lw-360,360))
            crop2 = data.sel(longitude = slice(0,lon_up))
            data = xr.concat([crop1,crop2],dim = 'longitude')
        elif ((lon_lw >180)&(lon_lw<360))&((lon_up>180)&(lon_up<360)):
            data = data.sel(longitude = slice(lon_lw-360,lon_up-360))
        else:
            data = data.sel(longitude = slice(lon_lw,lon_up))
    else:
        print('data_type not recognized.')

    return data

def compute_ref_mean(ref_ds):
    '''
    This function is used for NAO calculation.
    '''
    means = np.zeros(12)
    stds = np.zeros(12)
    for i in range(len(means)):
        month_ref = ref_ds.isel(time = ref_ds.time.dt.month == i+1)
        means[i] = np.mean(month_ref.sp)
        stds[i] = np.std(month_ref.sp)
    return means, stds

def geo_coord_borders_1GRID(lat, lon, Dataset_type):
    """
    Get the borders of the window of interest for CMIP6 Native grid;
    To be passed down into extract_window function.
    ---
    lat (np.float): latitude of location
    lon (np.float): longitude of location
    ---
    Returns:
        ((lat_lw_bord,lat_up,bord),(lon_lw_bord,lon_up_bord)): (tuple(tuple(np.float)))
    """
    ### The CMIP6 N96 Native grid has   144 gridcells of latitude, i.e. 1.25deg/cell, and
    ###                                 192 gridcells of longitude, i.e. 1.875deg/cell.

    ### The CMIP6 N216 Native grid has  324 gridcells of latitude, i.e. 0.5555deg/cell, and
    ###                                 432 gridcells of longitude, i.e. 0.8333deg/cell.

    ### The ERA5 Native grid has        721 (720) gridcells of latitude, i.e. 0.5555deg/cell, and
    ###                                 1440 gridcells of longitude, i.e. 0.8333deg/cell.

    ### Make sure the longitude is in [-180,180], not [0,360]:
    if (lon - 180 >0):
        lon = lon - 360
    if Dataset_type == 'GCM_N216':
        delta_lat = 0.56 * .5
        delta_lon = 0.8333 * .5
    elif Dataset_type == 'ERA5':
        delta_lat = 0.25 * .5
        delta_lon = 0.25 * .5


    lat_lw_bord = lat - delta_lat 
    lat_up_bord = lat + delta_lat

    lon_lw_bord = lon - delta_lon
    lon_up_bord = lon + delta_lon

    if ((lon_lw_bord >= 0)&(lon_up_bord > 0)):
        lon_lw_bord = lon - delta_lon
        lon_up_bord = lon + delta_lon
    elif ((lon_lw_bord < 0)&(lon_up_bord <= 0)):
        lon_lw_bord = lon - delta_lon + 360
        lon_up_bord = lon + delta_lon + 360
    elif ((lon_lw_bord < 0)&(lon_up_bord > 0)):
        lon_lw_bord = lon - delta_lon + 360
        lon_up_bord = lon + delta_lon
    else:
        print("Error: Something is wrong with the Longitude.")

    return ((lat_lw_bord,lat_up_bord),(lon_lw_bord,lon_up_bord))

def GetGeoBorders(lat,lon):
    lat_lw = lat - 3.125
    lat_up = lat + 3.125
    lon_lw = lon - 3.125
    lon_up = lon + 3.125
    return ((lat_lw,lat_up),(lon_lw,lon_up))

def Step1():
    # open datasets:
    start = time.time()
    ps_data = xr.open_dataset(paths.era5_79_18_ps)
    u_data = xr.open_dataset(paths.era5_79_18_u)
    v_data = xr.open_dataset(paths.era5_79_18_v)
    t_data = xr.open_dataset(paths.era5_79_18_t)
    pr_data = xr.open_dataset(paths.era5_79_18_pr)
    end = time.time()
    print("Time taken to open datasets: ", end-start, ' ; Step 1/13')
    return ps_data, u_data, v_data, t_data, pr_data

def Step2(ps_data, u_data, v_data, t_data, pr_data):
    start = time.time()
    # remove leap years:
    ps_data = ps_data.sel({'time': ~((ps_data['time'].dt.month == 2) & (ps_data['time'].dt.day == 29))})
    u_data = u_data.sel({'time': ~((u_data['time'].dt.month == 2) & (u_data['time'].dt.day == 29))})
    v_data = v_data.sel({'time': ~((v_data['time'].dt.month == 2) & (v_data['time'].dt.day == 29))})
    t_data = t_data.sel({'time': ~((t_data['time'].dt.month == 2) & (t_data['time'].dt.day == 29))})
    pr_data = pr_data.sel({'time': ~((pr_data['time'].dt.month == 2) & (pr_data['time'].dt.day == 29))})
    end = time.time()
    print("Time taken to remove leap years: ", end-start, ' ; Step 2/13')
    return ps_data, u_data, v_data, t_data, pr_data

def Step3(ps_data, u_data, v_data, t_data, pr_data, location):
    start = time.time()
    warnings.filterwarnings("ignore")
    lat, lon = GetCoordinates(location)
    geo_borders = GetGeoBorders(lat,lon)
    # Get the window around the location:
    ps_window = ExtractWindow(ps_data,geo_borders,data_type='ERA5')
    v_window = ExtractWindow(v_data,geo_borders,data_type='ERA5')
    u_window = ExtractWindow(u_data,geo_borders,data_type='ERA5')    
    t_window = ExtractWindow(t_data,geo_borders,data_type='ERA5')
    pr_window = ExtractWindow(pr_data,geo_borders,data_type='ERA5')
    end = time.time()
    print("Time taken to extract window: ", end-start, ' ; Step 3/13')
    return ps_window, u_window, v_window, t_window, pr_window

def Step4(ps_window, u_window, v_window, t_window, pr_window):
    start = time.time()
    # Get the dataframes:
    df_ps = ps_window.sp.to_dataframe().reset_index()
    df_u = u_window.u10.to_dataframe().reset_index()
    df_v = v_window.v10.to_dataframe().reset_index()
    df_t = t_window.t2m.to_dataframe().reset_index()
    df_pr = pr_window.tp.to_dataframe().reset_index()
    end = time.time()
    print("Time taken to convert to dataframe: ", end-start, ' ; Step 4/13')
    return df_ps, df_u, df_v, df_t, df_pr

def Step5(df_ps, df_u, df_v, df_t, df_pr):
    start = time.time()
    df_ps['t'] = df_ps.time.apply(lambda x: date_to_fractional_years_resample3hrs(x))
    df_u['t'] = df_ps['t']
    df_v['t'] = df_ps['t']
    df_t['t'] = df_ps['t']
    df_pr['t'] = df_ps['t']
    end = time.time()
    print("Time taken to convert to fractional years: ", end-start, ' ; Step 5/13')
    return df_ps, df_u, df_v, df_t, df_pr

def Step6(df_ps, df_u, df_v, df_t, df_pr):
    start = time.time()
    df_u.drop(columns=['time'],inplace=True)
    df_v.drop(columns=['time'],inplace=True)
    df_t.drop(columns=['time'],inplace=True)
    df_ps.drop(columns=['time'],inplace=True)
    df_pr.drop(columns=['time'],inplace=True)
    end = time.time()
    print("Time taken to drop time column: ", end-start, ' ; Step 6/13')
    return df_ps, df_u, df_v, df_t, df_pr

def Step7(df_ps, df_u, df_v, df_t, df_pr):
    start = time.time()
    df_merged = df_u.merge(df_v,on=['t','latitude','longitude'],how='inner')
    df_merged = df_merged.merge(df_t,on=['t','latitude','longitude'],how='inner')
    df_merged = df_merged.merge(df_ps,on=['t','latitude','longitude'],how='inner')
    df_merged = df_merged.merge(df_pr,on=['t','latitude','longitude'],how='inner')
    end = time.time()
    print("Time taken to merge dataframes: ", end-start, ' ; Step 7/13')
    return df_merged

def Step8(df_merged):
    start = time.time()
    lat_array= np.unique(df_merged['latitude'])
    lon_array= np.unique(df_merged['longitude'])
    lon_array[lon_array > 180] = lon_array[lon_array > 180] - 360
    lon_array.sort()

    for i in tqdm(range(25)):
        LAT = lat_array[i]
        for j in range(25):
            LON = lon_array[j]
            # if LON<0: LON = LON + 360

            A = df_merged[df_merged['latitude'] == LAT]
            B = A[A['longitude'] == LON]
            B = B.drop(columns = ['latitude','longitude'])#, inplace = True)
            B = B.rename(columns= {'u10':'u_'+str(i+1)+'_'+str(j+1),
                                'v10':'v_'+str(i+1)+'_'+str(j+1),
                                't2m':'T_'+str(i+1)+'_'+str(j+1),
                                'sp':'P_'+str(i+1)+'_'+str(j+1),
                                'tp':'pr_'+str(i+1)+'_'+str(j+1)})
            if i==0 and j == 0:
                final_df = B
            else:
                final_df = final_df.merge(B,on='t',how='inner')
    end = time.time()
    print("Time taken to create the grid-cell-wise features: ", end-start, ' ; Step 8/13')
    return final_df

def CalculateNAO():

    ps_data1 = xr.open_dataset(paths.era5_40_78_ps)
    ps_data2 = xr.open_dataset(paths.era5_79_18_ps)
    lat_reyk = 64.15
    lon_reyk = 22.8*-1 + 360
    lat_azor = 37.7
    lon_azor = 25.7*-1 + 360

    reyk_tuple = geo_coord_borders_1GRID(lat_reyk,lon_reyk,'ERA5')
    azor_tuple = geo_coord_borders_1GRID(lat_azor,lon_azor,'ERA5')

    pres_Reyk_1 = ExtractWindow(ps_data1, reyk_tuple,'ERA5')
    pres_Azor_1 = ExtractWindow(ps_data1, azor_tuple,'ERA5')

    pres_Reyk_2 = ExtractWindow(ps_data2, reyk_tuple,'ERA5')
    pres_Azor_2 = ExtractWindow(ps_data2, azor_tuple,'ERA5')
    del ps_data1, ps_data2
    pres_Azor = xr.concat([pres_Azor_1,pres_Azor_2],dim='time')
    pres_Reyk = xr.concat([pres_Reyk_1,pres_Reyk_2],dim='time')
    del pres_Azor_1, pres_Azor_2, pres_Reyk_1, pres_Reyk_2

    ref_rey = pres_Reyk
    ref_azo = pres_Azor

    ref_rey = ref_rey.isel(time = ref_rey.time.dt.year >= 1961)
    ref_rey = ref_rey.isel(time = ref_rey.time.dt.year <= 1990)

    ref_azo = ref_azo.isel(time = ref_azo.time.dt.year >= 1961)
    ref_azo = ref_azo.isel(time = ref_azo.time.dt.year <= 1990)

    means_rey, stds_rey = compute_ref_mean(ref_rey)
    means_azo, stds_azo = compute_ref_mean(ref_azo)

    azo_data_pd = pd.DataFrame({'ps': pres_Azor['sp'].isel(latitude =0).isel(longitude=0),'time': pres_Azor['time']})
    rey_data_pd = pd.DataFrame({'ps': pres_Reyk['sp'].isel(latitude =0).isel(longitude=0),'time': pres_Reyk['time']})

    azo_data_pd.ps = (azo_data_pd.ps - (means_azo[pres_Azor.time.dt.month-1]))/ stds_azo[pres_Azor.time.dt.month-1]
    rey_data_pd.ps = (rey_data_pd.ps - (means_rey[pres_Reyk.time.dt.month-1]))/ stds_rey[pres_Reyk.time.dt.month-1]

    NAO_ERA5 = pd.DataFrame({'nao':rey_data_pd.ps - azo_data_pd.ps, 'time': azo_data_pd.time})
    return NAO_ERA5

def Step9(form_df):
    start = time.time()
    ### calculate the vertical vorticity:
    for i in tqdm(range(24)):
        for j in range(24):
            ### dv/dx
            dv_dx = form_df['v_'+str(i+1)+'_'+str(j+2)] - form_df['v_'+str(i+1)+'_'+str(j+1)]     
            ### du/dy:
            du_dy = (form_df['u_'+str(i+2)+'_'+str(j+1)] - form_df['u_'+str(i+1)+'_'+str(j+1)]) * (-1)
            form_df['vort_'+str(i+1.5)+'_'+str(j+1.5)] = dv_dx - du_dy
    end = time.time()
    print('Time taken to calculate vertical vorticity: ', end-start, ' ; Step 9/13')
    return form_df

def Step10(form_df):
    start = time.time()
    ### add cumulative precipitation:
    for i in tqdm(range(25)):
        for j in range(25):
            my_data = form_df['pr_'+str(i+1)+'_'+str(j+1)]
            three_day_prec = list(np.zeros([23]))
            three_day_prec.extend(list(np.convolve(my_data,np.ones(24,dtype=int),'valid')))
            form_df['pr_cum3_'+str(i+1)+'_'+str(j+1)] = three_day_prec
            five_day_prec = list(np.zeros([39]))
            five_day_prec.extend(list(np.convolve(my_data,np.ones(40,dtype=int),'valid')))
            form_df['pr_cum5_'+str(i+1)+'_'+str(j+1)] = five_day_prec
    end = time.time()
    print('Time taken to calculate cumulative precipitation: ', end-start, ' ; Step 10/13')
    return form_df

def Step11(form_df):
    # Merge NAO data:
    start = time.time()
    NAO_file_exists = os.path.isfile('./NAO_era5.csv')
    # Add NAO:
    if NAO_file_exists == True:
        # Just add NAO:
        NAO_ERA5 = pd.read_csv('./NAO_era5.csv')
    else: 
        # Calc NAO from pressure files:
        NAO_ERA5 = CalculateNAO()
        fract_year = []
        for i in range(len(NAO_ERA5)):
            fract_year.append(date_to_fractional_years_resample3hrs(NAO_ERA5['time'][i]))
        NAO_ERA5['t'] = fract_year
        NAO_ERA5 = NAO_ERA5[~((NAO_ERA5['time'].dt.month == 2) & (NAO_ERA5['time'].dt.day == 29))]
        NAO_ERA5.to_csv('./NAO_era5.csv')
    
    form_df['t'] = np.round(form_df['t'], 5)
    NAO_ERA5.drop('Unnamed: 0', axis=1, inplace=True)
    NAO_ERA5['t'] = np.round(NAO_ERA5['t'], 5)

    form_df = form_df.merge(NAO_ERA5,on ='t', how='inner')
    end = time.time()
    print('Time taken to add NAO data: ', end-start, ' ; Step 11/13')
    return form_df

def GetTides(sample_times, location):
    lat, lon = GetCoordinates(location)
    
    if lon < 0: lon = lon + 360
    pd_range_time = pd.date_range(sample_times.iloc[0],sample_times.iloc[-1],freq='3H')
    point_df = pd.DataFrame({'lat': lat, 'lon':  lon, 'time': pd_range_time})
    # Add tides. Tides need to be calculated.
    print('Estimated time taken: ',.75*len(sample_times)/8/365,' mins.')

    out_drift = compute_tide_corrections(
        x=np.round(point_df.lon),
        y=np.round(point_df.lat),
        delta_time=point_df.time.values,
        DIRECTORY= paths.FES_tide_data,
        MODEL="FES2014",
        EPSG=4326,
        TYPE="drift",
        TIME="datetime",
        METHOD="bilinear",
    )
    point_df['tide'] = out_drift
    point_df
    point_df.to_csv('./tides_'+str(location)+'.csv')
    return
    
def Step12(form_df, location):
    start = time.time()
    sample_times = form_df['time']
    tides_file_name = './tides_'+str(location)+'.csv'
    tide_file_exists = os.path.isfile(tides_file_name)    
    if tide_file_exists:
        tides = pd.read_csv(tides_file_name)
        tides['time_ok'] = pd.to_datetime(tides['time'], format = '%Y-%m-%d %H:%M:%S')
        tides = tides[~((tides.time_ok.dt.month == 2) & (tides.time_ok.dt.day == 29))]
        if len(tides) == len(sample_times):
            form_df['tide'] = tides['tide']
        else:
            GetTides(sample_times,location)
            tides = pd.read_csv(tides_file_name)
            tides['time_ok'] = pd.to_datetime(tides['time'], format = '%Y-%m-%d %H:%M:%S')
            tides = tides[~((tides.time_ok.dt.month == 2) & (tides.time_ok.dt.day == 29))]
            form_df['tide'] = tides['tide']

    else:
        GetTides(sample_times,location)
        tides = pd.read_csv(tides_file_name)
        tides['time_ok'] = pd.to_datetime(tides['time'], format = '%Y-%m-%d %H:%M:%S')
        tides = tides[~((tides.time_ok.dt.month == 2) & (tides.time_ok.dt.day == 29))]
        form_df['tide'] = tides['tide']

    
    end = time.time()
    print('Time taken to add tides: ', end-start, ' ; Step 12/13')
    return form_df

def GetFloodsFromLocation(location):
    uk_haigh_meta = pd.read_csv(paths.Haigh_meta_path).drop(['Unnamed: 0','index'],axis=1)
    water_level= pd.read_csv(paths.water_levels_path,skiprows = 2, header =3)
    location_flood = uk_haigh_meta[uk_haigh_meta['location'] == location]['location_surgewatch'][0]

    ##### ADD FLOODS DATA TO THE MASSIVE DATAFRAME:
    water_at_location = water_level[water_level['Tide gauge site'] == location_flood]
    ### get the months and days of the storms:

    t_list = []
    for x in water_at_location['Date and time']:
        dt = datetime.datetime.strptime(x,'%d/%m/%Y %H:%M')
        t_list.append(date_to_fractional_years_resample3hrs(dt))
    water_at_location['t'] = t_list
    return water_at_location

def Step13(final_df, location):
    start = time.time()
    water_at_location = GetFloodsFromLocation(location)

    Yx1 = np.zeros([len(final_df)])
    Yx4 = np.zeros([len(final_df)])

    for i in range(len(water_at_location)):
        index_Flood = final_df[final_df['t'] == water_at_location['t'].iloc[i]].index.values
        Yx4[index_Flood-2] = 1.
        Yx4[index_Flood-1] = 1.
        Yx4[index_Flood] = 1.
        Yx4[index_Flood+1] = 1.
        Yx1[index_Flood] = 1.

    final_df['floods'] = Yx1
    final_df['floods_x4'] = Yx4
    end = time.time()
    print('Time taken to add floods: ', end-start, 'Step 13/13')
    return final_df
