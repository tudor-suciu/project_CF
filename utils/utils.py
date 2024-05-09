
from utils import paths
import pandas as pd
import warnings
from tqdm import tqdm
import datetime
import xarray as xr
import time
import numpy as np
import pyTMD
import os 
from shapely.geometry import Point
import geopandas as gpd

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
#--------------------------------------------------------------------------------------------------

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
#--------------------------------------------------------------------------------------------------

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
#--------------------------------------------------------------------------------------------------

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
#--------------------------------------------------------------------------------------------------

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
#--------------------------------------------------------------------------------------------------

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
#--------------------------------------------------------------------------------------------------

def GetGeoBorders(lat,lon):
    lat_lw = lat - 3.125
    lat_up = lat + 3.125
    lon_lw = lon - 3.125
    lon_up = lon + 3.125
    return ((lat_lw,lat_up),(lon_lw,lon_up))
#--------------------------------------------------------------------------------------------------

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
#--------------------------------------------------------------------------------------------------

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
#--------------------------------------------------------------------------------------------------

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
#--------------------------------------------------------------------------------------------------

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
#--------------------------------------------------------------------------------------------------

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
#--------------------------------------------------------------------------------------------------

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
#--------------------------------------------------------------------------------------------------

def Step7(df_ps, df_u, df_v, df_t, df_pr):
    start = time.time()
    df_merged = df_u.merge(df_v,on=['t','latitude','longitude'],how='inner')
    df_merged = df_merged.merge(df_t,on=['t','latitude','longitude'],how='inner')
    df_merged = df_merged.merge(df_ps,on=['t','latitude','longitude'],how='inner')
    df_merged = df_merged.merge(df_pr,on=['t','latitude','longitude'],how='inner')
    end = time.time()
    print("Time taken to merge dataframes: ", end-start, ' ; Step 7/13')
    return df_merged
#--------------------------------------------------------------------------------------------------

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
#--------------------------------------------------------------------------------------------------

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
#--------------------------------------------------------------------------------------------------

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
#--------------------------------------------------------------------------------------------------

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
#--------------------------------------------------------------------------------------------------

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
#--------------------------------------------------------------------------------------------------

def GetTides(sample_times, location):
    lat, lon = GetCoordinates(location)
    
    if lon < 0: lon = lon + 360
    pd_range_time = pd.date_range(sample_times.iloc[0],sample_times.iloc[-1],freq='3H')
    point_df = pd.DataFrame({'lat': lat, 'lon':  lon, 'time': pd_range_time})
    # Add tides. Tides need to be calculated.
    print('Estimated time taken: ',.75*len(sample_times)/8/365,' mins.')

    out_drift = pyTMD.compute_tide_corrections(
        x=point_df.lon.iloc[0],
        y=point_df.lat.iloc[0],
        delta_time=point_df.time.values,
        DIRECTORY= paths.FES_tide_data,
        MODEL="FES2014",
        EPSG=4326,
        TYPE="drift",
        TIME="datetime",
        METHOD="bilinear",
        EXTRAPOLATE=True,
    )
    point_df['tide'] = out_drift
    # point_df
    time.sleep(3)
    point_df.to_csv('./tides_'+str(location)+'.csv')
    return
#--------------------------------------------------------------------------------------------------
    
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
            tides.drop(columns = ['lat','lon'], inplace = True)
            form_df = pd.merge(form_df, tides, on = 'time', how = 'inner')

        else:
            GetTides(sample_times,location)
            tides = pd.read_csv(tides_file_name)
            tides['time_ok'] = pd.to_datetime(tides['time'], format = '%Y-%m-%d %H:%M:%S')
            tides = tides[~((tides.time_ok.dt.month == 2) & (tides.time_ok.dt.day == 29))]
            tides.drop(columns = ['lat','lon'], inplace = True)
            form_df = pd.merge(form_df, tides, on = 'time', how = 'inner')

    else:
        GetTides(sample_times,location)
        tides = pd.read_csv(tides_file_name)
        tides['time_ok'] = pd.to_datetime(tides['time'], format = '%Y-%m-%d %H:%M:%S')
        tides = tides[~((tides.time_ok.dt.month == 2) & (tides.time_ok.dt.day == 29))]
        tides.drop(columns = ['lat','lon'], inplace = True)
        form_df = pd.merge(form_df, tides, on = 'time', how = 'inner')

    
    end = time.time()
    print('Time taken to add tides: ', end-start, ' ; Step 12/13')
    return form_df
#--------------------------------------------------------------------------------------------------

def GetFloodsFromLocation(location):
    uk_haigh_meta = pd.read_csv(paths.Haigh_meta_path).drop(['Unnamed: 0','index'],axis=1)
    water_level= pd.read_csv(paths.water_levels_path,skiprows = 2, header =3)
    location_flood = uk_haigh_meta[uk_haigh_meta['location'] == location]['location_surgewatch'].iloc[0]

    ##### ADD FLOODS DATA TO THE MASSIVE DATAFRAME:
    water_at_location = water_level[water_level['Tide gauge site'] == location_flood]

    t_list = []
    for x in water_at_location['Date and time']:
        dt = datetime.datetime.strptime(x,'%d/%m/%Y %H:%M')
        t_list.append(date_to_fractional_years_resample3hrs(dt))
    water_at_location['t'] = t_list
    water_at_location['t'] = np.round(water_at_location['t'],5)
    return water_at_location
#--------------------------------------------------------------------------------------------------

def Step13(final_df, location):
    start = time.time()
    water_at_location = GetFloodsFromLocation(location)
    water_at_location['t'] = np.round(water_at_location['t'],5)

    Yx1 = np.zeros([len(final_df)])
    Yx4 = np.zeros([len(final_df)])

    for i in range(len(water_at_location)):
        to_find = water_at_location['t'].iloc[i]
        index_Flood = final_df[final_df['t'] == to_find].index.values

        if index_Flood:
            index_Flood= index_Flood[0]
        else:
            continue

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
#--------------------------------------------------------------------------------------------------

def determine_location(latitude, longitude):
    point = Point(longitude, latitude)
    
    
    world = gpd.read_file("/Users/tudor/Documents/phd/data/ne_10m_ocean/ne_10m_ocean.shp")  # Replace with the path to the downloaded file
    for geom in world["geometry"]:
        if point.within(geom):
            return False
    
    return True
#--------------------------------------------------------------------------------------------------

def Add_SquareCube_cols(df):
    '''
    This function adds the square and cube of the u, v, 
    and the square for the P and T columns.
    ---
    Input:
        df (pd.DataFrame): the dataframe
    Output:
        df (pd.DataFrame): the dataframe with the new columns
    '''
    for i in range(25):
        for j in range(25):
            u_Nonsquare_col = 'u_' + str(i+1) + '_' + str(j+1)
            u_square_col = 'u2_' + str(i+1) + '_' + str(j+1)
            v_Nonsquare_col = 'v_' + str(i+1) + '_' + str(j+1)
            v_square_col = 'v2_' + str(i+1) + '_' + str(j+1)
            u_cube_col = 'u3_' + str(i+1) + '_' + str(j+1)
            v_cube_col = 'v3_' + str(i+1) + '_' + str(j+1)

            df[u_square_col] = df[u_Nonsquare_col]**2
            df[v_square_col] = df[v_Nonsquare_col]**2
            df[u_cube_col] = df[u_Nonsquare_col]**3
            df[v_cube_col] = df[v_Nonsquare_col]**3

            p_Nonsquare_col = 'P_' + str(i+1) + '_' + str(j+1)
            p_square_col = 'P2_' + str(i+1) + '_' + str(j+1)
            t_Nonsquare_col = 'T_' + str(i+1) + '_' + str(j+1)
            t_square_col = 'T2_' + str(i+1) + '_' + str(j+1)

            df[p_square_col] = df[p_Nonsquare_col]**2
            df[t_square_col] = df[t_Nonsquare_col]**2
    return df
#--------------------------------------------------------------------------------------------------

def Prepare_PhysicalVar_Dfs(df):
    '''
    Creates dataframes for each physical variable, with the columns being the individual cells;
    This is used to then get statistics for each phyiscal variable
    for the whole area/ sea+land areas.
    ---
    Input:
        df (pd.DataFrame): dataframe
    Output:
        pressure, pressure2, temperature, temperature2 (pd.DataFrame)
        u_wind, v_wind, u2_wind, v2_wind, u3_wind, v3_wind (pd.DataFrame)
        precipitation, pr_cum3, pr_cum5, vort, air_density (pd.DataFrame)
    '''
    all_cells = []
    for x in np.arange(1,26):
        for y in np.arange(1,26):
            all_cells.append((x,y))

    vort_cells = []
    for x in np.arange(1,25):
        for y in np.arange(1,25):
            vort_cells.append((x+.5,y+.5))

    pressure_list = []
    pressure2_list = []
    temperature_list = []
    temperature2_list = []
    u_wind_list = []
    v_wind_list = []
    u2_wind_list = []
    v2_wind_list = []
    u3_wind_list = []
    v3_wind_list = []
    precipitation_list = []
    pr_cum3_list = []
    pr_cum5_list = []
    for x in all_cells:
        pressure_list.append('P_'+str(x[0])+'_'+str(x[1]))
        pressure2_list.append('P2_'+str(x[0])+'_'+str(x[1]))
        temperature_list.append('T_'+str(x[0])+'_'+str(x[1]))
        temperature2_list.append('T2_'+str(x[0])+'_'+str(x[1]))
        u_wind_list.append('u_'+str(x[0])+'_'+str(x[1]))
        v_wind_list.append('v_'+str(x[0])+'_'+str(x[1]))
        u2_wind_list.append('u2_'+str(x[0])+'_'+str(x[1]))
        v2_wind_list.append('v2_'+str(x[0])+'_'+str(x[1]))
        u3_wind_list.append('u3_'+str(x[0])+'_'+str(x[1]))
        v3_wind_list.append('v3_'+str(x[0])+'_'+str(x[1]))
        precipitation_list.append('pr_'+str(x[0])+'_'+str(x[1]))
        pr_cum3_list.append('pr_cum3_'+str(x[0])+'_'+str(x[1]))
        pr_cum5_list.append('pr_cum5_'+str(x[0])+'_'+str(x[1]))


    pressure = df[pressure_list]
    pressure2 = df[pressure2_list]
    temperature = df[temperature_list]
    temperature2 = df[temperature2_list]
    u_wind = df[u_wind_list]
    v_wind = df[v_wind_list]
    u2_wind = df[u2_wind_list]
    v2_wind = df[v2_wind_list]
    u3_wind = df[u3_wind_list]
    v3_wind = df[v3_wind_list]
    precipitation = df[precipitation_list]
    pr_cum3 = df[pr_cum3_list]
    pr_cum5 = df[pr_cum5_list]

    vort_list = []
    for x in vort_cells:
        vort_list.append('vort_'+str(x[0])+'_'+str(x[1]))

    vort = df[vort_list]


    air_density = pressure.copy()
    for i in range(air_density.shape[1]):
        air_density.iloc[:,i] = temperature.iloc[:,i]/pressure.iloc[:,i]
    
    dfs_dict = {
        'pressure': pressure,
        'pressure2': pressure2,
        'temperature': temperature,
        'temperature2': temperature2,
        'u_wind': u_wind,
        'v_wind': v_wind,
        'u2_wind': u2_wind,
        'v2_wind': v2_wind,
        'u3_wind': u3_wind,
        'v3_wind': v3_wind,
        'precipitation': precipitation,
        'pr_cum3': pr_cum3,
        'pr_cum5': pr_cum5,
        'vort': vort,
        'air_density': air_density
    }
    return dfs_dict
#--------------------------------------------------------------------------------------------------


def Make_PhyVars_Stats_Choices(df,phys_dict, choice_1d_dict, phys_choice_dict, stat_dict):
    '''
    Function that makes the choices of physical variables and statistics to be included 
    in the new dataframe, that is subject to further analysis.
    ---
    Inputs:
        df (pd.DataFrame): dataframe
        phys_dict (dict): dictionary of physical variables dataframes
        choice_1d_dict (dict): dictionary of 1d choice variables
        phys_choice_dict (dict): dictionary of physical variables choices
        stat_dict (dict): dictionary of statistics choices
    Outputs:
        df_new (pd.DataFrame): dataframe with chosen physical variables and statistics
    '''
    df_new = df[[key for key, value in choice_1d_dict.items() if value == 1]]
    chosen_phys = [key for key, value in phys_choice_dict.items() if value == 1]
    chosen_stat = [key for key, value in stat_dict.items() if value == 1]

    for phys in chosen_phys:
        for stat in chosen_stat:
            if stat == 'mean':
                df_new[phys + '_' + stat] = phys_dict[phys].mean(axis = 1)
            elif stat == 'min':
                df_new[phys + '_' + stat] = phys_dict[phys].min(axis = 1)
            elif stat == 'max':
                df_new[phys + '_' + stat] = phys_dict[phys].max(axis = 1)
            elif stat == 'std':
                df_new[phys + '_' + stat] = phys_dict[phys].std(axis = 1)
            elif stat == 'skew':
                df_new[phys + '_' + stat] = phys_dict[phys].skew(axis = 1)
            elif stat == 'kurtosis':
                df_new[phys + '_' + stat] = phys_dict[phys].kurtosis(axis = 1)
            elif stat == '1st_perc':
                df_new[phys + '_' + stat] = phys_dict[phys].quantile(0.01, axis = 1)
            elif stat == '99th_perc':
                df_new[phys + '_' + stat] = phys_dict[phys].quantile(0.99, axis = 1)
            elif stat == '10th_perc':
                df_new[phys + '_' + stat] = phys_dict[phys].quantile(0.1, axis = 1)
            elif stat == '90th_perc':
                df_new[phys + '_' + stat] = phys_dict[phys].quantile(0.9, axis = 1)
            else:
                print('Wrong stat name -- ERROR.')

    return df_new

#--------------------------------------------------------------------------------------------------


def GetChoiceDicts(choice_str_1,choice_str_2,choice_str_3,choice_str_4,choice_str_5,choice_str_6):
    '''
    '''
    choice_1d_dict = {
        'flood': int(str(choice_str_1)[1]),
        'tide' : int(str(choice_str_1)[2]),
        'nao' : int(str(choice_str_1)[3])
    }
    phys_choice_dict = {
        'pressure': int(str(choice_str_2)[1]),
        'pressure2': int(str(choice_str_2)[2]),
        'temperature': int(str(choice_str_2)[3]),
        'temperature2': int(str(choice_str_2)[4]),
        'u_wind': int(str(choice_str_3)[1]),
        'v_wind': int(str(choice_str_3)[2]),
        'u2_wind': int(str(choice_str_3)[3]),
        'v2_wind': int(str(choice_str_3)[4]),
        'u3_wind': int(str(choice_str_3)[5]),
        'v3_wind': int(str(choice_str_3)[6]),
        'precipitation': int(str(choice_str_4)[1]),
        'pr_cum3': int(str(choice_str_4)[2]),
        'pr_cum5': int(str(choice_str_4)[3]),
        'vort': int(str(choice_str_4)[4]),
        'air_density': int(str(choice_str_4)[5])
    }
    stat_dict = {
        'mean': int(str(choice_str_5)[1]),
        'std': int(str(choice_str_5)[2]),
        'min': int(str(choice_str_5)[3]),
        'max': int(str(choice_str_5)[4]),
        'skew': int(str(choice_str_5)[5]),
        'kurtosis': int(str(choice_str_5)[6]),
        '1st_perc': int(str(choice_str_6)[1]),
        '99th_perc': int(str(choice_str_6)[2]),
        '10th_perc': int(str(choice_str_6)[3]),
        '90th_perc': int(str(choice_str_6)[4])
    }

    return choice_1d_dict, phys_choice_dict, stat_dict
#--------------------------------------------------------------------------------------------------

def GetString_PhysAndStats(choice_str_1,choice_str_2,choice_str_3,choice_str_4,choice_str_5,choice_str_6):
    str1 = 'Phys: '
    if str(choice_str_1)[1] == '1':
        str1 += 'flood; '
    if str(choice_str_1)[2] == '1':
        str1 += 'tide; '
    if str(choice_str_1)[3] == '1':
        str1 += 'nao; '
    
    if str(choice_str_2)[1] == '1':
        str1 += 'P; '
    if str(choice_str_2)[2] == '1':
        str1 += 'P2; '
    if str(choice_str_2)[3] == '1':
        str1 += 'T; '
    if str(choice_str_2)[4] == '1':
        str1 += 'T2; '

    if str(choice_str_3)[1] == '1':
        str1 += 'u; '
    if str(choice_str_3)[2] == '1':
        str1 += 'v; '
    if str(choice_str_3)[3] == '1':
        str1 += 'u2; '
    if str(choice_str_3)[4] == '1':
        str1 += 'v2; '
    if str(choice_str_3)[5] == '1':
        str1 += 'u3; '
    if str(choice_str_3)[6] == '1':
        str1 += 'v3; '

    if str(choice_str_4)[1] == '1':
        str1 += 'pr; '
    if str(choice_str_4)[2] == '1':
        str1 += 'pr_cum3; '
    if str(choice_str_4)[3] == '1':
        str1 += 'pr_cum5; '
    if str(choice_str_4)[4] == '1':
        str1 += 'vort; '
    if str(choice_str_4)[5] == '1':
        str1 += 'air_density; '

    str2 = 'Stats: '

    if str(choice_str_5)[1] == '1':
        str2 += 'mean; '
    if str(choice_str_5)[2] == '1':
        str2 += 'std; '
    if str(choice_str_5)[3] == '1':
        str2 += 'min; '
    if str(choice_str_5)[4] == '1':
        str2 += 'max; '
    if str(choice_str_5)[5] == '1':
        str2 += 'skew; '
    if str(choice_str_5)[6] == '1':
        str2 += 'kurtosis; '

    if str(choice_str_6)[1] == '1':
        str2 += '1st_perc; '
    if str(choice_str_6)[2] == '1':
        str2 += '99th_perc; '
    if str(choice_str_6)[3] == '1':
        str2 += '10th_perc; '
    if str(choice_str_6)[4] == '1':
        str2 += '90th_perc; '
    
    return str1, str2