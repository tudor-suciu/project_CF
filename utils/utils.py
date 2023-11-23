
from utils import paths
import pandas as pd
import warnings
from tqdm import tqdm
import datetime
import xarray as xr

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

def GetGeoBorders(lat,lon):
    lat_lw = lat - 3.125
    lat_up = lat + 3.125
    lon_lw = lon - 3.125
    lon_up = lon + 3.125
    return ((lat_lw,lat_up),(lon_lw,lon_up))