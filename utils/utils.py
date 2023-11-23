
import paths
import pandas as pd

def GetCoordinates(location):
    '''
    location: name of location, in lowercase, e.g. 'aberdeen'.
    '''
    # Retrieve data from specified location in the UK
    # The metadata file is the paths.Haigh_meta_path
    df = pd.read_csv(paths.Haigh_meta_path)
    lat = df[df['location'] == location].latitude[0]
    lon = df[df['location'] == location].longitude[0]
    return lat, lon


