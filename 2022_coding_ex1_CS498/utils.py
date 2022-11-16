import numpy as np
import math

def quaternion_from_euler(ai, aj, ak): 
    '''
    Arguments:
        ai, aj, ak: Euler angles in RPY order (Roll, Pitch and Yaw) (float)
    Returns:
        q: The corresponding quaternion in format [qx, qy, qz, qw] (python list)
    '''
    q = None # your code here

    return q # 


def lonlat2xyz(lat, lon, lat0, lon0): 
    # WGS84 ellipsoid constants:
    a = 6378137
    b = 6356752.3142
    e = math.sqrt(1-b**2/a**2)
    
    x = a*math.cos(math.radians(lat0))*math.radians(lon-lon0)/math.pow(1-e**2*(math.sin(math.radians(lat0)))**2,0.5)
    y = a*(1 - e**2)*math.radians(lat-lat0)/math.pow(1-e**2*(math.sin(math.radians(lat0)))**2,1.5)
    
    return x, y # x and y coordinates in a reference frame with the origin in lat0, lon0
