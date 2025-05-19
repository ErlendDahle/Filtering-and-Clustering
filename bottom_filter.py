

import numpy as np
import xarray as xr

def scale_channel_1(data: xr.DataArray) -> xr.DataArray:

    """
    Scale channel 1 values to correct values.
    Parameters:
        data (xr.DataArray): The input DataArray.

    Returns:
        xr.Dataset: The modified data with correct channel 1 values.
    """

    downsampled_channel_1 = data[1,:,::2]
    data[1,:,:downsampled_channel_1.shape[1]]=downsampled_channel_1.values   
    data[1,:, downsampled_channel_1.shape[1]:]=np.nan


    return data


def scale_channel_1_up(data: xr.DataArray) -> xr.DataArray:

    """
    Scale channel 1 values to correct values by scaling up.
    Parameters:
        data (xr.DataArray): The input DataArray.

    Returns:
        xr.Dataset: The modified data with correct channel 1 values.
    """

    upsampled_channel_1 = np.repeat(data[1, :, :], 2, axis=1)
    data[1, :, :upsampled_channel_1.shape[1]] = upsampled_channel_1[:, :data.shape[2]]
    data[1, :, upsampled_channel_1.shape[1]:] = np.nan


    return data




def find_bottom(cutoff, data: xr.DataArray, dim: str = "range", start=50) -> xr.DataArray: #not really in use

    """
    Finds strong signals and removes everything else. Basically a theshold filter. 

    Parameters:
        data (xr.DataArray): The input DataArray.
        dim (str): The dimension along which to find the edge. Defaults to "range".

    Returns:
        xr.Dataset: The modified data with indices set to Nan if under bottom
    """
    Minimum_strength_1 =-10

    for i in range(start,cutoff):
        if data[0].isel({dim: i}).max() > Minimum_strength_1: 
            start = i
            break

    print("start: ", start)
    num_pings = data[0].sizes.get("ping_time",0)

    Minimum_strength =-30

    i, j = 0, 0
    for k in range(0,3):
        print("k: ", k) # channels
        for j in range(num_pings):# ping time     #data[k].sizes.get("ping_time",0)
            #print( j)
            for i in range(start, cutoff): # range
                if int(data[k].isel({dim: i})[{"ping_time": j}].sum()) > Minimum_strength:
    
                    data[k][{dim: slice(0, i-10 ), "ping_time": j}] = np.nan #Remove everthing over bottom
                    data[k][{dim: slice( i+50,cutoff ), "ping_time": j}] = np.nan
                    break
                    #data[k][{dim: slice( i+20,cutoff ), "ping_time": j}] = np.nan #Remove everthing under the last bottom
             

    return data


def remove_bottom(cutoff, data: xr.DataArray, skip_channel=[]) -> xr.DataArray:

    """
    Finds strong signals (bottom) and removes everything under. 

    Parameters:
        data (xr.DataArray): The input DataArray.
        dim (str): The dimension along which to find the edge. Defaults to "range".

    Returns:
        xr.Dataset: The modified data with indices set to Nan if under bottom
    """

    Minimum_strength_1 =-10
    start=400
    for i in range(start,cutoff):
        if data[0].isel({"range": i}).max() > Minimum_strength_1: 
            start = i
            break

    print("start: ", start) 

    num_channels = 3  # Number of channels
    num_pings = data[0].sizes.get("ping_time",0)
    Minimum_strength =-30
    cutoff_range = cutoff
    first_channel = True
    i, j = 0, 0
    for k in range(num_channels): # channels
        if k in skip_channel: # Skip channel
            continue
        for j in range(num_pings):# ping time    
            for i in range(start, cutoff_range): # range
                if int(data[k].isel({"range": i})[{"ping_time": j}].sum()) > Minimum_strength:
                    data[k][{"range": slice( i-40,cutoff ), "ping_time": j}] = np.nan
                    if first_channel:
                        for l in skip_channel:
                            data[l][{"range": slice( i-40,cutoff ), "ping_time": j}] = np.nan 
                    break        
        first_channel = False
    return data