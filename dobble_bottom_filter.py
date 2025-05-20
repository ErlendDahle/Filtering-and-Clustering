

import numpy as np
import xarray as xr
import filter_utilities as fu
from scipy.ndimage import binary_dilation


def find_false_bottom(cutoff,data: xr.DataArray, dim: str = "range",first_threshold=-50 ) -> xr.DataArray:

    """
    Finds strong signals and removes everything else. 

    Parameters:
        data (xr.DataArray): The input DataArray.
        dim (str): The dimension along which to find the edge. Defaults to "range".
        first_threshold (float): The threshold a one ping time needs for a range to start search for a signal. Defaults to -50.

    Returns:
        xr.Dataset: The modified data with indices set to Nan if under bottom
    """

    start = 100
    for i in range(start,cutoff):
        if data.isel({dim: i}).max() > first_threshold: 
            start = i
            break


  
    num_pings = data[0].sizes.get("ping_time",0)
    i, j = 0, 0
    #print("start: ", start)
    for k in range(3):
        
        ping_time_data = data[k].isel(ping_time=0)
        threshold = ping_time_data.mean().values *0.7
    
        for j in range(num_pings):# ping time     
            last_bottom= 0 
            for i in range(start, cutoff): # range
                if int(data[k].isel({dim: i})[{"ping_time": j}].sum()) > threshold:
                    data[k][{dim: slice(last_bottom, i -20), "ping_time": j}] = np.nan #Remove everthing over bottom
                    i+=20
                    last_bottom = i
            #if last_bottom != 0:
            data[k][{dim: slice( last_bottom,cutoff), "ping_time": j}] = np.nan #Remove everthing under the last bottom
             
    
    return data





def create_sonar_line( data: xr.Dataset,cutoff):

    """
    Compare Bathymetry and sonar bottom data.

    Parameters:
        data (xr.DataArray): The input DataArray.
        cutoff (int): The cutoff value.


    Returns:
        Numpy Array: The first non nan values of the sonar data.
    """


    num_channels = 3  # Number of channels
    num_pings = data[0].sizes.get("ping_time",0)
    first_non_nan_values = np.full((num_channels,num_pings), np.nan)

    for k in range(num_channels): #Channels
        for i in range(num_pings):  # Ping time
            ping_data = data["Sv"][k, i, :cutoff]
            ping_range = data["range"].values
            non_nan_indices = ~np.isnan(ping_data)
            if non_nan_indices.any() == True:
                first_non_nan_values[k,i] = - ping_range[np.where(non_nan_indices)[0][0]]
    
    return first_non_nan_values



def create_sonar_lines(data: xr.DataArray,cutoff):
    cluster_ranges=[]
    num_channels = 3  # Number of channels

    #Indentify clusters
    for channel in range(num_channels): #Channels
        start=None
        count=0
        valid_count=0
        for i in range(cutoff): 
            valid_values = np.sum(~np.isnan(data[channel, 0:20, i])) >= 10
            if valid_values:
                count=0
                valid_count+=1
                if start is None:
                    start=i
            elif start is not None:
                count+=1
                if count>10 and valid_count>10:
                    cluster_ranges.append([channel,start,i])
                    start=None
                    count=0 
                    valid_count=0


    # Count clusters per channel
    clusters_count = np.zeros(num_channels, dtype=int)
    for channel, _, _ in cluster_ranges:
        clusters_count[channel] += 1

    print("Clusters per channel: ", clusters_count)
    num_pings = data[0].sizes.get("ping_time",0)
    cluster_lines = np.full((len(cluster_ranges),num_pings), np.nan)

    # Extract cluster lines
    for channel in range(num_channels): #Channels
        start = 0
        offset = sum(clusters_count[:channel])

        for j in range(clusters_count[channel]):
            current_cluster = offset + j
            if j==int(clusters_count[channel])-1:
                cutoff_range = cutoff
            else:
                next_cluster_start = cluster_ranges[current_cluster + 1][1] 
                cutoff_range = cluster_ranges[current_cluster][2] + (next_cluster_start - cluster_ranges[current_cluster][2]) / 2

        

            for i in range(num_pings):
                ping_data = data[channel, i, int(start):int(cutoff_range)]
                ping_range = data["range"].values
                non_nan_indices = np.where(~np.isnan(ping_data))[0]

                if non_nan_indices.size > 0:
                    cluster_lines[current_cluster, i] = -(ping_range[non_nan_indices[0]] + ping_range[int(start)])

            start = cutoff_range


    return cluster_lines, cluster_ranges


def delete_clusters_with_high_nan_ratio(cluster_lines, cluster_ranges):
    to_delete = []
    for i in range(len(cluster_ranges)):
        nan_ratio = np.isnan(cluster_lines[i]).sum() / cluster_lines.shape[1]
        
        if nan_ratio > 0.5:
            to_delete.append(i)
    cluster_lines = np.delete(cluster_lines, to_delete, axis=0)
    cluster_ranges = np.delete(cluster_ranges, to_delete, axis=0)

    return cluster_lines, cluster_ranges


def cutoff_mask_ranges(mask, cluster_ranges):
    for i in range(len(cluster_ranges)):
        mask[cluster_ranges[i][0], :, :(cluster_ranges[i][1]-180)] = False
        mask[cluster_ranges[i][0], :, (cluster_ranges[i][2]+100):] = False

    if (cluster_ranges[:,0]==0).any()==False:
        mask[0, :, :] = False
    if (cluster_ranges[:,0]==1).any()==False:
        mask[1, :, :] = False       
    if (cluster_ranges[:,0]==2).any()==False:
        mask[2, :, :] = False   

    return mask



def delete_high_mse_clusters(cluster_lines, cluster_ranges, Bathymetry):
    to_delete = []

    for i in range(len(cluster_ranges)):
        cluster_lines[i] = fu.interpolate_nan(cluster_lines[i])
        mse = fu.compare_signals(cluster_lines[i] - np.mean(cluster_lines[i]), Bathymetry - np.mean(Bathymetry))
        print("MSE: ", mse)
        if mse > 100:
            to_delete.append(i)
   

    cluster_lines = np.delete(cluster_lines, to_delete, axis=0)
    cluster_ranges = np.delete(cluster_ranges, to_delete, axis=0)

    return cluster_lines, cluster_ranges



def remove_false_seabottom(data: xr.DataArray, mask_data, cluster_ranges ) -> xr.DataArray:

    mask = ~np.isnan(mask_data)

    for i in range(len(cluster_ranges)):
        mask[cluster_ranges[i][0], :, :(cluster_ranges[i][1])] = False
        mask[cluster_ranges[i][0], :, (cluster_ranges[i][2]):] = False

    if (cluster_ranges[:,0]==0).any()==False:
        mask[0, :, :] = False
    if (cluster_ranges[:,0]==1).any()==False:
        mask[1, :, :] = False       
    if (cluster_ranges[:,0]==2).any()==False:
        mask[2, :, :] = False   


    dilated_mask = binary_dilation(mask, iterations=5)


    data= xr.where(dilated_mask, np.nan, data)

    return data


def remove_false_bottom(data: xr.DataArray, mask_data, cluster_ranges, iterations) -> xr.DataArray:

    mask = ~np.isnan(mask_data)

    for i in range(len(cluster_ranges)):
        mask[cluster_ranges[i][0], :, :(cluster_ranges[i][1])] = False
        mask[cluster_ranges[i][0], :, (cluster_ranges[i][2]):] = False

    if (cluster_ranges[:,0]==0).any()==False:
        mask[0, :, :] = False
    if (cluster_ranges[:,0]==1).any()==False:
        mask[1, :, :] = False       
    if (cluster_ranges[:,0]==2).any()==False:
        mask[2, :, :] = False   


    dilated_mask = binary_dilation(mask, iterations)

    data[0,:,:]= xr.where(dilated_mask[0,:,:], np.nan, data[0,:,:])
    data[1,:,:]= xr.where(dilated_mask[1,:,:], np.nan, data[1,:,:])
    data[2,:,:]= xr.where(dilated_mask[2,:,:], np.nan, data[2,:,:])
    
    return data

