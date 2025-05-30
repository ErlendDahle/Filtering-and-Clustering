
import numpy as np
import xarray as xr
from sklearn.cluster import DBSCAN


def clustering(data: xr.Dataset,min_range,max_range) -> xr.Dataset:
    """
    Perform clustering on the data using DBSCAN.

    Parameters:
        data (xr.Dataset): The input dataset containing Sv data.
        min_range (int): The minimum range index to consider.
        max_range (int): The maximum range index to consider.
    Returns:
        data (xr.Dataset): The modified dataset.
        ping_range_sv_array_clean (np.ndarray): The cleaned array of ping, range, and Sv values.
        labels (np.ndarray): The cluster labels assigned to the data points.    
    """

    channels, pings, ranges = data["Sv"].shape


    mean = data["Sv"][:,:,:].mean().values
    print("Mean Sv: ", mean)
    threshold = mean+10
    print("Threshold Sv: ", threshold)


    data["Sv"] = data["Sv"].where(
        (data["Sv"][0, :, :] >= threshold) | 
        (data["Sv"][1, :, :] >= threshold) | 
        (data["Sv"][2, :, :] >= threshold)
    )

    data["Sv"] = data["Sv"][:,:,min_range:max_range]

    data= data.isel(range=slice(min_range, None))

    channels, pings, ranges = data["Sv"].shape

    ping_indices = np.arange(0, pings)
    range_indices = np.arange(0, ranges)

    ping_time_values = np.repeat(ping_indices, ranges)
    range_values = np.tile(range_indices, pings)

    sv_values_0 = data["Sv"].values[0, ping_time_values, range_values]
    sv_values_1 = data["Sv"].values[1, ping_time_values, range_values]
    sv_values_2 = data["Sv"].values[2, ping_time_values, range_values]

    diff_0_1 = sv_values_0 - sv_values_1
    diff_1_2 = sv_values_1 - sv_values_2
    diff_0_2 = sv_values_0 - sv_values_2

    ping_range_array = np.column_stack(( ping_time_values, range_values,sv_values_0,sv_values_1,sv_values_2,diff_0_1,diff_0_2,diff_1_2))

    valid_rows = ~np.isnan(ping_range_array[:, 2]) & ~np.isnan(ping_range_array[:, 3]) & ~np.isnan(ping_range_array[:, 4])


    ping_range_sv_array_clean = ping_range_array[valid_rows]


    db = DBSCAN(eps=40, min_samples=2000).fit(ping_range_sv_array_clean[:,:])


    labels = db.labels_

    return data,ping_range_sv_array_clean, labels
