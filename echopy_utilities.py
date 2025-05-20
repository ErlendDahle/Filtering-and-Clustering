from typing import Tuple, Dict
import numpy as np
import xarray as xr
from scipy import interpolate

# Tool that takes an xarray DataArray and finds the edge of the populated domain of the data along one dimension.
def find_edge(data: xr.DataArray,
              dim: str = "range",
              check_corners: Tuple[Dict[str, int]] = ({"ping_time": 0}, {"ping_time": -1})) -> int:
    """
    Finds the edge of the populated domain of the data along a specified dimension.

    Parameters:
        data (xr.DataArray): The input_path DataArray.
        dim (str): The dimension along which to find the edge. Defaults to "range".
        check_corners (Tuple[Dict[str, int]]): A tuple of dictionaries specifying the corners to check first.
            Defaults to ({"ping_time": 0}, {"ping_time": -1}).

    Returns:
        int: The index of the edge along the specified dimension.
    """
    
    start, i = 0, 0
    for i in range(data.sizes[dim]):
        if not data.isel({dim: i}).sum() == 0:
            start = i
            break

    for i in range(start, data.sizes[dim]):
        # Check first and last element in slice of data at index i, then sum:
        for corner in check_corners:
            if not int(data.isel({dim: i})[corner].sum()) == 0:
                break
        else:
            if int(data.isel({dim: i}).sum()) == 0:
                return i
  
    return i



# def find_bottom(cutoff, data: xr.DataArray, dim: str = "range") -> xr.DataArray:

#     """
#     Finds the edge of the populated domain of the data along a specified dimension.

#     Parameters:
#         data (xr.DataArray): The input DataArray.
#         dim (str): The dimension along which to find the edge. Defaults to "range".

#     Returns:
#         xr.Dataset: The modified data with indices set to Nan if under bottom.
#     """

#     Minimum_strength_1 = -20
#     Minimum_strength_2 = -30

#     i, j, start = 0, 0, 0
#     for i in range(300,cutoff-500):
#         if data.isel({dim: i}).max() > Minimum_strength_1: 
#             start = i
#             break
#     #data.sizes.get("ping_time",0)
#     for j in range(data.sizes.get("ping_time",0)):# ping time
#         for i in range(start, cutoff-500): # range
#             if int(data.isel({dim: i})[{"ping_time": j}].sum()) > Minimum_strength_2:
#                 data[{dim: slice(i+5, cutoff), "ping_time": j}] = np.nan
                


    
#     return data


# Utility that creates the range dimension from the range_sample dimension calibrated to meters
def range_from_range_sample(data: xr.Dataset) -> xr.Dataset:
    """
    Create a new xarray Dataset with a "range" coordinate
    based on the "range_sample" coordinate in the input_path Dataset calibrated to meters.

    Parameters:
        data (xr.Dataset): The input_path Dataset.

    Returns:
        xr.Dataset: The new Dataset with the "range" coordinate.
    """
    data = data.assign_coords(
        {"range": ("range_sample",
                   data.coords["range_sample"].values * float(data["echo_range"].isel(ping_time=0,
                                                                                      channel=0,
                                                                                      range_sample=1)))}
    ).swap_dims({"range_sample": "range"})
    data["range"] = data["range"].assign_attrs(
        {"long_name": "range", "units": "meter"}
    )
    return data


# Utility for interpolating missing values in echogram, one channel
def interpolate_missing_pixels(
        image: np.ndarray[float],
        mask: np.ndarray[bool],
        method: str = 'nearest',
        fill_value: int = 0
):
    """
    :param image: a 2D image
    :param mask: a 2D boolean image, True indicates missing values
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """
    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = interpolate.griddata(
        (known_x, known_y), known_v, (missing_x, missing_y),
        method=method, fill_value=fill_value
    )

    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values

    return interp_image

# Wrapper for interpolate_missing_pixels reading a full echogram and interpolating each channel separately.
def interp2d(data: xr.DataArray, method: str = 'nearest', fill_value: int = 0) -> xr.DataArray:
    for channel in range(data.sizes["channel"]):
        # noinspection PyTypeChecker
        data.values[channel] = interpolate_missing_pixels(
            data.values[channel],
            np.ma.masked_invalid(data[channel].values).mask,
            method=method,
            fill_value=fill_value
        )
    return data