
import numpy as np
import xarray as xr
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt


def stripe_noise_filter(data: xr.DataArray,cutoff, notch_width,radius,horisontal_filter=False ) -> xr.DataArray:

    original_sv_data = data.values[:, :, :cutoff]  
    # Identify the NaN mask from the original data (white areas)
    nan_mask = np.isnan(original_sv_data)
    # Extract sonar image 
    sv_data = data.values[:, :, :cutoff]  
    mean_value = np.nanmean(sv_data)
    sv_data = np.where(np.isnan(sv_data), mean_value, sv_data)

    # Apply 2D Fourier Transform
    F = fft2(sv_data)
    F_shifted = fftshift(F)

    # Get image dimensions
    rows, cols = sv_data[0,:,:].shape
    center_x, center_y = cols // 2, rows // 2

    mask = np.ones((rows, cols), dtype=np.float32)
    # Remove vertical noise 
    mask[:, center_x - notch_width : center_x + notch_width] = 0
    if horisontal_filter:   # Remove horizontal noise
        mask[center_y - notch_width : center_y + notch_width, :] = 0

    Y, X = np.ogrid[:rows, :cols]
    distance_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    circle_mask = distance_from_center <= radius
    mask[circle_mask] = 1

    fig, axes = plt.subplots(1, 1, figsize=(6, 6))

    F_magnitude = np.log(1 + np.abs(F_shifted))
    # Plot Fourier Magnitude Spectrum
    # axes[0].imshow(F_magnitude[1,:,:], cmap='gray', aspect='auto')
    # axes[0].set_title("Fourier Magnitude Spectrum")

    # # Plot Mask Overlay on Fourier Spectrum
    axes.imshow(F_magnitude[1,:,:], cmap='gray', aspect='auto')
    axes.imshow(mask, cmap='gray', alpha=0.5)  # Overlay mask with transparency
    axes.set_title("Filter Mask Applied to Spectrum")

    plt.show()

    # Apply mask in frequency domain
    F_filtered = F_shifted * mask

    # Inverse Fourier Transform
    F_inv_shifted = ifftshift(F_filtered)
    filtered_sv_data = np.real(ifft2(F_inv_shifted))
    # # Replace original Sv data with filtered result
    filtered_sv_data[nan_mask] = np.nan 
    data.values[:,:, :cutoff] = filtered_sv_data

    return data 