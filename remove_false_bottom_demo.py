import xarray as xr
import false_bottom_filter
import numpy as np
import echopype as ep
import matplotlib.pyplot as plt
import polars as pl
import echopy_utilities
import filter_utilities 



dataframe = pl.read_parquet("/Users/erlenddahle/Documents/NTNU/Masteroppgave/Koding/partials/D20230724-T133518.parquet")


Bathymetry = dataframe['bathymetry'].to_numpy().transpose()
Bathymetry = np.concatenate((np.full(200, Bathymetry[0]), Bathymetry))
Bathymetry = Bathymetry[:-200]


path = "/Users/erlenddahle/Documents/NTNU/Masteroppgave/Koding/Data/D20230724-T133518" #3 


#echo_data_sv = xr.open_dataset(path + "_bottom.nc")
echo_data_sv = xr.open_dataset(path + "_fake_bottom.nc")
cutoff = echopy_utilities.find_edge(echo_data_sv["Sv"][0])


cutoff= 2150
cluster_lines, cluster_ranges=false_bottom_filter.create_sonar_lines(echo_data_sv["Sv"],cutoff)
print(cluster_lines.shape)
plt.figure(figsize=(10, 6))
for i in range(len(cluster_ranges)):
    plt.plot(cluster_lines[i],label=f'Sonar channel {cluster_ranges[i][0]}')

plt.plot(Bathymetry, label='Bathymetry')
plt.ylim(-400, 0)  
plt.ylabel('Depth (m)')
plt.title('Bathymetry and Sonar data')
plt.legend()
plt.grid(True)
plt.show()

cluster_lines=filter_utilities.moving_average(cluster_lines,window_size=10)
cluster_lines, cluster_ranges =false_bottom_filter.delete_clusters_with_high_nan_ratio(cluster_lines,cluster_ranges)
cluster_lines, cluster_ranges =false_bottom_filter.delete_high_mse_clusters(cluster_lines, cluster_ranges, Bathymetry)



#manual delete:
# cluster_lines = np.delete(cluster_lines, [1, 3, 4], axis=0)
# cluster_ranges = np.delete(cluster_ranges, [1, 3, 4], axis=0)



plt.figure(figsize=(10, 6))
for i in range(len(cluster_ranges)):
    print(i)
    plt.plot(cluster_lines[i] - np.mean(cluster_lines[i]),label=f'Sonar channel {cluster_ranges[i][0]}')

plt.plot(Bathymetry - np.mean(Bathymetry), label='bathymetry')
plt.ylim(-75, 75)  # Set y-axis limits from 0 to -450  
plt.ylabel('Depth (m)')
plt.title('Bathymetry and sonar data')
plt.legend()
plt.grid(True)
plt.show() 


echo_data_full = xr.open_dataset(path + "_bottom_removed.nc")
echo_data_full["Sv"].load()

echo_data_full["Sv"]= false_bottom_filter.remove_false_bottom(echo_data_full["Sv"],echo_data_sv["Sv"], cluster_ranges)


echo_data_full["Sv"][:, :,:cutoff].plot.pcolormesh(
    x="ping_time",
    y="range",
    # y="range_sample",
    # y="echo_range",
    row="channel",
    figsize=(15, 10),
    yincrease=False,  # reverse the range
    # vmin=-90,
    # vmax=-10,
    robust=True,
    cmap="jet"
    
)
plt.show() 