import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import Clustering
from collections import Counter

path = "/Users/erlenddahle/Documents/NTNU/Masteroppgave/Koding/Data/D20230724-T133518" #3 


#echo_data_sv = xr.open_dataset(path + "_bottom_removed.nc")
echo_data_sv = xr.open_dataset(path + "_removed_fake_bottom.nc")


echo_data_sv["Sv"].load()


min_range = 175#40 35
max_range = 1000#850 450
echo_data_sv,ping_range_sv_array,labels=Clustering.clustering(echo_data_sv, min_range, max_range)


echo_data_sv["Sv"][:,:,:max_range-min_range].plot.pcolormesh(
    x="ping_time",
    y="range",
    row="channel",
    figsize=(15, 10),
    yincrease=False,  
    # vmin=-90,
    # vmax=-10, 
    robust=True,
    cmap="jet"

)

plt.show()


channels, pings, ranges = echo_data_sv["Sv"].shape
reshaped_sv = np.full((channels, pings, ranges), np.nan)
labels_used=labels

for i in range(ping_range_sv_array.shape[0]):
    ping_idx = int(ping_range_sv_array[i,0])
    range_idx = int(ping_range_sv_array[i,1])

    reshaped_sv[0, ping_idx, range_idx] = labels_used[i]
    reshaped_sv[1, ping_idx, range_idx] = labels_used[i]
    reshaped_sv[2, ping_idx, range_idx] = labels_used[i]


echo_data_sv["Sv"] = (("channel", "ping_time", "range"), reshaped_sv)


n_clusters_ = len(set(labels_used)) - (1 if -1 in labels_used else 0)
print("Number of clusters: ", n_clusters_)
cluster_sizes = Counter(labels_used)
for cluster_id, size in cluster_sizes.items():
    if cluster_id == -1:
        continue  # Skip noise points
    print(f"Cluster {cluster_id}: {size} points")
echo_data_sv["Sv"][:1,:,:max_range-min_range].plot.pcolormesh(

    x="ping_time",
    y="range",
    row="channel",
    figsize=(15, 5),
    yincrease=False,  
    robust=True,
    cmap="jet"

)
plt.show()