import xarray as xr
import echopy_utilities 
import matplotlib.pyplot as plt
import bottom_filter

path = "/Users/erlenddahle/Documents/NTNU/Masteroppgave/Koding/Data/D20230724-T133518" #3 



#echo_data_sv = xr.open_dataset(path + "_bottom.nc")
echo_data_sv = xr.open_dataset(path + "_stripe_noise.nc")
#echo_data_sv = xr.open_dataset(path + "_bottom_removed.nc")

echo_data_sv["Sv"].load()

cutoff = echopy_utilities.find_edge(echo_data_sv["Sv"][2])
print(cutoff)
echo_data_sv["Sv"] = bottom_filter.remove_bottom(cutoff,echo_data_sv["Sv"])


# bottom_path = path + "_bottom_removed.nc"
# if os.path.exists(bottom_path):
#     os.remove(bottom_path)  
# echo_data_sv["Sv"].to_netcdf(path + "_bottom_removed.nc")

print(echo_data_sv["Sv"].shape)
echo_data_sv["Sv"][:, :,:cutoff].plot.pcolormesh(
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