import echopype as ep
import matplotlib.pyplot as plt
import echopy_utilities 
import bottom_filter
import numpy as np
import fourier_filter 
import os
import xarray as xr 

# Define path to sonar data

path = "/Users/erlenddahle/Documents/NTNU/Masteroppgave/Koding/Data/D20230701-T001751" #1
path = "/Users/erlenddahle/Documents/NTNU/Masteroppgave/Koding/Data_raw/D20230701-T093433" #2
path = "/Users/erlenddahle/Documents/NTNU/Masteroppgave/Koding/Data/D20230724-T133518" #3 
#path= "/Users/erlenddahle/Documents/NTNU/Masteroppgave/Koding/Data_raw/D20230731-T235246" #4
#path = "/Users/erlenddahle/Documents/NTNU/Masteroppgave/Koding/Data_raw/D20221231-T234438" #5 Januar data, ganske bra 
#path = "/Users/erlenddahle/Documents/NTNU/Masteroppgave/Koding/Data/B202407-D20240526-T082458" #6 Iceland data
#path = "/Users/erlenddahle/Documents/NTNU/Masteroppgave/Koding/Data_Januar/D20230101-T073647"

#path = "/Users/erlenddahle/Documents/NTNU/Masteroppgave/Koding/July2023/D20230701-T081641"
#path= "/Users/erlenddahle/Documents/NTNU/Masteroppgave/Koding/Data_raw/D20230701-T232031" #4


#path = "/Users/erlenddahle/Documents/NTNU/Masteroppgave/Koding/Feb2023/D20230201-T081427"
echo_data = ep.open_raw(raw_file=(path + ".raw"), sonar_model="EA640") #sonar_model="EK80" 
# #echo_data = ep.open_raw(raw_file=(path + ".raw"), sonar_model="EK80") #sonar_model="EK80" #sonar_model="EA640"
echo_data_sv = ep.calibrate.compute_Sv(echo_data, waveform_mode="CW", encode_mode="power")
echo_data_sv = echopy_utilities.range_from_range_sample(echo_data_sv)
echo_data_sv["Sv"] = bottom_filter.scale_channel_1(echo_data_sv["Sv"])



# echo_data_sv = xr.open_dataset(path + "_.nc")
cutoff = echopy_utilities.find_edge(echo_data_sv["Sv"][2])

notch_width=5
radius =20

echo_data_sv["Sv"]=fourier_filter.stripe_noise_filter(echo_data_sv["Sv"], cutoff, notch_width , radius )   

fake_bottom_path = path + "_stripe_noise.nc"
if os.path.exists(fake_bottom_path):
    os.remove(fake_bottom_path)  
echo_data_sv["Sv"].to_netcdf(path + "_stripe_noise.nc")

# Plot the filtered sonar image
#print(echo_data_sv["Sv"][:, :,:cutoff].values)
echo_data_sv["Sv"][:, :,:cutoff].plot.pcolormesh(
    x="ping_time",
    y="range",
    row="channel",
    figsize=(15, 10),
    yincrease=False,  # reverse the range
    robust=True,
    cmap="jet"
)
#plt.title("Data After 2D Fourier Filtering")
plt.show()