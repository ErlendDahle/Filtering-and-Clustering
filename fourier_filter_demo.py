import echopype as ep
import matplotlib.pyplot as plt
import echopy_utilities 
import bottom_filter
import fourier_filter 
import os


# Define path to  data
path = "/Users/erlenddahle/Documents/NTNU/Masteroppgave/Koding/Data_raw/D20230724-T133518" #3 


echo_data = ep.open_raw(raw_file=(path + ".raw"), sonar_model="EA640") #sonar_model="EK80" 
echo_data_sv = ep.calibrate.compute_Sv(echo_data, waveform_mode="CW", encode_mode="power")
echo_data_sv = echopy_utilities.range_from_range_sample(echo_data_sv)
echo_data_sv["Sv"] = bottom_filter.scale_channel_1(echo_data_sv["Sv"])
cutoff = echopy_utilities.find_edge(echo_data_sv["Sv"][2])

notch_width=5
radius =20

echo_data_sv["Sv"]=fourier_filter.stripe_noise_filter(echo_data_sv["Sv"], cutoff, notch_width , radius )   

fake_bottom_path = path + "_stripe_noise.nc"
if os.path.exists(fake_bottom_path):
    os.remove(fake_bottom_path)  
echo_data_sv["Sv"].to_netcdf(path + "_stripe_noise.nc")

#plot the filtered sonar image

echo_data_sv["Sv"][:, :,:cutoff].plot.pcolormesh(
    x="ping_time",
    y="range",
    row="channel",
    figsize=(15, 10),
    yincrease=False,  
    robust=True,
    cmap="jet"
)
plt.title("Data After 2D Fourier Filtering")
plt.show()