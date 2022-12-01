import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt
from scipy.integrate import cumtrapz
from scipy import stats
import math

### INPUT PARAMETERS ###
acc_data_csv = r"./out_realsense.txt"
separator = ";"
NUMBER_STATIC_EPOCHS = 100 #int(0.2/0.002)
SLOPE_velocity = 0#-0.00157864

# Butterworth low pass filter
sample_rate = 500 # Hz
cutoff = 5 # desired cutoff frequency of the filter in Hz
nyq = 0.5 * sample_rate  # nyquist frequency
order = 2

### FUNCTIONS ###
def NumIntegration(input_array, time_array):
    out_array = np.zeros(len(input_array))
    for i in range(1,len(input_array)):
        out_array[i] = out_array[i-1] + (input_array[i]+input_array[i-1])/2*(time_array[i]-time_array[i-1])
    #out_array = cumtrapz(input_array, time_array, initial=0)
    return out_array

def butter_lowpass_filter(data, cutoff, fs, order, nyq):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def butter_highpass_filter(data, cutoff, fs, order, nyq):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

### MAIN ###
#raw_data_matrix = np.loadtxt(acc_data_csv, dtype=np.str, delimiter=separator, skiprows=2)
raw_data_matrix = np.loadtxt(acc_data_csv, dtype=float, delimiter=',', skiprows=0) #phyphox
print(np.shape(raw_data_matrix))
print(raw_data_matrix)



AX =  raw_data_matrix[:,1] #[:,1]
AX = AX.astype(float) # [m/s2]
AX_bias = np.mean(AX[:NUMBER_STATIC_EPOCHS])
print("AX_bias:", AX_bias)
AX = AX - AX_bias
TIME = raw_data_matrix[:,0] # [:,-2]
#TIME = np.linspace(0, 500, len(AX))
TIME = TIME.astype(float)*0.001 # [sec]
TIME = TIME - TIME[0]
print(AX[:5])
print(TIME[:5])
plt.plot(TIME[:], AX[:], 'b-')
plt.show()

VX = NumIntegration(AX, TIME)
slope, intercept, r_value, p_value, std_err = stats.linregress(TIME[500:2000], VX[500:2000])
print("Velocity line fitting (slope, intercept): {} {}".format(slope, intercept))
interpolated = TIME * slope
VX = VX - SLOPE_velocity * TIME

SX = NumIntegration(VX, TIME)

plt.plot(TIME[:], AX[:], 'b-', TIME[:], VX[:], 'r-', TIME[:], SX[:], 'y-', TIME[:], interpolated[:], 'b--')#, TIME, SX, 'g-') #plt.scatter(TIME, AX, color='r',marker=".", s=1)
plt.show()

### BUTTERWORTH ###
dfiltered_data = butter_lowpass_filter(AX, cutoff, sample_rate, order, nyq)
VX = NumIntegration(dfiltered_data, TIME)
VX = butter_highpass_filter(VX, 1, sample_rate, order, nyq)
SX = NumIntegration(VX, TIME)

plt.plot(TIME[:], AX[:], 'b-', TIME[:], dfiltered_data[:], 'r-', TIME[:], VX[:], 'g-', TIME[:], SX[:], 'g--')
plt.show()