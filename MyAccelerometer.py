import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt
from scipy.integrate import cumtrapz
import math

### INPUT PARAMETERS ###
acc_data_csv = r"./phyphox/sinusoidale5.csv"
separator = ";"
NUMBER_STATIC_EPOCHS = int(0.2/0.002)

# Butterworth low pass filter
sample_rate = 500 # Hz
cutoff = 10 # desired cutoff frequency of the filter in Hz
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

### MAIN ###
#raw_data_matrix = np.loadtxt(acc_data_csv, dtype=np.str, delimiter=separator, skiprows=2)
raw_data_matrix = np.loadtxt(acc_data_csv, dtype=float, delimiter=',', skiprows=1) #phyphox
print(np.shape(raw_data_matrix))
print(raw_data_matrix)



x = np.linspace(math.pi*(3/2), 50, num=1000)
print(x)
y = np.sin(x)
print(y)
plt.plot(x,y)
plt.show()



AX =  raw_data_matrix[:,2] #[:,1]
AX = AX.astype(float) # [m/s2]
AX_bias = np.mean(AX[:NUMBER_STATIC_EPOCHS])
print("AX_bias:", AX_bias)
AX = AX - AX_bias
TIME = raw_data_matrix[:,0] # [:,-2]
#TIME = TIME.astype(int)*0.001 # [sec]
AX=y
TIME=x
VX = NumIntegration(AX, TIME)
SX = NumIntegration(VX, TIME)

plt.plot(TIME[:], AX[:], 'b-', TIME[:], VX[:], 'r-', TIME[:], SX[:], 'y-')#, TIME, SX, 'g-') #plt.scatter(TIME, AX, color='r',marker=".", s=1)
plt.show()
quit()
### BUTTERWORTH ###
dfiltered_data = butter_lowpass_filter(AX, cutoff, sample_rate, order, nyq)
VX = NumIntegration(dfiltered_data, TIME)
SX = NumIntegration(VX, TIME)

plt.plot(TIME[:], AX[:], 'b-', TIME[:], dfiltered_data[:], 'r-', TIME[:], VX[:], 'g-', TIME[:], SX[:], 'g--')
plt.show()