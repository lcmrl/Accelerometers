import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt
from scipy.integrate import cumtrapz
from scipy import stats
from scipy.fft import rfft, rfftfreq, irfft
import math

### INPUT PARAMETERS ###
acc_data_csv = r"./out_realsense.txt"
NUMBER_STATIC_EPOCHS = 100

### FUNCTIONS ###
def NumIntegration(input_array, time_array):
    out_array = np.zeros(len(input_array))
    for i in range(1,len(input_array)):
        out_array[i] = out_array[i-1] + (input_array[i]+input_array[i-1])/2*(time_array[i]-time_array[i-1])
    #out_array = cumtrapz(input_array, time_array, initial=0)
    return out_array

### MAIN ###
raw_data_matrix = np.loadtxt(acc_data_csv, dtype=float, delimiter=',', skiprows=0) #phyphox
AX =  raw_data_matrix[:,1]
AX = AX.astype(float) # [m/s2]
AX_bias = np.mean(AX[:NUMBER_STATIC_EPOCHS])
AX = AX - AX_bias
TIME = raw_data_matrix[:,0]
TIME = TIME.astype(float)*0.001 # [sec]
TIME = TIME - TIME[0]
plt.plot(TIME[:], AX[:], 'b-')
plt.show()


# Number of samples in normalized_tone
N = len(TIME)

yf = rfft(AX)
xf = rfftfreq(N, 1)

plt.plot(xf, np.abs(yf))
plt.show()

# Filtering frequency
print(yf[len(yf)-20:len(yf)-1])
yf[len(yf)-2200:len(yf)-1] = 0
plt.plot(xf, np.abs(yf))
plt.show()

# Applying the Inverse FFT
AX = irfft(yf)
plt.plot(AX[:])
plt.show()

VX = NumIntegration(AX, TIME)
SX = NumIntegration(VX, TIME)

plt.plot(TIME[:], AX[:], 'b-', TIME[:], VX[:], 'r-', TIME[:], SX[:], 'y-')
plt.show()


yf = rfft(SX)
xf = rfftfreq(N, 1)

plt.plot(xf, np.abs(yf))
plt.show()

# Filtering frequency
print(yf[len(yf)-20:len(yf)-1])
yf[0:13] = 0
plt.plot(xf, np.abs(yf))
plt.show()

# Applying the Inverse FFT
AX = irfft(yf)
plt.plot(AX[:])
plt.show()