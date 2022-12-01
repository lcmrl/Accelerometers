### References ###
# https://en.wikipedia.org/wiki/Kalman_filter#Overview_of_the_calculation
# https://en.wikipedia.org/wiki/Extended_Kalman_filter
# https://towardsdatascience.com/the-kalman-filter-and-external-control-inputs-70ea6bcbc55f
# https://towardsdatascience.com/kalman-filter-in-a-nutshell-e66154a06862

import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import math

### INPUT PARAMETERS ###
acc_data_csv = r"./out_realsense.txt"
separator = ","
NUMBER_STATIC_EPOCHS = 100

### MAIN ###
VX = []
SX = []
# Import acceleration
raw_data_matrix = np.loadtxt(acc_data_csv, dtype=float, delimiter=separator, skiprows=0)
AX =  raw_data_matrix[:,1]
AX = AX.astype(float) # [m/s2]
AX_bias = np.mean(AX[:NUMBER_STATIC_EPOCHS])
AX = AX - AX_bias
TIME = raw_data_matrix[:,0]
TIME = TIME.astype(float)*0.001 # [sec]
TIME = TIME - TIME[0]
plt.plot(TIME[:], AX[:], 'b-')
plt.show()

# Initial state
x_k_1 = np.array([[0, 0, 0]]).T # position, velocity, acceleration
print(f"initial state: \n{x_k_1}")

# Time interval
dt = TIME[1] - TIME[0]
print(f"\ndelta_t [sec] = {dt}")

# The state-transition model
F_k = np.array([
    [1,   dt,  dt**2/2], #dt**2/2
    [0,    1,       dt],
    [0,    0,        1]
])
print(f"\nThe state-transition model:\n{F_k}")

# The covariance of the process noise
a = 0.5
G = np.array([[dt**2/2, dt, 0]]).T
Q_k = np.dot(G, np.dot(G.T, a**2))
#Q_k = np.array([
#    [1**2,        0,         0],
#    [     0,   1**2,         0],
#    [     0,        0,    1**2]
#])
print(f"\nThe covariance of the process noise:\n{Q_k}")
P_k_1 = Q_k

# The covariance of the observation noise
R_k = 0.5**2

# The observation model
H = np.array([[0, 0, 1]], ndmin=2)
print(f"\nThe observation model:\n{H}")

for i in range(0, len(AX)):
    # Prediction
    x_k_priori = F_k @ x_k_1 # + B @ u
    P_k_priori = F_k @ P_k_1 @ linalg.inv(F_k)

    # Innovation
    z_k = AX[i] # Observation at time k
    y = z_k - np.dot(H, x_k_priori) # Innovation residual
    S_k =  np.dot(H, np.dot(P_k_priori, H.T)) + R_k # Innovation covariance
    K_k = np.dot(P_k_priori, np.dot(H.T, linalg.inv(S_k))) # Optimal gain
    #K_k = np.array([[1,1,1]]).T
    print(f"\nOptimal gain:\n{K_k}")
    x_k_posteriori = x_k_priori + np.dot(K_k, y)
    KH = np.dot(K_k, H)
    P_k = np.dot((np.identity(np.shape(KH)[0]) - KH), P_k_priori)
    y_post_fit = z_k - np.dot(H, x_k_posteriori)

    # Update
    SX.append(x_k_posteriori[0])
    VX.append(x_k_posteriori[1])
    x_k_1 = x_k_posteriori
    P_k_1 = P_k


plt.plot(TIME[:], AX[:], 'r-', TIME[:], VX[:], 'b-', TIME[:], SX[:], 'g-')
plt.show()