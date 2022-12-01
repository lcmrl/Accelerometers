import pyrealsense2 as rs
import numpy as np

DATA = []

def initialize_camera():
    # start the frames pipe
    p = rs.pipeline()
    conf = rs.config()
    #conf.enable_all_streams()
    conf.enable_stream(rs.stream.accel, rs.format.motion_xyz32f)
    #conf.enable_stream(rs.stream.gyro)#, rs.format.motion_xyz32f, 200)
    prof = p.start(conf)
    return p


def gyro_data(gyro):
    return np.asarray([gyro.x, gyro.y, gyro.z])


def accel_data(accel):
    return np.asarray([accel.x, accel.y, accel.z])

p = initialize_camera()
print("[Camera initialized]")

for i in range(0,5000):
    f = p.wait_for_frames()
    timestamp = f.get_timestamp()
    accel = accel_data(f[0].as_motion_frame().get_motion_data())
    #gyro = gyro_data(f[1].as_motion_frame().get_motion_data())
    #print("accelerometer: ", accel)
    #print("gyro: ", gyro)
    DATA.append((timestamp, accel))#, gyro))
#p.stop()
with open('./out_realsense.txt', 'w') as out:
    for d in DATA:
        out.write("{},{},{},{}\n".format(d[0], d[1][0], d[1][1], d[1][2]))
    p.stop()



quit()
try:
    #while True:
    for i in range(0,500):
        f = p.wait_for_frames()
        timestamp = f.timestamp
        accel = accel_data(f[0].as_motion_frame().get_motion_data())
        gyro = gyro_data(f[1].as_motion_frame().get_motion_data())
        #print("accelerometer: ", accel)
        #print("gyro: ", gyro)
        DATA.append((timestamp, accel, gyro))
    #p.stop()
    with open('./out_realsense.txt', 'w') as out:
        for d in DATA:
            out.write("{},{},{},{}\n".format(d[0], d[1][0], d[1][1], d[1][2]))

finally:
    p.stop()


