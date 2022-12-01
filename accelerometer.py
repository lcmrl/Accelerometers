import numpy as np

FORMAT = "phyphox" # "telemetryextractor" or "phyphox"
#raw_data_path = r"C:\Users\Luscias\Downloads\Raw Data.csv"
raw_data_path = r"C:\Users\Luscias\Desktop\provvisorio\provvisorio1\Raw Data.csv"

if FORMAT == "phyphox":
    with open(raw_data_path, 'r') as raw_data_file:
        acc_matrix = np.loadtxt(raw_data_file, dtype=float, delimiter=',', skiprows=1)
        data_shape = np.shape(acc_matrix)
        velocity_matrix = np.zeros(data_shape)
        velocity_matrix[:,0] = acc_matrix[:,0]
        loc_matrix = np.zeros(data_shape)
        loc_matrix[:,0] = acc_matrix[:,0]

        for i in range(1, data_shape[0]):
            t1 = velocity_matrix[i-1, 0]
            t2 = velocity_matrix[i, 0]
            dt = t2 - t1

            vx1 = velocity_matrix[i-1, 1]
            ax1 = acc_matrix[i-1, 1]
            vx2 = vx1 + ax1 * dt
            velocity_matrix[i, 1] = vx2

            vy1 = velocity_matrix[i-1, 2]
            ay1 = acc_matrix[i-1, 2]
            vy2 = vy1 + ay1 * dt
            velocity_matrix[i, 2] = vy2

            vz1 = velocity_matrix[i-1, 3]
            az1 = acc_matrix[i-1, 3]
            vz2 = vz1 + az1 * dt
            velocity_matrix[i, 3] = vz2

            velocity_matrix[i, 4] = (vx2**2+vy2**2+vz2**2)**0.5

        print(velocity_matrix)

        for i in range(1, data_shape[0]):
            t1 = loc_matrix[i-1, 0]
            t2 = loc_matrix[i, 0]
            dt = t2 - t1

            sx1 = loc_matrix[i-1, 1]
            vx1 = velocity_matrix[i-1, 1]
            sx2 = sx1 + vx1 * dt
            loc_matrix[i, 1] = sx2

            sy1 = loc_matrix[i-1, 2]
            vy1 = velocity_matrix[i-1, 2]
            sy2 = sy1 + vy1 * dt
            loc_matrix[i, 2] = sy2

            sz1 = loc_matrix[i-1, 3]
            vz1 = velocity_matrix[i-1, 3]
            sz2 = sz1 + vz1 * dt
            loc_matrix[i, 3] = sz2

            loc_matrix[i, 4] = (sx2**2+sy2**2+sz2**2)**0.5

        print(loc_matrix)

    with open("./loc_output.txt", 'w') as out_file:
        for i in range(0, data_shape[0]):
            out_file.write("{},{},{},{}\n".format(loc_matrix[i,0], loc_matrix[i,1], loc_matrix[i,2], loc_matrix[i,3]))

    with open("./vel_output.txt", 'w') as out_file:
        for i in range(0, data_shape[0]):
            out_file.write("{},{},{},{}\n".format(velocity_matrix[i,0], velocity_matrix[i,1], velocity_matrix[i,2], velocity_matrix[i,3]))

    with open("./acc_output.txt", 'w') as out_file:
        for i in range(0, data_shape[0]):
            out_file.write("{},{},{},{}\n".format(acc_matrix[i,0], acc_matrix[i,1], acc_matrix[i,2], acc_matrix[i,3]))







elif FORMAT == "telemetryextractor":
    with open(raw_data_path, 'r') as raw_data_file:
        acc_matrix = np.loadtxt(raw_data_file, dtype=np.str, delimiter='","', skiprows=1)
        acc_matrix = acc_matrix[:, 1:5]
        data_shape = np.shape(acc_matrix)
        for i in range(0, data_shape[0]):
            date, hour = acc_matrix[i, 0].split('T', 1)
            hour = hour[:-1]
            #print(date, hour)
            hour, min, sec = hour.split(':', 2)
            hour, min, sec = float(hour), float(min), float(sec)
            time = hour*60*60 + min*60 + sec
            #print(hour, min, sec)
            acc_matrix[i, 0] = time
        acc_matrix = acc_matrix.astype('float')
        
        velocity_matrix = np.zeros(data_shape)
        velocity_matrix[:,0] = acc_matrix[:,0]
        loc_matrix = np.zeros(data_shape)
        loc_matrix[:,0] = acc_matrix[:,0]
    
        for i in range(1, data_shape[0]):
            t1 = velocity_matrix[i-1, 0]
            t2 = velocity_matrix[i, 0]
            dt = t2 - t1
    
            vx1 = velocity_matrix[i-1, 1]
            ax1 = acc_matrix[i-1, 1]
            vx2 = vx1 + ax1 * dt
            velocity_matrix[i, 1] = vx2
    
            vy1 = velocity_matrix[i-1, 2]
            ay1 = acc_matrix[i-1, 2]
            vy2 = vy1 + ay1 * dt
            velocity_matrix[i, 2] = vy2
    
            vz1 = velocity_matrix[i-1, 3]
            az1 = acc_matrix[i-1, 3]
            vz2 = vz1 + az1 * dt
            velocity_matrix[i, 3] = vz2
    
            #velocity_matrix[i, 4] = (vx2**2+vy2**2+vz2**2)**0.5
    
        print(velocity_matrix)
    
        for i in range(1, data_shape[0]):
            t1 = loc_matrix[i-1, 0]
            t2 = loc_matrix[i, 0]
            dt = t2 - t1
    
            sx1 = loc_matrix[i-1, 1]
            vx1 = velocity_matrix[i-1, 1]
            sx2 = sx1 + vx1 * dt
            loc_matrix[i, 1] = sx2
    
            sy1 = loc_matrix[i-1, 2]
            vy1 = velocity_matrix[i-1, 2]
            sy2 = sy1 + vy1 * dt
            loc_matrix[i, 2] = sy2
    
            sz1 = loc_matrix[i-1, 3]
            vz1 = velocity_matrix[i-1, 3]
            sz2 = sz1 + vz1 * dt
            loc_matrix[i, 3] = sz2
    
            #loc_matrix[i, 4] = (sx2**2+sy2**2+sz2**2)**0.5
    
        print(loc_matrix)
    
    with open("./acc_output.txt", 'w') as out_file:
        for i in range(0, data_shape[0]):
            out_file.write("{},{},{},{}\n".format(loc_matrix[i,0], loc_matrix[i,1], loc_matrix[i,2], loc_matrix[i,3]))