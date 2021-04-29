import numpy as np
import math
#pulls acceleration or gravity data
def pull_data(dir_name, file_name):
    f = open(dir_name + '/' + file_name + '.csv')
    xs = []
    ys = []
    zs = []
    rs = []
    timestamps = []
    f.readline() # ignore headers
    for line in f:
        value = line.split(',')
        if len(value) > 3:
            timestamps.append(float(value[0]))
            x = float(value[4])
            y = float(value[3])
            z = float(value[2])
            r = math.sqrt(x ** 2 + y ** 2 + z ** 2)
            xs.append(x)
            ys.append(y)
            zs.append(z)
            rs.append(r)
    return np.array(xs), np.array(ys), np.array(zs), np.array(rs), np.array(timestamps)

#pulls orientation data
def pull_orientation_data(dir_name, file_name):
    f = open(dir_name + '/' + file_name + '.csv')
    qx = []
    qy = []
    qz = []
    qw = []
    yaw = []
    roll = []
    pitch = []
    timestamps = []
    f.readline() # ignore headers
    for line in f:
        value = line.split(',')
        if len(value) > 3:
            timestamps.append(float(value[0]))
            x = float(value[3])
            y = float(value[7])
            w = float(value[6])
            z = float(value[4])

            qx.append(x)
            qy.append(y)
            qz.append(z)
            qw.append(w)

            yaw.append(float(value[2]))
            roll.append(float(value[5]))
            pitch.append(float(value[8]))

    return np.array(qx), np.array(qy), np.array(qz), np.array(qw), np.array(yaw), np.array(roll), np.array(pitch), np.array(timestamps)
