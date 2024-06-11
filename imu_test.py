import struct
import os
import numpy as np
from pyomyo import Myo, emg_mode
from collections import deque

path_quaternion = "/quaternion"
path_gyroscope = "/gyroscope"
path_acceleration = "/acceleration"

Class_num = 7

class Collector(object):
    def __init__(self, name="Collector", color=(0, 255, 0)):
        self.name = name
        self.color = color
        self.create_directory('data')  # Create directories for data storage

        for i in range(Class_num):
            with open('data' + path_quaternion + '/quat_vals%d.dat' % i, 'ab') as f: pass
            with open('data' + path_gyroscope + '/gyro_vals%d.dat' % i, 'ab') as f: pass
            with open('data' + path_acceleration + '/accl_vals%d.dat' % i, 'ab') as f: pass

    def create_directory(self, directory):
        try:
            if not os.path.exists(directory + path_quaternion):
                os.makedirs(directory + path_quaternion)
            if not os.path.exists(directory + path_gyroscope):
                os.makedirs(directory + path_gyroscope)
            if not os.path.exists(directory + path_acceleration):
                os.makedirs(directory + path_acceleration)
        except OSError:
            print("Error: Failed to create the directory.")

    def store_quat_data(self, cls, vals):
        with open('data' + path_quaternion + '/quat_vals%d.dat' % cls, 'ab') as f:
            f.write(pack('4i', *vals))

    def store_gyro_data(self, cls, vals):
        with open('data' + path_gyroscope + '/gyro_vals%d.dat' % cls, 'ab') as f:
            f.write(pack('3i', *vals))

    def store_accl_data(self, cls, vals):
        with open('data' + path_acceleration + '/accl_vals%d.dat' % cls, 'ab') as f:
            f.write(pack('3i', *vals))

class MyoIMU(Myo):
    def __init__(self, cls, tty=None, mode=emg_mode.PREPROCESSED, hist_len=25):
        Myo.__init__(self, tty, mode=mode)
        self.cls = cls
        self.hist_len = hist_len
        self.history = deque([0] * self.hist_len, self.hist_len)
        self.last_pose = None
        self.pose_handlers = []

class IMUController(object):
    def __init__(self, m):
        self.recording = -1
        self.m = m
        self.quat = (0,) * 4
        self.gyro = (0,) * 3
        self.accl = (0,) * 3

    def __call__(self, quat, gyro, accl):
        self.quat = quat
        self.gyro = gyro
        self.accl = accl
        if self.recording >= 0:
            self.m.cls.store_quat_data(self.recording, quat)
            self.m.cls.store_gyro_data(self.recording, gyro)
            self.m.cls.store_accl_data(self.recording, accl)

        print("quat:", self.quat)
        print("gyro:", self.gyro)
        print("accl:", self.accl)

def pack(fmt, *args):
    return struct.pack('<' + fmt, *args)

if __name__ == '__main__':
    # Data Collect
    m = MyoIMU(Collector(), mode=emg_mode.RAW)
    imu_hnd = IMUController(m)
    m.add_imu_handler(imu_hnd)
    m.connect()

    # Set Myo LED color to model color
    m.set_leds(m.cls.color, m.cls.color)
    import time
    try:
        while True:
            m.run()
            time.sleep(0.4)
    except KeyboardInterrupt:
        pass
    finally:
        m.disconnect()
        print("Disconnected from Myo.")
