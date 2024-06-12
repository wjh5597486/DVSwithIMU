from pyomyo import Myo, emg_mode

class IMU:
    def __init__(self, args):
        # hyperparameter to Save
        self.gyro = args.gyro
        self.quat = args.quat
        self.accl = args.accl
        self.show = True

        # device
        # if device is running, the handler function will be executed
        self.device = Myo(mode=emg_mode.RAW)
        self.device.add_imu_handler(self.store_data)
        self.device.connect()  # device on

        # device parameter
        self.recording: bool = False
        self.remain_time: float = 0.0

        # storage
        self.storage = []


    def store_data(self, quat: tuple, gyro: tuple, accl: tuple):
        if self.show:
            print(quat, gyro, accl)
        if self.recording:
            data = ()
            if self.gyro:
                data += gyro
            if self.quat:
                data += quat
            if self.accl:
                data += accl

            self.storage.append(data)

    def empty(self):
        self.storage = []

    def run(self, recording: bool, remain_time: float):
        self.recording = recording
        self.remain_time = remain_time
        self.device.run()

    def close(self):
        self.device.disconnect()

    def load_data(self):
        return self.storage

