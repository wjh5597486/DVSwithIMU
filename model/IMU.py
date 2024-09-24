from pyomyo import Myo, emg_mode

class IMU:
    def __init__(self, args):
        # Hyperparameters to control the saved data
        self.gyro = args.gyro
        self.quat = args.quat
        self.accl = args.accl
        self.show = True

        # Myo device initialization
        self.device = Myo(mode=emg_mode.RAW)
        self.device.add_imu_handler(self.store_data)
        self.device.connect()  # Connect to the Myo device

        # Device variables for recording state
        self.recording = False
        self.storage = []

    def store_data(self, quat: tuple, gyro: tuple, accl: tuple):
        """Handler function to store IMU data when recording."""
        if self.show:
            print(f"Quaternion: {quat}, Gyroscope: {gyro}, Accelerometer: {accl}")
        if self.recording:
            # Selectively store gyro, quat, accl based on flags
            data = tuple()
            if self.gyro:
                data += gyro
            if self.quat:
                data += quat
            if self.accl:
                data += accl

            self.storage.append(data)

    def empty(self):
        """Clear the stored IMU data."""
        self.storage.clear()

    def run(self, recording: bool, remain_time: float):
        """Run the Myo device if recording is enabled."""
        self.recording = recording
        self.device.run()

    def close(self):
        """Disconnect the Myo device."""
        self.device.disconnect()

    def load_data(self):
        """Return the stored IMU data."""
        return self.storage
