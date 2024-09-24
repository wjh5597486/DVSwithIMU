import os
import pickle
import numpy as np
import threading
from datetime import datetime
import platform
import winsound


def beep(sound="Tink"):
    """Play beep sound based on OS platform."""
    if platform.system() == 'Windows':
        winsound.Beep(1000, 200)  # Frequency 1000Hz, duration 200ms
    else:
        os.system(f'afplay /System/Library/Sounds/{sound}.aiff')


def beep_sound(sound="Tink"):
    """Multithreaded beep sound execution to avoid blocking the main thread."""
    threading.Thread(target=beep, args=(sound,), daemon=True).start()


def get_save_path(save_path, subject_id, action, file_idx, data_type, prefix="SubClsIdx"):
    """
    Generate a path for saving specific data (event, frame, imu) with a suffix.
    data_type can be "evt", "frm", or "imu" to generate appropriate file names.
    """
    save_folder_path = os.path.join(save_path, f"{subject_id:03}")
    os.makedirs(save_folder_path, exist_ok=True)
    if data_type == "evt":
        file_path = os.path.join(save_folder_path, f'{prefix}_{subject_id:03}_{action:03}_{file_idx:03}_{data_type}.pkl')
    else:  # frame and imu
        file_path = os.path.join(save_folder_path, f'{prefix}_{subject_id:03}_{action:03}_{file_idx:03}_{data_type}.npy')
    return file_path


def save_data_part(data, file_path, data_type):
    """
    Save a single part of data (event, frame, imu) to a file with a suffix.
    data_type: "evt", "frm", "imu" specifies the type of data being saved.
    """
    try:
        if data_type == "evt":
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        else:
            np.save(file_path, np.stack(data))

        print(f"Data saved at {file_path}")

    except Exception as e:
        print(f"Error saving {data_type} data: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise RuntimeError(f"Error occurred while saving {data_type} data.") from e


def get_valid_file_idx(save_path, subject_id, action, file_idx, data_types, prefix="SubClsIdx"):
    """
    Increment file_idx until a valid, non-existing file index is found for all data types.
    """
    while True:
        paths = [get_save_path(save_path, subject_id, action, file_idx, dt, prefix) for dt in data_types]
        if not any(os.path.exists(path) for path in paths):
            break
        file_idx += 1
    return file_idx


class Controller:
    def __init__(self, ctr_args, imu=None, dvs=None):
        """Initialize Controller with optional IMU and DVS."""
        self.imu = imu
        self.dvs = dvs

        # Recording parameters
        self.record_duration = ctr_args.record_duration
        self.record_interval = ctr_args.record_interval
        self.repeat = ctr_args.repeat
        self.save_path = ctr_args.save_path
        self.subject = ctr_args.subject
        self.action = ctr_args.action
        self.file_idx = ctr_args.file_idx

    def run_recording(self):
        """Run the data collection process for IMU and DVS."""
        try:
            start_time, record_start_time, record_end_time = self._calculate_times()

            beep_pending = True
            cur_time = datetime.now().timestamp()

            while cur_time < record_end_time:
                cur_time = datetime.now().timestamp()
                recording, remain_time = self._calculate_recording_status(cur_time, record_start_time, record_end_time)

                if beep_pending and recording:
                    beep_pending = False
                    beep_sound("Tink")

                if self.dvs:
                    self.dvs.run(recording, remain_time)
                if self.imu:
                    self.imu.run(recording, remain_time)

            beep_sound("Pop")

        except Exception as e:
            raise RuntimeError("Error occurred during recording.") from e

    def save_data(self):
        """Save the recorded data with rollback on failure."""
        save_paths = []  # Track the paths to roll back in case of failure

        try:
            # Find a valid file index where no files exist for the current subject and action
            self.file_idx = get_valid_file_idx(self.save_path, self.subject, self.action, self.file_idx, ["evt", "frm", "imu"])

            # Save each part of data separately with suffixes
            if self.dvs:
                event_path = get_save_path(self.save_path, self.subject, self.action, self.file_idx, "evt")
                frame_path = get_save_path(self.save_path, self.subject, self.action, self.file_idx, "frm")
                save_data_part(self.dvs.load_data()[0], event_path, "evt")
                save_data_part(self.dvs.load_data()[1], frame_path, "frm")
                save_paths.extend([event_path, frame_path])

            if self.imu:
                imu_path = get_save_path(self.save_path, self.subject, self.action, self.file_idx, "imu")
                save_data_part(self.imu.load_data(), imu_path, "imu")
                save_paths.append(imu_path)

        except Exception as e:
            print(f"Saving failed, rolling back files: {e}")
            for path in save_paths:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"Rolled back {path}")
            raise RuntimeError("Rolling back due to save error.") from e

        else:
            self.file_idx += 1

    def start(self):
        """Start the data collection process and repeat as specified."""
        while self.repeat > 0:
            try:
                self.run_recording()  # Record data
            except RuntimeError as e:
                print(f"Recording failed: {e}")
                # Skip saving and continue to the next iteration
            else:
                self.save_data()  # Save data only if recording was successful

            # Empty data buffers if the devices exist
            if self.imu:
                self.imu.empty()
            if self.dvs:
                self.dvs.empty()

            self.repeat -= 1

        beep_sound("Purr")

    def _calculate_times(self):
        """Helper method to calculate start, record, and end times."""
        start_time = datetime.now().timestamp()
        record_start_time = start_time + self.record_interval
        record_end_time = record_start_time + self.record_duration + 0.2
        return start_time, record_start_time, record_end_time

    def _calculate_recording_status(self, cur_time, record_start_time, record_end_time):
        """Helper method to determine recording status and remaining time."""
        recording = cur_time >= record_start_time
        remain_time = record_end_time - cur_time if recording else record_start_time - cur_time
        return recording, remain_time

