from datetime import datetime
from model.IMU import IMU
from model.DVS import DVS
import os
import pickle
import numpy as np


class Controller:
    def __init__(self, ctr_args, imu_args, dvs_args):
        self.imu = IMU(imu_args)
        self.dvs = DVS(dvs_args)

        # hyperparameters of save
        self.prefix = "SubClsIdx"
        self.id = ctr_args.subject
        self.action = ctr_args.action
        self.file_idx = ctr_args.file_idx

        # hyperparameters
        self.record_duration = ctr_args.record_duration
        self.record_interval = ctr_args.record_interval
        self.repeat = ctr_args.repeat
        self.save_imu = ctr_args.save_imu
        self.save_path = ctr_args.save_path

    def run(self):
        # compute start and end time.
        start_time = datetime.now().timestamp()
        record_start_time = start_time + self.record_interval
        record_end_time = record_start_time + self.record_duration + 0.1

        # run devices by record_end_time
        cur_time = datetime.now().timestamp()
        while cur_time < record_end_time:

            # compute time.
            cur_time = datetime.now().timestamp()

            # compute recoding and time
            recording = bool(cur_time >= record_start_time)  # check record or waiting
            remain_time = record_end_time - cur_time if recording else record_start_time - cur_time

            # run devices
            self.dvs.run(recording, remain_time)
            if self.save_imu:
                self.imu.run(recording, remain_time)


    def save(self):
        """
        load and check IMU and DVS data,
        if the lengths are shorter than the expected lengths.
        the Saving is not allowed and do not count the number of record.
        """
        # load
        event, frame = self.dvs.load_data()
        imu = self.imu.load_data()

        expected_len_frame = self.dvs.fps * self.record_duration  # frame per second * duration
        expected_len_event = self.dvs.fps * self.record_duration  # frame per second * duration
        expected_len_imu = 50 * self.record_duration * self.save_imu  # imu frequency * duration

        # check length of data.
        length_check = (len(frame) >= expected_len_frame, len(event) >= expected_len_event, len(imu) >= expected_len_imu)
        if length_check != (True, True, True):
            return "The lengths of data are not enough"

        # cut the length
        frame = frame[:expected_len_frame]
        event = event[:expected_len_event]
        imu = imu[:expected_len_imu]

        # save
        event_path, frame_path, imu_path = self.get_paths()

        try:
            # save event
            with open(event_path, 'wb') as f:
                pickle.dump(event, f)
            print("Event saved:", event_path)

            # save frame
            frames = np.stack(frame)
            np.save(frame_path, frames)
            print("Frame saved:", frame_path)

            # save imu
            imu_data = np.stack(imu)
            np.save(imu_path, imu_data)
            print("IMU saved:", imu_path)

            return True  # All saves were successful

        except Exception as e:

            # Delete the files that were partially saved
            if os.path.exists(event_path):
                os.remove(event_path)
            if os.path.exists(frame_path):
                os.remove(frame_path)
            if os.path.exists(imu_path):
                os.remove(imu_path)
            return e

    def get_paths(self):
        save_folder_path = self.save_path
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
            print(f'{save_folder_path} has been created.')

        while True:
            assert 0 <= self.file_idx < 1000, "Invalid index range, check your files"
            frame_path = save_folder_path + f'{self.prefix}_{self.id:03}_{self.action:03}_{self.file_idx:03}_frm.npy'
            event_path = save_folder_path + f'{self.prefix}_{self.id:03}_{self.action:03}_{self.file_idx:03}_evt.pkl'
            imu_path = save_folder_path + f'{self.prefix}_{self.id:03}_{self.action:03}_{self.file_idx:03}_imu.npy'
            if not os.path.exists(frame_path) and not os.path.exists(event_path) and not os.path.exists(imu_path):
                break
            self.file_idx += 1
        return event_path, frame_path, imu_path

    def empty(self):
        self.imu.empty()
        self.dvs.empty()

    def start(self):
        # initial check
        self.run()
        if self.imu:
            self.imu.show = False

        # recording
        while self.repeat:
            self.run()
            success_to_save = self.save()  # save
            if not success_to_save:
                print("Error occurred while saving data:", success_to_save)
                self.repeat -= 1
            self.empty()

