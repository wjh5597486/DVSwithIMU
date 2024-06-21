from model.Controller import Controller
import argparse

ctr_parser = argparse.ArgumentParser(description="Control Options")
imu_parser = argparse.ArgumentParser(description="IMU Optons")
dvs_parser = argparse.ArgumentParser(description="Frame and Event Options")
# Add arguments to control options group
ctr_parser.add_argument("--save_imu", type=bool, default=True)

ctr_parser.add_argument("--subject", type=int, default=10, help="")
ctr_parser.add_argument("--action", type=int, default=1, help="")
ctr_parser.add_argument("--file_idx", type=int, default=1, help="It automatically checks and increases")
ctr_parser.add_argument("--record_interval", type=float, default=3, help="waiting time before recording")
ctr_parser.add_argument("--record_duration", type=float, default=3, help="record time")
ctr_parser.add_argument("--save_path", type=str, default="./data/")
ctr_parser.add_argument("--repeat", type=int, default=10, help="Repeat time")


# Add arguments to IMU options group
imu_parser.add_argument("--gyro", type=bool, default=True, help="Gyroscope")
imu_parser.add_argument("--quat", type=bool, default=True, help="Quaternion")
imu_parser.add_argument("--accl", type=bool, default=True, help="Acceleration")


# Add arguments to DVS options group
dvs_parser.add_argument("--fps", type=int, default=10, help="Description of DVS option 1")
dvs_parser.add_argument("--noise_filter", type=bool, default=False, help="BackgroundActivityNoiseFilter")
dvs_parser.add_argument("--decay_filter", type=bool, default=False, help="FastDecayNoiseFilter")
dvs_parser.add_argument("--max_event_num", type=int, default=20000, help="the maximum number of events per batch")
dvs_parser.add_argument("--display_scale", type=float, default=4)

ctr_parser = ctr_parser.parse_args()
imu_parser = imu_parser.parse_args()
dvs_parser = dvs_parser.parse_args()


controller = Controller(ctr_args=ctr_parser,
                        imu_args=imu_parser,
                        dvs_args=dvs_parser)

controller.start()

