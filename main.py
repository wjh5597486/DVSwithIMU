import argparse
from model.Controller import Controller
from model.DVS import DVS
from model.IMU import IMU

parser = argparse.ArgumentParser(description="Controller, DVS, and IMU Options")

# Controller options
parser.add_argument("--save_imu", action="store_true", help="Flag to save IMU data, default is False")
parser.add_argument("--subject", type=int, default=1, help="Subject number")
parser.add_argument("--action", type=int, default=1, help="Action number")
parser.add_argument("--record_interval", type=float, default=3, help="Waiting time before recording")
parser.add_argument("--record_duration", type=float, default=3, help="Recording time in seconds")
parser.add_argument("--repeat", type=int, default=10, help="Number of times to repeat recording")
parser.add_argument("--save_path", type=str, default="./data/", help="Path to save data")
parser.add_argument("--file_idx", type=int, default=1, help="Starting file index")

# DVS options
parser.add_argument("--fps", type=int, default=10, help="Frames per second for DVS")
parser.add_argument("--noise_filter", action="store_true", help="Enable Background Activity Noise Filter")
parser.add_argument("--decay_filter", action="store_true", help="Enable Fast Decay Noise Filter")
parser.add_argument("--max_event_num", type=int, default=20000, help="Maximum number of events per batch")
parser.add_argument("--display_scale", type=float, default=1.5, help="Scale factor for DVS display")

# IMU options
parser.add_argument("--gyro", action="store_true", help="Enable gyroscope data collection")
parser.add_argument("--quat", action="store_true", help="Enable quaternion data collection")
parser.add_argument("--accl", action="store_true", help="Enable acceleration data collection")

# Parse arguments
args = parser.parse_args()

# Separate arguments into different categories for clarity
ctr_args = argparse.Namespace(
    save_imu=args.save_imu,
    subject=args.subject,
    action=args.action,
    record_interval=args.record_interval,
    record_duration=args.record_duration,
    repeat=args.repeat,
    save_path=args.save_path,
    file_idx=args.file_idx
)

imu_args = argparse.Namespace(
    gyro=args.gyro,
    quat=args.quat,
    accl=args.accl
)

dvs_args = argparse.Namespace(
    fps=args.fps,
    noise_filter=args.noise_filter,
    decay_filter=args.decay_filter,
    max_event_num=args.max_event_num,
    display_scale=args.display_scale
)

# Initialize and start the controller
imu = IMU(imu_args) if ctr_args.save_imu else None
dvs = DVS(dvs_args)
controller = Controller(ctr_args=ctr_args, imu=imu, dvs=dvs)
controller.start()
