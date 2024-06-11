import argparse

from model.DVSmodel import DVSmodel

"""
resolution = (260, 346)
"""
"""

"""
subject = {
    2: "Junghwan Lee",
    3: "Geunbo Yang",
    4: "Yoontae Park",
    5: "Jiseok Yang",
    100: "Jihwan Won",

}

idx2class = {
    1: "Standing",
    2: "Walking",
    3: "Running",
    4: "Jumping",
    5: "Clapping",
    6: "Throwing",
    7: "Kicking",
    8: "Punching",
    9: "Squirting",
    10: "Wave with hand",
}

# parser
parser = argparse.ArgumentParser("Record Process")

# parser - save
parser.add_argument("--id", type=int, default="100", help="subject id/number")
parser.add_argument("--action", type=int, default="4", help="action id/number")
parser.add_argument("--i", type=int, default="1", help="start file name this index number")

# parser record setting
parser.add_argument("--fps", type=int, default=10)
parser.add_argument("--record_time", type=float, default=3)
parser.add_argument("--record_interval", type=float, default=2)
parser.add_argument("--max_event_num", type=int, default=20000, help="the maximum number of events per batch")

# parser event filter
parser.add_argument("--noise_filter", type=bool, default=True, help="BackgroundActivityNoiseFilter")
parser.add_argument("--decay_filter", type=bool, default=False, help="FastDecayNoiseFilter")

# parser folder
parser.add_argument("--save_folder", type=str, default="../data", help="Save path")

parser.add_argument("--repeat", type=int, default=10)

args = parser.parse_args()

print(f"Subject: {subject[args.id]}")
print(f"Action : {args.action}, {idx2class[args.action]} ")
print(f"FPS    : {args.action}")
print()

if __name__ == "__main__":
    model = DVSmodel(args)
    model.start()
    model.stop()
