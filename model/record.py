import argparse

from model.DVSmodel import DVSmodel

"""
resolution = (260, 346)
"""
"""

"""
subject = {
    1: "Jihwan Won",
    2: "Junghwan Lee",
    3: "Geunbo Yang",
    4: "Yoontae Park",
    5: "Jiseok Yang"
}

idx2class = {
    1: "right circle with hand",
    2: "left circle with hand",
    3: "star with hand",
    4: "wave with hand",
    5: "triangle with hand",
    6: "rectangle with hand",
    7: "V with hand",
    8: "X with hand",
    9: "wave with fingers",
    10: "nothing"
}

# parser
parser = argparse.ArgumentParser("Record Process")

# parser - save
parser.add_argument("--id", type=int, default="1", help="subject id/number")
parser.add_argument("--action", type=int, default="10", help="action id/number")
parser.add_argument("--i", type=int, default="1", help="start file name this index number")

# parser record setting
parser.add_argument("--fps", type=int, default=10)
parser.add_argument("--record_time", type=float, default=3)
parser.add_argument("--record_interval", type=float, default=3)
parser.add_argument("--max_event_num", type=int, default=20000, help="the maximum number of events per batch")

# parser event filter
parser.add_argument("--noise_filter", type=bool, default=True, help="BackgroundActivityNoiseFilter")
parser.add_argument("--decay_filter", type=bool, default=False, help="FastDecayNoiseFilter")

# parser folder
parser.add_argument("--save_folder", type=str, default="../data", help="Save path")

parser.add_argument("--repeat", type=int, default=20)

args = parser.parse_args()

print(f"Subject: {subject[args.id]}")
print(f"Action : {args.action}, {idx2class[args.action]} ")
print(f"FPS    : {args.action}")
print()

if __name__ == "__main__":
    model = DVSmodel(args)
    model.start()
    model.stop()
