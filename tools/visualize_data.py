import numpy as np
import cv2 as cv
import pickle

SUB = 0
CLS = 1
IDX = 5

FRAME = 1
EVENT = 1
stop_point = ...
# idx or None

path = f"../data/{SUB:03}/SubClsIdx_{SUB:03}_{CLS:03}_{IDX:03}"

# load frames
frames = np.load(path + "_frm.npy")
# load data
with open(path + "_evt.pkl", "rb") as f:
    events = pickle.load(f)

cv.namedWindow("Preview", cv.WINDOW_NORMAL)

for idx, (image, event) in enumerate(zip(frames, events)):
    if idx == stop_point:
        while True:
            if cv.waitKey(100) == 27:  # 10:enter, 27: esc, 32: space,
                exit(0)

    if not FRAME:  # remove frame
        image = image * 0

    if EVENT:  # event
        for _, x, y, polar in event:
            if polar:
                image[y][x][1] = 255
            else:
                image[y][x][2] = 255

    cv.imshow("Preview", image)
    if cv.waitKey(100) == 27:  # 10:enter, 27: esc, 32: space,
        exit(0)
