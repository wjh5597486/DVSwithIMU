import argparse
import os
import cv2 as cv
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--test_id', type=int, default=0)
parser.add_argument('--subject', type=int, default=11)
parser.add_argument('--action', type=int, default=1)
parser.add_argument('--file_idx', type=int, default=1)
parser.add_argument('--n', type=int, default=10)
parser.add_argument('--h', type=int, default=5)
parser.add_argument('--w', type=int, default=2)
parser.add_argument('--scale', type=int, default=0.5)

parser.add_argument('--data_path', type=str, default="./data")
parser.add_argument('--gen_path', type=str, default="./data_gen")
parser.add_argument('--fps', type=int, default=10)

args = parser.parse_args()
assert args.h * args.w == args.n, "H multiply N must equal W"

class_to_text = {
    1: "Standing",
    2: "Walking",
    3: "Running",
    4: "Jumping",
    5: "Throwing",
    6: "Punching",
    7: "Kicking",
    8: "Hands up",
    9: "Squirt",
    10: "Bowing"
}


def put_text_on_video(video, text, org, fontFace,
                      fontScale, color, thickness):
    t, h, w, c = video.shape
    for i in range(t):
        cv.putText(video[i], text, org=org, fontFace=fontFace,
                              fontScale=fontScale, color=color, thickness=thickness)


def put_outline_on_video(video, color, thickness):
    t, h, w, c = video.shape
    for i in range(t):
        cv.rectangle(video[i], (0, 0), (w, h), color=color, thickness=thickness)

def rescale_video(video, scale):
    t, h, w, c = video.shape
    resized_h, resized_w = int(h * scale), int(w * scale)
    out_video = np.zeros((t, resized_h, resized_w, c), dtype=np.uint8)
    for i in range(t):
        resized_frame = cv.resize(video[i], (resized_w, resized_h), interpolation=cv.INTER_AREA)
        out_video[i] = resized_frame
    return out_video

# image call
sub = args.subject
cls = args.action
data_path = args.data_path
sub_folder = f'{sub:03}'
gen_path = args.gen_path


videos = []
idx = args.file_idx
for i in range(idx, idx + args.n):
    # load input
    file_name = f"SubClsIdx_{sub:03}_{cls:03}_{i:03}"
    event_path = os.path.join(data_path, sub_folder, file_name + "_evt.npy")
    frame_path = os.path.join(data_path, sub_folder, file_name + "_frm.npy")
    class_path = os.path.join(data_path, sub_folder, file_name + "_cls.npy")

    # load output
    output_event_path = os.path.join(gen_path, sub_folder, file_name + "_evt.npy")
    output_class_path = os.path.join(gen_path, sub_folder, file_name + "_cls.npy")

    input_event = np.load(event_path)
    input_frame = np.load(frame_path)
    input_class = np.load(class_path)
    output_event = np.load(output_event_path)
    output_class = np.load(output_class_path)

    # rescaling
    t, h, w, c = input_event.shape
    input_frame = rescale_video(input_frame, args.scale)
    input_event = rescale_video(input_event, args.scale)
    output_event = rescale_video(output_event, args.scale)

    input_class_text = class_to_text[input_class.item()]
    output_class_text = class_to_text[output_class.item()]
    color = (255, 0, 0) if input_class_text == output_class_text else (0, 0, 255)

    # put text on video,  frame[nothing], event[class], gen_event[class]
    put_text_on_video(input_event, input_class_text, org=(10, 40), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                      fontScale=0.8, color=color, thickness=2)
    put_text_on_video(output_event, output_class_text, org=(10, 40), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                      fontScale=0.8, color=color, thickness=2)

    # Concatenate Videos horizontally
    video = np.concatenate((input_frame, input_event, output_event), axis=2)

    # Draw outlines
    put_outline_on_video(video, color, thickness=3)

    # Append videos
    videos.append(video)

# Arrange videos
videos = np.array(videos)
n, t, h, w, c = videos.shape
videos = videos.reshape(args.h, args.w, t, h, w, c)
videos = np.transpose(videos, (2, 0, 3, 1, 4, 5))
videos = videos.reshape(t, args.h * h, args.w * w, c)

index = 0  # 현재 프레임 인덱스
while True:
    cv.imshow('Video', videos[index])
    if cv.waitKey(1000 // args.fps) == 27:
        break

    index += 1
    if index >= t:
        index = 0

cv.destroyAllWindows()