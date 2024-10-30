import numpy as np
import cv2 as cv
import pickle
import os


def load_data(sub, cls, idx, base_path="../data"):
    """
    Load frame and event data from the given subject, class, and index.

    Parameters:
    - sub: Subject ID.
    - cls: Class ID.
    - idx: File index.
    - base_path: Base path for the data folder.

    Returns:
    - frames: Loaded frames (numpy array).
    - events: Loaded events (list).
    """
    path = os.path.join(base_path, f"{sub:03}", f"SubClsIdx_{sub:03}_{cls:03}_{idx:03}")

    # Load frames
    frames = np.load(path + "_frm.npy")

    # Load event data
    with open(path + "_evt.pkl", "rb") as f:
        events = pickle.load(f)

    return frames, events


def display_video(frames, events, frame_enabled=True, event_enabled=True, stop_point=None):
    """
    Display the video of frames and events with optional control over frame and event display.

    Parameters:
    - frames: Frame data (3D numpy array).
    - events: Event data (list of event frames).
    - frame_enabled: Flag to enable/disable frame display.
    - event_enabled: Flag to enable/disable event display.
    - stop_point: Frame index where the video should stop. If None, play all frames.
    """
    cv.namedWindow("Preview", cv.WINDOW_NORMAL)

    for idx, (frame, event) in enumerate(zip(frames, events)):
        if stop_point is not None and idx == stop_point:
            wait_for_exit()

        if not frame_enabled:
            frame = np.zeros_like(frame)  # Remove frame by setting to black

        if event_enabled:
            frame = overlay_events_on_frame(frame, event)
        cv.imshow("Preview", frame)
        if cv.waitKey(100) == 27:  # ESC to exit
            break

    cv.destroyAllWindows()


def overlay_events_on_frame(frame, event):
    """
    Overlay event data onto the given frame.

    Parameters:
    - frame: The original frame to overlay the events on.
    - event: List of events with (timestamp, x, y, polarity).

    Returns:
    - Modified frame with event overlay.
    """
    for _, x, y, polar in event:
        if polar:  # Polarity positive -> set to green channel
            frame[y, x, 1] = 255  # Green channel
        else:  # Polarity negative -> set to red channel
            frame[y, x, 2] = 255  # Red channel
    return frame


def wait_for_exit():
    """
    Wait until the ESC key is pressed to exit.
    """
    while True:
        if cv.waitKey(100) == 27:  # ESC to exit
            exit(0)


def run_visualization(sub, cls, idx, frame_enabled=True, event_enabled=True, stop_point=None):
    """
    Run the full visualization pipeline by loading data and displaying the video.

    Parameters:
    - sub: Subject ID.
    - cls: Class ID.
    - idx: File index.
    - frame_enabled: Enable or disable frame display.
    - event_enabled: Enable or disable event display.
    - stop_point: Index to stop the video display (optional).
    """
    frames, events = load_data(sub, cls, idx)
    display_video(frames, events, frame_enabled, event_enabled, stop_point)




if __name__ == "__main__":
    # 설정 값
    SUB = 1
    CLS = 1
    IDX = 1
    FRAME_ENABLED = False
    EVENT_ENABLED = True
    STOP_POINT = None  # None or set a specific frame index to stop

    # 시각화 실행
    run_visualization(SUB, CLS, IDX, FRAME_ENABLED, EVENT_ENABLED, STOP_POINT)