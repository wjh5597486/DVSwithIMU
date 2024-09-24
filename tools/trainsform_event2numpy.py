# tools/transform_event2npy.py

import numpy as np
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


def transform_event_and_frame_to_npy(frames, events, output_path="output_data"):
    """
    Save the frames and events as 2D numpy files.

    Parameters:
    - frames: Frame data (numpy array).
    - events: Event data (list of event frames).
    - output_path: Path to save the numpy files.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Save the combined frame and event as a numpy array
    for idx, (frame, event) in enumerate(zip(frames, events)):
        # Combine frame and event
        frame_with_event = overlay_events_on_frame(frame.copy(), event)

        # Save frame with event as .npy file
        npy_path = os.path.join(output_path, f"frame_event_{idx:03}.npy")
        np.save(npy_path, frame_with_event)
        print(f"Frame and event saved as {npy_path}")

    print(f"All data saved to {output_path}")


if __name__ == "__main__":
    # Settings
    SUB = 1
    CLS = 1
    IDX = 2
    BASE_PATH = "../data"
    OUTPUT_FILES_DIR = "output_npy_data"

    # Load frame and event data
    frames, events = load_data(SUB, CLS, IDX, BASE_PATH)

    # Save as combined numpy files (frame and event)
    transform_event_and_frame_to_npy(frames, events, OUTPUT_FILES_DIR)
