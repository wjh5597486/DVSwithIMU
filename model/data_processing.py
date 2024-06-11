import os
import pickle

import numpy as np
import torch
from tqdm import tqdm


def data_processing(folder_name="./data/"):
    x_event = []
    x_frame = []
    y_label = []

    sub_folders = os.listdir(folder_name)
    for sub_name in sub_folders:
        files = os.listdir(os.path.join(folder_name, sub_name))

        for file in tqdm(files):
            frame = None
            label = None
            event = None

            file_path = os.path.join(folder_name, sub_name, file)

            """
            이거 더이상 동작안함... 사용 금지..
            :이벤트를 받아오지 않을 거임..
            """
            # frame and label
            if file_path[-3:] == 'npy':
                # frame
                frame = np.load(file_path)

                # label
                label = file_path[-11:-8]

            # event
            elif file_path[-3:] == 'pkl':
                event = np.zeros((30, 260, 346, 2))
                with open(file_path, 'rb') as f:
                    events = pickle.load(f)
                    for t, event_t in enumerate(events):
                        for event_unit in event_t:
                            time, w, h, polarity = event_unit
                            event[t, h, w, polarity] += 1

            # append
            if event is not None:
                x_event.append(event)
            if frame is not None:
                x_frame.append(frame)
            if label is not None:
                y_label.append(int(label))

    # transform
    x_frame = torch.tensor(np.stack(x_frame), dtype=torch.float)
    y_label = torch.tensor(np.stack(y_label), dtype=torch.long)
    x_event = torch.tensor(np.stack(x_event), dtype=torch.float)

    return x_frame, x_event, y_label


if __name__ == "__main__":
    x_frame, x_event, y_label = data_processing()

    torch.save(x_frame, '../data_processed/x_frame.pth')
    torch.save(x_event, '../data_processed/x_event.pth')
    torch.save(y_label, '../data_processed/y_label.pth')

    print("fin")
