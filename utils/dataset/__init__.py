import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def _get_path_list(folder_name="data/DVS"):
    frm_path_list = []  # frame
    imu_path_list = []  # imu
    evt_path_list = []  # event
    cls_path_list = []  # class label
    sub_list = []  # subject list

    sub_folders = os.listdir(folder_name)
    for sub_name in sub_folders:
        files = os.listdir(os.path.join(folder_name, sub_name))
        for file in files:
            file_path = os.path.join(folder_name, sub_name, file)
            ext = file_path[-7:-4]

            if file_path.endswith(".npy"):
                if ext == "evt":  # event
                    evt_path_list.append(file_path)
                elif ext == "imu":  # imu
                    imu_path_list.append(file_path)
                elif ext == "frm":  # frame
                    frm_path_list.append(file_path)
                elif ext == "cls":  # class
                    cls_path_list.append(file_path)
                    # subject
                    subject = int(file_path.split('_')[1])
                    sub_list.append(subject)

    return (np.array(frm_path_list), np.array(evt_path_list), np.array(imu_path_list),
            np.array(cls_path_list), np.array(sub_list))


def split_data_by_sub(subject_list, frame_path, event_path, imu_path, label_path, sub_list):
    return {
        'frame': frame_path[np.isin(subject_list, sub_list)],
        'event': event_path[np.isin(subject_list, sub_list)],
        'imu': imu_path[np.isin(subject_list, sub_list)],
        'label': label_path[np.isin(subject_list, sub_list)]
    }


# inter와 intra에서 공통적으로 사용할 데이터 로드 함수
def load_data(data_path):
    frame_path, event_path, imu_path, label_path, subject_list = _get_path_list(data_path)
    return frame_path, event_path, imu_path, label_path, subject_list


# 모달리티에 맞는 데이터를 튜플의 형태로 반환
def get_data_by_modalities(data, modal_list):
    return tuple(data[modality] for modality in modal_list)


# inter 모드
def inter(modal_list, train_sub, test_sub, batch_size, data_path='data/DVS'):
    # 데이터 로드
    frame_path, event_path, imu_path, label_path, subject_list = load_data(data_path)

    # Train/Test 데이터 경로 분리
    train_data = split_data_by_sub(subject_list, frame_path, event_path, imu_path, label_path, train_sub)
    test_data = split_data_by_sub(subject_list, frame_path, event_path, imu_path, label_path, test_sub)

    # 입력 모달리티와 출력 모달리티가 유효한지 검증
    for modal in modal_list:
        assert modal in train_data, f"Invalid input modality '{modal}'. Choose from {list(train_data.keys())}"

    # 여러 모달리티를 입력과 출력으로 사용할 수 있도록 처리
    train_data = get_data_by_modalities(train_data, modal_list)
    test_data = get_data_by_modalities(test_data, modal_list)

    # Train/Test 데이터셋 생성
    train_dataset = DataPathSet(train_data)
    test_dataset = DataPathSet(test_data)

    # DataLoader 생성
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    return train_dataloader, test_dataloader


# intra 모드
def intra(modal_list, train_sub, ratio, batch_size, data_path='data/DVS'):
    # 데이터 로드
    frame_path, event_path, imu_path, label_path, subject_list = load_data(data_path)

    # 비율에 따른 데이터 분할
    array = np.random.choice([0, 1], size=len(frame_path), p=[ratio, 1 - ratio])

    # Train 데이터 경로 (비율에 따른 분할)
    train_sub_idx = np.logical_and(np.isin(subject_list, train_sub), array == 0)
    test_sub_idx = np.logical_and(np.isin(subject_list, train_sub), array == 1)

    train_data = {
        'frame': frame_path[train_sub_idx],
        'event': event_path[train_sub_idx],
        'imu': imu_path[train_sub_idx],
        'label': label_path[train_sub_idx]
    }

    # Test 데이터 경로 (비율에 따른 분할)
    test_data = {
        'frame': frame_path[test_sub_idx],
        'event': event_path[test_sub_idx],
        'imu': imu_path[test_sub_idx],
        'label': label_path[test_sub_idx]
    }

    # 입력 모달리티와 출력 모달리티가 유효한지 검증
    for modal in modal_list:
        assert modal in train_data, f"Invalid input modality '{modal}'. Choose from {list(train_data.keys())}"

    # 여러 모달리티를 입력과 출력으로 사용할 수 있도록 처리
    train_data = get_data_by_modalities(train_data, modal_list)
    test_data = get_data_by_modalities(test_data, modal_list)

    # Train/Test 데이터셋 생성
    train_dataset = DataPathSet(train_data)
    test_dataset = DataPathSet(test_data)

    # DataLoader 생성
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_dataloader, test_dataloader


class DataPathSet(Dataset):
    def __init__(self, args):
        super().__init__()
        self.modalities = args  # 여러 모달리티 데이터 저장
        self.len = len(self.modalities[0])  # 첫 모달리티 데이터 길이 사용

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        try:
            # 각 모달리티 데이터에서 해당 index에 있는 값을 반환
            return tuple(np.load(modality[index]) for modality in self.modalities)
        except Exception as e:
            print(f"Error loading data at index {index}: {e}")
            raise

