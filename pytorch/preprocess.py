import h5py
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import gdown
import tqdm
import os

from ITrackerData import TabletGazePreprocessData


def preprocess_dataset(dataset_path, output_path='preprocessed.h5'):
    """
    Produces a .h5 file with annotated data.
    """

    # Create the data loader and load all frames.
    loader = torch.utils.data.DataLoader(TabletGazePreprocessData(dataset_path), pin_memory=True, shuffle=False)

    # if os.path.exists(output_path):
    #     os.remove(output_path)

    for video_data in tqdm.tqdm(loader):
        face = []
        eye_l = []
        eye_r = []
        metadata_record = []
        frame_times = []
        frame_indices = []
        frame_group = video_data[0]
        subject = video_data[1].item()
        trial = video_data[2].item()
        pose = video_data[3].item()
        for frame_data in frame_group:
            imFace, imEyeL, imEyeR, faceGrid, frame_time, index = frame_data
            face.append(imFace.numpy().copy()[0])
            eye_l.append(imEyeL.numpy().copy()[0])
            eye_r.append(imEyeR.numpy().copy()[0])
            frame_times.append(frame_time[0].item())
            metadata_record.append(faceGrid.numpy().copy()[0])
            frame_indices.append(index.numpy().copy()[0])

        face_encode = np.array(face).astype(np.uint16)
        eye_l_encode = np.array(eye_l).astype(np.uint16)
        eye_r_encode = np.array(eye_r).astype(np.uint16)
        metadata_encode = np.array(metadata_record).astype(np.float32)
        frame_indices_encode = np.array(frame_indices).astype(np.uint32)
        frame_times_encode = np.array(frame_times).astype(np.float32)
        with h5py.File(output_path, 'a') as output:
            output.create_dataset(f'{subject}/{trial}_{pose}/faces', data=face_encode)
            output.create_dataset(f'{subject}/{trial}_{pose}/eyes_l', data=eye_l_encode)
            output.create_dataset(f'{subject}/{trial}_{pose}/eyes_r', data=eye_r_encode)
            output.create_dataset(f'{subject}/{trial}_{pose}/metadata', data=metadata_encode)
            output.create_dataset(f'{subject}/{trial}_{pose}/frame_indices', data=frame_indices_encode)
            output.create_dataset(f'{subject}/{trial}_{pose}/frame_times', data=frame_times_encode)


if __name__ == '__main__':
    preprocess_dataset('TabletGazeDataset')
