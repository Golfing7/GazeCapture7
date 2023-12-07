import argparse
import pathlib

import h5py
import numpy as np
import tqdm
import cv2
from concurrent.futures import ThreadPoolExecutor
import threading

from recognize_face import detect_features


def add_mat_data_to_hdf5(person_id: str, dataset_dir: pathlib.Path,
                         output_path: pathlib.Path, sem: threading.Semaphore) -> None:
    with h5py.File(dataset_dir / f'{person_id}.mat', 'r') as f_input:
        images = f_input.get('Data/data')[()]
        labels = f_input.get('Data/label')[()][:, :4]
    assert len(images) == len(labels) == 3000

    images = images.transpose(0, 2, 3, 1).astype(np.uint8)
    
    poses = labels[:, 2:]
    gazes = labels[:, :2]

    filtered_images = []
    filtered_poses = []
    filtered_gazes = []
    eyes = []
    for i, image in tqdm.tqdm(enumerate(images), desc="Eye processing"):
        im = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img, face_bbx, eye_features = detect_features(im)
        if len(face_bbx) == 0:
            continue

        right_eye = eye_features[0][0]
        left_eye = eye_features[0][1]
        
        eyes.append([np.array(right_eye).astype(np.uint16), np.array(left_eye).astype(np.uint16)])
        filtered_images.append(images[i])
        filtered_poses.append(poses[i])
        filtered_gazes.append(gazes[i])

    sem.acquire()
    with h5py.File(output_path, 'a') as f_output:
        f_output.create_dataset(f'{person_id}/count', data=len(filtered_images))
        for index, (image, gaze, eye_pair,
                    pose) in tqdm.tqdm(enumerate(zip(filtered_images, filtered_gazes, eyes, filtered_poses)),
                                       leave=False):
            f_output.create_dataset(f'{person_id}/image/{index:04}',
                                    data=image)
            f_output.create_dataset(f'{person_id}/eyes/{index:04}',
                                    data=eye_pair)
            f_output.create_dataset(f'{person_id}/pose/{index:04}', data=pose)
            f_output.create_dataset(f'{person_id}/gaze/{index:04}', data=gaze)
    sem.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output-dir', '-o', type=str, required=True)
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / 'MPIIFaceGaze.h5'
    if output_path.exists():
        raise ValueError(f'{output_path} already exists.')

    dataset_dir = pathlib.Path(args.dataset)
    sem = threading.Semaphore()
    all_futures = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for person_id in range(15):
            person_id = f'p{person_id:02}'
            future = executor.submit(add_mat_data_to_hdf5, person_id, dataset_dir, output_path, sem)
            all_futures.append(future)
        
        with tqdm.tqdm(total=15) as pbar:
            def update(thing):
                nonlocal pbar
                pbar.update(1)
                return
            future.add_done_callback(update)


if __name__ == '__main__':
    main()