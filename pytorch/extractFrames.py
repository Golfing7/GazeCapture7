import cv2

'''
Parses frames out of a video file for the use of testing the images.
'''

import numpy as np
import cv2
import os
import av

# print(os.environ['OPENCV_DATA'])

cascades_path = os.path.join( os.path.dirname( cv2.__file__ ), 'data/' )
face_cascade = cv2.CascadeClassifier(cascades_path + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cascades_path + 'haarcascade_eye.xml')

eye_l_cascade = cv2.CascadeClassifier(cascades_path + 'haarcascade_lefteye_2splits.xml')
eye_r_cascade = cv2.CascadeClassifier(cascades_path + 'haarcascade_righteye_2splits.xml')

# This code is converted from https://github.com/CSAILVision/GazeCapture/blob/master/code/faceGridFromFaceRect.m

# Given face detection data, generate face grid data.
#
# Input Parameters:
# - frameW/H: The frame in which the detections exist
# - gridW/H: The size of the grid (typically same aspect ratio as the
#     frame, but much smaller)
# - labelFaceX/Y/W/H: The face detection (x and y are 0-based images
#     coordinates)
# - parameterized: Whether to actually output the grid or just the
#     [x y w h] of the 1s square within the gridW x gridH grid.
# https://drive.google.com/drive/folders/1ZcYb4eH2jPndS5nkqQFcLHdGNM9dTF5C?usp=sharing
# https://drive.google.com/file/d/gpip1UdhuJ_bulreFyGa8CziK4tdwXeHC2PIh/view?usp=sharing

def faceGridFromFaceRect(frameW, frameH, gridW, gridH, labelFaceX, labelFaceY, labelFaceW, labelFaceH, parameterized):

    scaleX = gridW / frameW
    scaleY = gridH / frameH

    if parameterized:
      labelFaceGrid = np.zeros(4)
    else:
      labelFaceGrid = np.zeros(gridW * gridH)

    grid = np.zeros((gridH, gridW))

    # Use one-based images coordinates.
    xLo = round(labelFaceX * scaleX)
    yLo = round(labelFaceY * scaleY)
    w = round(labelFaceW * scaleX)
    h = round(labelFaceH * scaleY)

    if parameterized:
        labelFaceGrid = [xLo, yLo, w, h]
    else:
        xHi = xLo + w
        yHi = yLo + h

        # Clamp the values in the range.
        xLo = int(min(gridW, max(0, xLo)))
        xHi = int(min(gridW, max(0, xHi)))
        yLo = int(min(gridH, max(0, yLo)))
        yHi = int(min(gridH, max(0, yHi)))

        faceLocation = np.ones((yHi - yLo, xHi - xLo))
        grid[yLo:yHi, xLo:xHi] = faceLocation

        # Flatten the grid.
        grid = np.transpose(grid)
        labelFaceGrid = grid.flatten()

    return labelFaceGrid

def detect_eyes(face, img, gray):
    [x,y,w,h] = face
    roi_gray = gray[y:y+h, x:x+w]

    eye_l = eye_l_cascade.detectMultiScale(roi_gray)
    eye_r = eye_r_cascade.detectMultiScale(roi_gray)
    # eyes = eye_cascade.detectMultiScale(roi_gray)
    rEye_sorted_by_size = sorted(eye_r, key=lambda x: -x[2])
    lEye_sorted_by_size = sorted(eye_l, key=lambda x: -x[2])
    largest_r = rEye_sorted_by_size[0] if len(rEye_sorted_by_size) > 0 else []
    largest_l = lEye_sorted_by_size[0] if len(lEye_sorted_by_size) > 0 else []
    if len(largest_r) == 0 or len(largest_l) == 0:
        return ()
    # offset by face start
    fo = list(map(lambda eye: [face[0] + eye[0], face[1] + eye[1], eye[2], eye[3]], [largest_r, largest_l]))
    return fo

def get_face_grid(face, frameW, frameH, gridSize):
    faceX,faceY,faceW,faceH = face

    return faceGridFromFaceRect(frameW, frameH, gridSize, gridSize, faceX, faceY, faceW, faceH, True)

def extract_image_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_detections = face_cascade.detectMultiScale(gray, 1.3, 5)

    left_to_right_face_detections = sorted(face_detections, key=lambda detection: detection[0])

    faces = []
    face_features = []
    for [x,y,w,h] in left_to_right_face_detections:
        face = [x, y, w, h]
        eyes = detect_eyes(face, img, gray)
        face_grid = get_face_grid(face, img.shape[1], img.shape[0], 25)

        faces.append(face)
        face_features.append([eyes, face_grid])

    return img, faces, face_features


def crop_to_bounds(img, bounds):
    [x, y, w, h] = bounds
    cropped = img[y:y + h, x:x + w]
    return cropped


def draw_detected_features(img, faces, face_features):
    # eye_images = []
    # for (ex,ey,ew,eh) in eyes:
    #     eye_images.append(np.copy(img[y+ey:y+ey+eh,x+ex:x+ex+ew]))
    for i, face in enumerate(faces):
        [x, y, w, h] = face
        eyes, face_grid = face_features[i]

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        r = 0
        for [ex,ey,ew,eh] in eyes:
            cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,r,0),2)
            r += 255


gridSize = 25

def get_frames(video_file, stream=None):
    """
    Parses all frames out of the given video file and returns an array of PIL images.
    """

    container = av.open(video_file)
    video = container.streams.video[0]

    to_return = []
    for idx, frame in enumerate(container.decode(video)):
        image = cv2.cvtColor(frame.to_rgb().to_ndarray(), cv2.COLOR_RGB2BGR)
        frame_time = float(frame.pts * video.time_base)
        if stream is None or not callable(stream):
            to_return.append([image, frame_time])
        else:
            stream(image, frame_time, idx)
    container.close()

    return to_return

def count_frames(video_file, stream=None):
    """
    Parses all frames out of the given video file and returns an array of PIL images.
    """

    video = cv2.VideoCapture(video_file)
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    return total

# Total Videos: 817
# Total Frames: 2574661
# Total Usable Frames: 2087224

if __name__ == '__main__':
    import h5py
    total_used_frames = 0
    with h5py.File("preprocessed.h5") as f:
        for subject in range(1, 52):
            for trial in range(1, 5):
                for pose in range(1, 5):
                    data = f.get(f"{subject}/{trial}_{pose}/frame_indices")
                    total_used_frames += len(data)
    
    print(total_used_frames)

    total_frames = 0
    total_videos = 0
    for f in os.listdir('TabletGazeDataset'):
        if not os.path.isdir(os.path.join('TabletGazeDataset', f)):
            continue
            
        for sub in os.listdir(os.path.join('TabletGazeDataset', f)):
            if not sub.endswith(".mp4"):
                continue
            total_videos += 1
            total_frames += count_frames(os.path.join("TabletGazeDataset", f, sub))

    print(f"Total Videos: {total_videos}")
    print(f"Total Frames: {total_frames}")
    pass