import cv2

'''
Parses frames out of a video file for the use of testing the images.
'''

import numpy as np
import cv2
import time
import os
import scipy.io as sio

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
# - labelFaceX/Y/W/H: The face detection (x and y are 0-based image
#     coordinates)
# - parameterized: Whether to actually output the grid or just the
#     [x y w h] of the 1s square within the gridW x gridH grid.

def faceGridFromFaceRect(frameW, frameH, gridW, gridH, labelFaceX, labelFaceY, labelFaceW, labelFaceH, parameterized):

    scaleX = gridW / frameW
    scaleY = gridH / frameH

    if parameterized:
      labelFaceGrid = np.zeros(4)
    else:
      labelFaceGrid = np.zeros(gridW * gridH)

    grid = np.zeros((gridH, gridW))

    # Use one-based image coordinates.
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
    start_ms = (time.time_ns() / 1000000)
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

def get_frames(video_file):
    """
    Parses all frames out of the given video file and returns an array of PIL images.
    """

    video = cv2.VideoCapture(video_file)
    success, image = video.read()
    read_frames = []
    frames_read = 0
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    while success:
        frame_time = frames_read / frame_rate
        read_frames.append([image, frame_time])
        success, image = video.read()
        frames_read += 1
    return read_frames

import predict

if __name__ == '__main__':

    data = sio.loadmat('gazePts.mat', squeeze_me=True, struct_as_record=True)
    start_time = sio.loadmat('startTime.mat', squeeze_me=True, struct_as_record=True)
    print(start_time['startTime'])
    for data_pt in data['gazePts'].tolist():
        print(data_pt)

    frames = get_frames("../data/51_4_4.mp4")
    count = 0
    # for frame, f_time in frames:
    #     data = extract_image_features(frame)
    #     if len(data[1]) > 0 and len(data[2][0]) >= 2:
    #         draw_detected_features(frame, data[1], data[2])
    #         cv2.imwrite(f'out{count}.jpg', frame)
    #         print('wrote')
    #         count += 1