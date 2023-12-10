'''
Parses frames out of a video file for the use of testing the images.
'''

import numpy as np
import cv2
import av

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

def generate_centered_face_grid(gridW, gridH):
    labelFaceGrid = np.zeros(gridW * gridH)
    grid = np.zeros((gridH, gridW))

    xLo = 7
    yLo = 7

    xHi = 18
    yHi = 18

    faceLocation = np.ones((yHi - yLo, xHi - xLo))
    grid[yLo:yHi, xLo:xHi] = faceLocation

    # Flatten the grid.
    grid = np.transpose(grid)
    labelFaceGrid = grid.flatten()

    return labelFaceGrid

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

def get_face_grid(face, frameW, frameH, gridSize):
    faceX,faceY,faceW,faceH = face

    return faceGridFromFaceRect(frameW, frameH, gridSize, gridSize, faceX, faceY, faceW, faceH, True)


def crop_to_bounds(img, bounds):
    [x, y, w, h] = bounds
    cropped = img[y:y + h, x:x + w]
    return cropped


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