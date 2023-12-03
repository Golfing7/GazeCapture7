import cv2
import numpy as np
import mediapipe as mp
import time
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from extractFrames import get_face_grid, get_frames
from threading import local

detector_storage = local()
# base_options = python.BaseOptions(model_asset_path='detector.tflite')
# options = vision.FaceDetectorOptions(base_options=base_options)
# detector_storage.__dict__.setdefault('detector', vision.FaceDetector.create_from_options(options))


# base_options = python.BaseOptions(model_asset_path='detector.tflite')
# options = vision.FaceDetectorOptions(base_options=base_options)
# detector = vision.FaceDetector.create_from_options(options)

def detect_features(np_img):
    if not hasattr(detector_storage, "detector"):
        detector_storage.base_options = python.BaseOptions(model_asset_path='detector.tflite')
        detector_storage.options = vision.FaceDetectorOptions(base_options=detector_storage.base_options)
        detector_storage.detector = vision.FaceDetector.create_from_options(detector_storage.options)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_img)
    dresult = detector_storage.detector.detect(mp_image)
    if len(dresult.detections) == 0:
        return [np_img, [], []]

    im_width = np_img.shape[1]
    im_height = np_img.shape[0]
    detection = dresult.detections[0]
    right_eye = detection.keypoints[0]
    left_eye = detection.keypoints[1]

    face_bbx = detection.bounding_box
    built_face_bbx = [face_bbx.origin_x, face_bbx.origin_y, face_bbx.width, face_bbx.height]
    face_size = face_bbx.width

    right_eye_px = [math.floor(right_eye.x * im_width), math.floor(right_eye.y * im_height)]
    left_eye_px = [math.floor(left_eye.x * im_width), math.floor(left_eye.y * im_height)]

    eye_ratio = math.ceil(face_size / 8)
    right_eye_bbx = [right_eye_px[0] - eye_ratio, right_eye_px[1] - eye_ratio, eye_ratio * 2, eye_ratio * 2]
    left_eye_bbx = [left_eye_px[0] - eye_ratio, left_eye_px[1] - eye_ratio, eye_ratio * 2, eye_ratio * 2]

    return np_img, [built_face_bbx], [[right_eye_bbx, left_eye_bbx],
                                      get_face_grid(built_face_bbx, im_width, im_height, 25)]


if __name__ == '__main__':
    print("Starting")
    frame_data = get_frames('51_4_4.mp4')
    print("Read all frames...")
    for i, (frame, frame_time) in enumerate(frame_data):
        detect_features(frame)
        if i % 100 == 0:
            print(f"Checkpoint {i}")

    pass