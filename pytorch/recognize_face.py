import cv2
import numpy as np
import insightface
import onnxruntime
from insightface.app import FaceAnalysis
from extractFrames import get_face_grid, draw_detected_features
print(onnxruntime.get_device())
print(onnxruntime.__version__)

app = FaceAnalysis(providers=['ROCMExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))


def extract_faces(img):
    faces = app.get(img)

    face_bbx = []
    eyes = []
    for i in range(len(faces)):
        face = faces[i]
        box = face.bbox.astype(int)
        face_bbx.append(box)
        if face.kps is not None:
            kps = face.kps.astype(int)
            rEye = kps[0]
            lEye = kps[1]
            eyes.append([rEye, lEye])

    return face_bbx, eyes


def insight_extract(img):
    if True:
        return [img, [[509, 196, 313, 412]], [[[[551, 307, 78, 78], [704, 309, 78, 78]], [10, 7, 6, 14]]]]
    face_detections, eyes = extract_faces(img)

    faces = []
    face_features = []
    for i in range(len(face_detections)):
        bbx = face_detections[i]
        eye_points = eyes[i]
        face = [bbx[0], bbx[1], bbx[2] - bbx[0], bbx[3] - bbx[1]]
        head_width = bbx[2] - bbx[0]

        # Save eye data
        right_eye = eye_points[0]
        left_eye = eye_points[1]

        # Create eye bounding boxes as a proportion of head width.
        eye_radius = int(head_width / 8)
        right_eye_box = [right_eye[0] - eye_radius, right_eye[1] - eye_radius, eye_radius * 2, eye_radius * 2]
        left_eye_box = [left_eye[0] - eye_radius, left_eye[1] - eye_radius, eye_radius * 2, eye_radius * 2]

        face_grid = get_face_grid(face, img.shape[1], img.shape[0], 25)

        faces.append(face)
        eye_boxes = [right_eye_box, left_eye_box]
        face_features.append([eye_boxes, face_grid])

    return img, faces, face_features


if __name__ == '__main__':
    # face_bbx, eyes = extract_faces(cv2.imread('test.jpg'))
    # cv2.imwrite('out__.jpg', draw_on(cv2.imread('test.jpg'), face_bbx, eyes))
    img, faces, features = insight_extract(cv2.imread('out_0.jpg'))
    print(faces, features)
    # data = insight_extract(cv2.imread('out_0.jpg'))
    draw_detected_features(img, faces, features)
    cv2.imwrite('output.jpg', img)
