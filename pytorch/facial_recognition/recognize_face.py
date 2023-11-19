import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def extract_faces(img_file):
    import cv2
    img = cv2.imread(img_file)
    faces = app.get(img)

    face_bbx = []
    eyes = []
    for i in range(len(faces)):
        face = faces[i]
        box = face.bbox.astype(int)
        face_bbx.append(box)
        if face.kps is not None:
            kps = face.kps.astype(int)
            # print(landmark.shape)
            rEye = kps[0]
            lEye = kps[1]
            eyes.append([rEye, lEye])


    return face_bbx, eyes

def draw_on(img, faces):
        import cv2
        dimg = img.copy()
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(int)
            color = (0, 0, 255)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
            if face.kps is not None:
                kps = face.kps.astype(int)
                #print(landmark.shape)
                for l in range(kps.shape[0]):
                    if l > 1:
                        continue
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color,
                               2)
            if face.gender is not None and face.age is not None:
                cv2.putText(dimg,'%s,%d'%(face.sex,face.age), (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)

            #for key, value in face.items():
            #    if key.startswith('landmark_3d'):
            #        print(key, value.shape)
            #        print(value[0:10,:])
            #        lmk = np.round(value).astype(np.int)
            #        for l in range(lmk.shape[0]):
            #            color = (255, 0, 0)
            #            cv2.circle(dimg, (lmk[l][0], lmk[l][1]), 1, color,
            #                       2)
        return dimg

if __name__ == '__main__':
    face_bbx, eyes = extract_faces('out_0.jpg')

    print(face_bbx, eyes)
