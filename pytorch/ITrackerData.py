import cv2
import torch.utils.data as data
import scipy.io as sio
from PIL import Image
import os
import os.path
import torchvision.transforms as transforms
import torch
import numpy as np
import math
import extractFrames, recognize_face
import re

'''
Data loader for the iTracker.
Use prepareDataset.py to convert the dataset from http://gazecapture.csail.mit.edu/ to proper format.

Author: Petr Kellnhofer ( pkel_lnho (at) gmai_l.com // remove underscores and spaces), 2018. 

Website: http://gazecapture.csail.mit.edu/

Cite:

Eye Tracking for Everyone
K.Krafka*, A. Khosla*, P. Kellnhofer, H. Kannan, S. Bhandarkar, W. Matusik and A. Torralba
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016

@inproceedings{cvpr2016_gazecapture,
Author = {Kyle Krafka and Aditya Khosla and Petr Kellnhofer and Harini Kannan and Suchendra Bhandarkar and Wojciech Matusik and Antonio Torralba},
Title = {Eye Tracking for Everyone},
Year = {2016},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}

'''

MEAN_PATH = './'

def loadMetadata(filename, silent = False, struct_as_record = False):
    try:
        # http://stackoverflow.com/questions/6273634/access-array-contents-from-a-mat-file-loaded-using-scipy-io-loadmat-python
        if not silent:
            print('\tReading metadata from %s...' % filename)
        metadata = sio.loadmat(filename, squeeze_me=True, struct_as_record=struct_as_record)
    except:
        print('\tFailed to read the meta file "%s"!' % filename)
        return None
    return metadata

class SubtractMean(object):
    """Normalize an tensor images with mean.
    """

    def __init__(self, meanImg):
        self.meanImg = transforms.ToTensor()(meanImg / 255)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor images of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized images.
        """       
        return tensor.sub(self.meanImg)

class TabletGazeData(data.Dataset):
    def __init__(self, dataPath, split='all', imSize=(224, 224), gridSize=(25, 25)):
        """
        The tablet gaze dataset has extra metadata for each video...

        The first, stored in gazePts.mat, stores the order and location of the dot appearance for ALL trials.
        The second, stored in startTime.mat, stores the time (in seconds)? that the trial begins. (Discard all frames prior to this timestamp)

        The trial was conducted with 35 dot positions. After the start time, every 3 seconds the dot changes positions.
        """
        self.data_path = dataPath
        self.imSize = imSize
        self.gridSize = gridSize

        print('Loading TabletGaze dataset...')

        self.gaze_pts = loadMetadata(os.path.join(dataPath, 'gazePts.mat'), struct_as_record=True)['gazePts'].tolist()
        self.gaze_points_x = self.gaze_pts[2]
        self.gaze_points_y = self.gaze_pts[3]
        self.startTime = loadMetadata(os.path.join(dataPath, 'startTime.mat'), struct_as_record=True)['startTime'].tolist()

        self.faceMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_face_224.mat'))['image_mean']
        self.eyeLeftMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_left_224.mat'))['image_mean']
        self.eyeRightMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_right_224.mat'))['image_mean']

        self.transformFace = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.faceMean),
        ])
        self.transformEyeL = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.eyeLeftMean),
        ])
        self.transformEyeR = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.eyeRightMean),
        ])

        self.subjects = 1
        self.screenWidthCM = 22.62
        self.screenHeightCM = 14.14

        self.screenCenterXCM = self.screenWidthCM / 2
        self.screenCenterYCM = self.screenHeightCM / 2

        if split == 'train':
            subject_split = range(0, math.ceil(self.subjects * 0.8)) # Use 80% of the set for training
        elif split == 'test':
            subject_split = range(math.ceil(self.subjects * 0.8), self.subjects) # Use 20% of the set for testing
        else:
            subject_split = range(0, self.subjects)

        self.indices = []
        for subject in subject_split:
            for trial in range(0, 4):
                for pose in range(0, 4):
                    self.indices.append([subject + 1, trial + 1, pose + 1])


    def makeGrid(self, params):
        gridLen = self.gridSize[0] * self.gridSize[1]
        grid = np.zeros([gridLen, ], np.float32)

        indsY = np.array([i // self.gridSize[0] for i in range(gridLen)])
        indsX = np.array([i % self.gridSize[0] for i in range(gridLen)])
        condX = np.logical_and(indsX >= params[0], indsX < params[0] + params[2])
        condY = np.logical_and(indsY >= params[1], indsY < params[1] + params[3])
        cond = np.logical_and(condX, condY)

        grid[cond] = 1
        return grid

    def get_dot_index(self, subject, trial, pose, current_time):
        subject_set = self.startTime[subject]
        time_start = subject_set[trial][pose]
        # If the trial hasn't started yet, don't process it!
        if current_time < time_start:
            return -1

        # There is a new dot every 3 seconds.
        dot_index = math.floor((current_time - time_start) / 3)
        return dot_index

    def __getitem__(self, index):
        """
        Gets a batch of frames from the given video index.
        """
        subject, trial, pose = self.indices[index]
        data_points = []
        video_name = f'{subject}/{subject}_{trial}_{pose}.mp4'
        path_to_video = os.path.join(self.data_path, video_name)
        print('Loading frames for %s' % path_to_video)
        frames = extractFrames.get_frames(path_to_video)
        print('Loaded %s frames for %s' % (len(frames), path_to_video))
        num = 0
        for frame, frame_time in frames:
            num += 1
            if num % 100 == 0:
                print('Loading frame %s for video file %s' % (num, path_to_video))

            # Frame time is normally stored in MS. We need seconds.
            frame_time = frame_time / 1000
            # Get the dot index and check if the trial hasn't started or has already finished.
            dot_index = self.get_dot_index(subject - 1, trial - 1, pose - 1, frame_time)
            if dot_index < 0 or dot_index >= 35:
                continue

            topleft_dot_x_cm = self.gaze_points_x[dot_index]
            topleft_dot_y_cm = self.gaze_points_y[dot_index]

            # We need to transform the dot locations to be based from the center of the screen, which is what iTracker uses.
            dot_x_cm = topleft_dot_x_cm - self.screenCenterXCM
            dot_y_cm = -topleft_dot_y_cm

            gaze = np.array([dot_x_cm, dot_y_cm], np.float32)

            features = recognize_face.insight_extract(frame)
            faces = features[1]
            # Skip the face if it wasn't detected!
            if len(faces) == 0:
                continue

            eyes, faceGrid = features[2][0]
            # No eyes detected? Skip!
            # TODO In the future, maybe implement blink detection? The paper mentioned it, so it may be important.
            if len(eyes) < 2:
                continue

            imFace = extractFrames.crop_to_bounds(frame, faces[0])
            imEyeR = extractFrames.crop_to_bounds(frame, eyes[0])
            imEyeL = extractFrames.crop_to_bounds(frame, eyes[1])

            imFace = Image.fromarray(cv2.cvtColor(imFace, cv2.COLOR_BGR2RGB))
            imEyeR = Image.fromarray(cv2.cvtColor(imEyeR, cv2.COLOR_BGR2RGB))
            imEyeL = Image.fromarray(cv2.cvtColor(imEyeL, cv2.COLOR_BGR2RGB))

            imFace = self.transformFace(imFace)
            imEyeR = self.transformEyeR(imEyeR)
            imEyeL = self.transformEyeL(imEyeL)

            faceGrid = self.makeGrid(faceGrid)

            # to tensor
            row = torch.LongTensor([int(index)])
            faceGrid = torch.FloatTensor(faceGrid)
            data_points.append([row, imFace, imEyeL, imEyeR, faceGrid, gaze, dot_index, frame_time])

        print('Loaded %s data frames from %s %s %s' % (len(data_points), subject, trial, pose))
        return data_points


    def __len__(self):
        return len(self.indices)

class ITrackerData(data.Dataset):
    def __init__(self, dataPath, split = 'train', imSize=(224,224), gridSize=(25, 25)):

        self.dataPath = dataPath
        self.imSize = imSize
        self.gridSize = gridSize

        print('Loading iTracker dataset...')
        metaFile = os.path.join(dataPath, 'metadata.mat')
        #metaFile = 'metadata.mat'
        if metaFile is None or not os.path.isfile(metaFile):
            raise RuntimeError('There is no such file %s! Provide a valid dataset path.' % metaFile)
        self.metadata = loadMetadata(metaFile)
        if self.metadata is None:
            raise RuntimeError('Could not read metadata file %s! Provide a valid dataset path.' % metaFile)

        self.faceMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_face_224.mat'))['image_mean']
        self.eyeLeftMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_left_224.mat'))['image_mean']
        self.eyeRightMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_right_224.mat'))['image_mean']
        
        self.transformFace = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.faceMean),
        ])
        self.transformEyeL = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.eyeLeftMean),
        ])
        self.transformEyeR = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.eyeRightMean),
        ])


        if split == 'test':
            mask = self.metadata['labelTest']
        elif split == 'val':
            mask = self.metadata['labelVal']
        else:
            mask = self.metadata['labelTrain']

        self.indices = np.argwhere(mask)[:,0]
        print('Loaded iTracker dataset split "%s" with %d records...' % (split, len(self.indices)))

    def loadImage(self, path):
        try:
            im = Image.open(path).convert('RGB')
        except OSError:
            raise RuntimeError('Could not read images: ' + path)
            #im = Image.new("RGB", self.imSize, "white")

        return im


    def makeGrid(self, params):
        gridLen = self.gridSize[0] * self.gridSize[1]
        grid = np.zeros([gridLen,], np.float32)
        
        indsY = np.array([i // self.gridSize[0] for i in range(gridLen)])
        indsX = np.array([i % self.gridSize[0] for i in range(gridLen)])
        condX = np.logical_and(indsX >= params[0], indsX < params[0] + params[2]) 
        condY = np.logical_and(indsY >= params[1], indsY < params[1] + params[3]) 
        cond = np.logical_and(condX, condY)

        grid[cond] = 1
        return grid

    def __getitem__(self, index):
        index = self.indices[index]

        imFacePath = os.path.join(self.dataPath, '%05d/appleFace/%05d.jpg' % (self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]))
        imEyeLPath = os.path.join(self.dataPath, '%05d/appleLeftEye/%05d.jpg' % (self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]))
        imEyeRPath = os.path.join(self.dataPath, '%05d/appleRightEye/%05d.jpg' % (self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]))

        imFace = self.loadImage(imFacePath)
        imEyeL = self.loadImage(imEyeLPath)
        imEyeR = self.loadImage(imEyeRPath)

        imFace = self.transformFace(imFace)
        imEyeL = self.transformEyeL(imEyeL)
        imEyeR = self.transformEyeR(imEyeR)

        gaze = np.array([self.metadata['labelDotXCam'][index], self.metadata['labelDotYCam'][index]], np.float32)

        faceGrid = self.makeGrid(self.metadata['labelFaceGrid'][index,:])

        # to tensor
        row = torch.LongTensor([int(index)])
        faceGrid = torch.FloatTensor(faceGrid)
        gaze = torch.FloatTensor(gaze)

        return row, imFace, imEyeL, imEyeR, faceGrid, gaze
    
        
    def __len__(self):
        return len(self.indices)


class ITrackerImageData(data.Dataset):
    def __init__(self, image, imSize=(224, 224), gridSize=(25, 25)):
        self.imSize = imSize
        self.gridSize = gridSize

        # metaFile = 'metadata.mat'
        self.faceMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_face_224.mat'))['image_mean']
        self.eyeLeftMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_left_224.mat'))['image_mean']
        self.eyeRightMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_right_224.mat'))['image_mean']

        self.transformFace = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.faceMean),
        ])
        self.transformEyeL = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.eyeLeftMean),
        ])
        self.transformEyeR = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.eyeRightMean),
        ])

        self.indices = [image]
        print('Loaded iTracker dataset split with %d records...' % (len(self.indices)))

    def loadImage(self, path):
        try:
            im = Image.open(path).convert('RGB')
        except OSError:
            raise RuntimeError('Could not read images: ' + path)
            # im = Image.new("RGB", self.imSize, "white")

        return im

    def makeGrid(self, params):
        gridLen = self.gridSize[0] * self.gridSize[1]
        grid = np.zeros([gridLen, ], np.float32)

        indsY = np.array([i // self.gridSize[0] for i in range(gridLen)])
        indsX = np.array([i % self.gridSize[0] for i in range(gridLen)])
        condX = np.logical_and(indsX >= params[0], indsX < params[0] + params[2])
        condY = np.logical_and(indsY >= params[1], indsY < params[1] + params[3])
        cond = np.logical_and(condX, condY)

        grid[cond] = 1
        return grid

    def __getitem__(self, index):
        image = self.indices[index]
        features = extractFrames.extract_image_features(image)

        faces = features[1]
        eyes, faceGrid = features[2][0]

        imFace = extractFrames.crop_to_bounds(image, faces[0])
        imEyeL = extractFrames.crop_to_bounds(image, eyes[0])
        imEyeR = extractFrames.crop_to_bounds(image, eyes[1])

        imFace = Image.fromarray(cv2.cvtColor(imFace, cv2.COLOR_BGR2RGB))
        imEyeL = Image.fromarray(cv2.cvtColor(imEyeL, cv2.COLOR_BGR2RGB))
        imEyeR = Image.fromarray(cv2.cvtColor(imEyeR, cv2.COLOR_BGR2RGB))

        imFace = self.transformFace(imFace)
        imEyeL = self.transformEyeL(imEyeL)
        imEyeR = self.transformEyeR(imEyeR)

        faceGrid = self.makeGrid(faceGrid)

        # to tensor
        row = torch.LongTensor([int(index)])
        faceGrid = torch.FloatTensor(faceGrid)

        return row, imFace, imEyeL, imEyeR, faceGrid

    def __len__(self):
        return len(self.indices)
