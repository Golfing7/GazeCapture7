import math, shutil, os, time, argparse

import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from main import AverageMeter

from ITrackerData import *
from ITrackerModel import ITrackerModel


'''
Make predictions on provided images.
'''

parser = argparse.ArgumentParser(description='iTracker-pytorch-predictor.')
parser.add_argument('--pytorch_model', help="Path to pytorch model")
parser.add_argument('--f', default=None, required=False)
args = parser.parse_args()

pytorch_model = None

def predict(image):
    global pytorch_model
    if pytorch_model is None:
        pytorch_model = load_model()

    data_set = ITrackerImageData(image)

    row, imFace, imEyeL, imEyeR, faceGrid = data_set[0]

    imFace = torch.autograd.Variable(imFace.unsqueeze(dim=0), requires_grad=False)
    imEyeL = torch.autograd.Variable(imEyeL.unsqueeze(dim=0), requires_grad=False)
    imEyeR = torch.autograd.Variable(imEyeR.unsqueeze(dim=0), requires_grad=False)
    faceGrid = torch.autograd.Variable(faceGrid.unsqueeze(dim=0), requires_grad=False)

    with torch.no_grad():
        prediction = pytorch_model(imFace, imEyeL, imEyeR, faceGrid)

    print(prediction)



def run_tabletgaze_generalization():
    global pytorch_model
    if pytorch_model is None:
        pytorch_model = load_model()

    data_set = TabletGazePostprocessData(dataPath="data/tablet/")
    test_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True)

    losses = AverageMeter()
    lossesLin = AverageMeter()

    criterion = nn.MSELoss().cuda() # Mean squared error loss function
    for i, (frame_group) in enumerate(test_loader):
        for frame_data in frame_group:
            row, imFace, imEyeL, imEyeR, faceGrid, gaze, index, frame_time = frame_data

            expected_x = gaze[0][0]
            expected_y = gaze[0][1]
            imFace = torch.autograd.Variable(imFace, requires_grad=False)
            imEyeL = torch.autograd.Variable(imEyeL, requires_grad=False)
            imEyeR = torch.autograd.Variable(imEyeR, requires_grad=False)
            faceGrid = torch.autograd.Variable(faceGrid, requires_grad=False)
            gaze = torch.autograd.Variable(gaze, requires_grad=False)

            with torch.no_grad():
                output = pytorch_model(imFace, imEyeL, imEyeR, faceGrid)

            loss = criterion(output, gaze)

            lossLin = output - gaze
            lossLin = torch.mul(lossLin, lossLin)
            lossLin = torch.sum(lossLin, 1)
            lossLin = torch.mean(torch.sqrt(lossLin))

            losses.update(loss.data.item(), imFace.size(0))
            lossesLin.update(lossLin.item(), imFace.size(0))

            print('Epoch (val): [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Expected {expected_x:.4f}[{output_x}] ({expected_y}[{output_y}])\t'
                  'Frame Data {gaze_index:.4f} {frame_time:.4f}\t'
                  'Error L2 {lossLin.val:.4f} ({lossLin.avg:.4f})\t'.format(
                1, i, len(test_loader),
                gaze_index=index[0], frame_time=frame_time[0],
                expected_x=expected_x, expected_y=expected_y, output_x=output[0][1], output_y=output[0][1], loss=losses, lossLin=lossesLin))


def load_model(filename="checkpoint.pth.tar"):
    # if args.pytorch_model is None:
    #     raise RuntimeError("Data path not set!")
    #
    # if not os.path.isfile(args.pytorch_model):
    #     raise RuntimeError("PyTorch model at %s does not exist!" % args.pytorch_model)

    saved = torch.load(filename, map_location=torch.device('cpu'))

    model = ITrackerModel()
    model = torch.nn.DataParallel(model)
    # model.cuda()

    state = saved['state_dict']
    print(saved['best_prec1'])
    print(saved['epoch'])
    try:
        model.module.load_state_dict(state)
    except:
        model.load_state_dict(state)
    return model

if __name__ == '__main__':
    run_tabletgaze_generalization()