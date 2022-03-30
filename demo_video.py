import sys
import time
# import numpy as np
# from PIL import Image, ImageDraw
from numpy import False_
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import torch
import argparse

"""hyper parameters"""
use_cuda = True

def detect_video(cfgfile, weightfile,vidfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    #begin video capture
    cap = cv2.VideoCapture(vidfile)
    
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
    else: 
        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            # if ret == True:
            #     # Display the resulting frame
            #     cv2.imshow('Frame',frame)
            # else:
            #     break
    #When everything done, release the video capture object
    cap.release()
    
    print("Starting the YOLO loop...")

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    elif num_classes == 2 :
        namesfile = 'data/custom.names'
    class_names = load_class_names(namesfile)

    while True:
        ret, img = cap.read()
        sized = cv2.resize(img, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
        finish = time.time()
        print('Predicted in %f seconds.' % (finish - start))

        result_img = plot_boxes_cv2(img, boxes[0], savename=None, class_names=class_names)

        cv2.imshow('Yolo demo', result_img)
        cv2.waitKey(1)

    cap.release()

def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default='./cfg/yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='./checkpoints/Yolov4_epoch1.pth',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-imgfile', type=str,
                        help='path of your image file.', dest='imgfile')
    parser.add_argument('-vidfile', type=str,
                        help='path of your video file.', dest='vidfile')
    parser.add_argument('-torch', type=bool, default=False,
                        help='use torch weights')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    print(args)
    if args.vidfile:
        detect_video(args.cfgfile, args.weightfile, args.vidfile)
    
