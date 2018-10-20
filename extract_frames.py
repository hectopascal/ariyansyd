#!/usr/bin/env python3

import sys
import cv2
import os
import os.path
import argparse
import logging

parser = argparse.ArgumentParser(description='Preprocess input file(s)')
parser.add_argument('source', type=str, help='source file')
parser.add_argument('destination', type=str, help='destination folder')
parser.add_argument('log_level', type=str, help='logging level')
args = parser.parse_args()

log = logging.getLogger(__name__)

def preprocess_frames(source_path):
    # do blur detection?
    pass

def extract_frames(video_file, destination):
    """ Extract frames from a video file
    
    Extracts frames from a video file using opencv
    
    Arguments:
        video_file {[type]} -- [description]
    """
    video_capture = cv2.VideoCapture(video_file)

    if not video_capture.isOpened():
        log.error("Could not open file:", file_path)
        return False

    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = video_capture.get(cv2.CAP_PROP_FPS)

    print('Width: {}')
    for fr in range(frame_count):
        ok, frame = video_capture.read()
        out = os.path.join(destination, "{}_{:08d}.jpg".format(video_file, fr))  # JPEG??
        print(out)
        cv2.imwrite(out, frame)


if __name__ == '__main__':
    
    if not os.path.exists(args.source):
        log.error("Source path does not exist")
        sys.exit()    

    if os.path.exists(args.destination):
        if len(os.listdir(args.destination)):
            log.error("Destination path is not empty")
            sys.exit()
    else:
        try:
            os.mkdir(args.destination)
        except OSError as err:
            log.error("Could not create destination directory {}".format(err))
            sys.exit()

    extract_frames(args.source, args.destination)

