# import libraries 
import cv2
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


# initialize the working directory
input_dir = 'input'
video_dir = 'video'
output_dir = 'frame'


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
    help="input video file name")
ap.add_argument("-p", "--preprocess", type=str, default="thresh",
    help="type of preprocessing to be done")
ap.add_argument("-w", "--width", type=int, default=320,
    help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320,
    help="resized image height (should be multiple of 32)")
args = vars(ap.parse_args())


in_path = os.path.join(input_dir,video_dir)
out_path =  os.path.join(input_dir, output_dir)


file = args["video"]
file_path = os.path.join(in_path, file)
print(file_path)


vidcapture = cv2.VideoCapture(file_path)
success,image = vidcapture.read()
count = 0
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
while success:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if args["preprocess"] == "thresh":
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    elif args["preprocess"] == "blur":
        gray = cv2.medianBlur(gray, 3)

    cv2.imwrite(output_dir+"\\frame%d.jpg" % count, gray)
    success,image = vidcapture.read()
    count += 1
print('video to frame converted successfully for file: ', file)

