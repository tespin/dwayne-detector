import argparse
import numpy as np
from process import resize
from process import shape_to_numpy
from process import rect_to_bounding
import dlib
import cv2

# TODO: poll video stream
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dwayne", required=True, help="path to dwayne")
ap.add_argument("-u", "--unknown", required=True, help="path to unknown person")
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-r", "--recognition-model", required=True, help="path to facial recognition model")
args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
recognizer = dlib.face_recognition_model_v1(args["recognition_model"])



# take two images as input
# detect face
# align face
# compute face descriptors
# calculate Euclidean distance between two input faces (dwayne descriptor and other)
