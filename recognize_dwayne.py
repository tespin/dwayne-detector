import argparse
import math
import numpy as np
import scipy.misc
from process import resize
from process import shape_to_numpy
from process import rect_to_bounding
from face_aligner import FaceAligner
import dlib
import cv2
import os

# TODO: poll video stream
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dwayne", required=True, help="path to dwayne")
ap.add_argument("-u", "--unknown", required=True, help="path to unknown person")
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-r", "--recognition-model", required=True, help="path to facial recognition model")
args = vars(ap.parse_args())

images = []

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
recognizer = dlib.face_recognition_model_v1(args["recognition_model"])
aligner = FaceAligner(predictor, desiredFaceWidth=256)

# define dwayne's image
#imDwayne = cv2.imread(args["dwayne"])
#imDwayne = scipy.misc.imread(args["dwayne"], mode='RGB')
imDwayne = cv2.imread(args["dwayne"])

dwayneDetected = detector(imDwayne, 1)

for rect in dwayneDetected:
    dwayneShape = predictor(imDwayne, rect)

    dwayneEncoding = np.array(recognizer.compute_face_descriptor(imDwayne, dwayneShape, num_jitters=1))  
#    print("Dwayne encoding: {}".format(dwayneEncoding))

# define other image with another detectable face
#imUnknown = cv2.imread(args["unknown"])
#imUnknown = scipy.misc.imread(args["unknown"], mode='RGB')
#imUnknown = scipy.misc.imread(args["unknown"], mode='RGB')
imUnknown = cv2.imread(args["unknown"])

unknownDetected = detector(imUnknown, 1)

for rect in unknownDetected:
    unknownShape = predictor(imUnknown, rect)

    unknownEncoding = np.array(recognizer.compute_face_descriptor(imUnknown, unknownShape, num_jitters=1))
#    print("Unknown encoding: {}".format(unknownEncoding))

distance = np.linalg.norm(dwayneEncoding - unknownEncoding)

print("Distance: {}".format(distance))

if distance < 0.6:
    print("It's a picture of Dwayne The Rock Johnson!")
else:
    print("It's not a picture of The Rock!")

