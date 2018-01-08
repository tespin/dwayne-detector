import argparse
import math
import numpy as np
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
imDwayne = cv2.imread(args["dwayne"])

# define other image with another detectable face
imUnknown = cv2.imread(args["unknown"])

images.extend([args["dwayne"], args["unknown"]])

encodings = []

for path in images:
    image = cv2.imread(path)
    image = resize(image, width=500)
    cv2.imshow("Input {}".format(path), image)

    rects = detector(image, 1)
    numFaces = len(rects)
    print("Number of detected faces in image {}: {}.".format(path, numFaces))

    for rect in rects:
        shape = predictor(image, rect)
        descriptor = recognizer.compute_face_descriptor(image, shape)
        #print("Descriptor for image {}: {}".format(path, descriptor))
        encodings.extend(descriptor)

print(np.linalg.norm(encodings[0] - encodings[1]))
cv2.waitKey(0)
