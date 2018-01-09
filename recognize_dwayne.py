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

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
recognizer = dlib.face_recognition_model_v1(args["recognition_model"])
aligner = FaceAligner(predictor, desiredFaceWidth=256)

paths = [args["dwayne"], args["unknown"]]
encodings = []

for path in paths:
    image = cv2.imread(path)
    image = resize(image, width=500)
#    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(image, 1)

    for rect in rects:
        shape = predictor(image, rect)
        encoding = np.array(recognizer.compute_face_descriptor(image, shape, num_jitters=1))
        #print("Encoding for face {}: {}".format(os.path.basename(path), encoding))
        encodings.append(encoding)


    print("Encodings size: {}".format(len(encodings)))

distance = np.linalg.norm(encodings[0] - encodings[1])

if distance < 0.6:
    print("It's a picture of Dwayne The Rock Johnson! Distance: {}".format(distance))
else:
    print("It's not a picture of The Rock! Distance: {}".format(distance))
