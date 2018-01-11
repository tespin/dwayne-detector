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

baseline_encoding = None

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-encoding", type=int, default=-1, help="[OPTIONAL]whether encodings should be saved to disk")
ap.add_argument("-l", "--load-encoding", type=int, default=-1, help="[OPTIONAL]whether or not encodings should be loaded")
ap.add_argument("-e", "--encoding-input", type=str, help="[OPTIONAL] path to encoding")
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-r", "--recognition-model", required=True, help="path to facial recognition model")
args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
recognizer = dlib.face_recognition_model_v1(args["recognition_model"])

encodings = []

# if no encoding is supplied, compute one with given path "encoding"
if args["load_encoding"] < 0:
    input = cv2.imread(args["encoding_input"])
    input = resize(input, width=500)

    bounds = detector(input, 1)

    for bound in bounds:
        shape = predictor(input, bound)
        baseline = np.array(recognizer.compute_face_descriptor(input, shape, num_jitters=1))
        encodings.append(baseline)
elif args["load_encoding"] is 1:
    baseline = dlib.deserialize(args["encoding_input"])

    
# if baseline encoding should be saved
if args["save_encoding"] > 0:
    # encodings[0] is saved
    dlib.serialize(encodings[0], args["encoding_input"])

