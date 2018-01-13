from videostream import VideoStream
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
import pickle
import os
import time

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

# if no encoding is supplied, compute one with given path "encoding-input"
if args["load_encoding"] < 0:
    print("[INFO] Loading image...")
    input = cv2.imread(args["encoding_input"])
    input = resize(input, width=500)

    bounds = detector(input, 1)

    print("[INFO] Computing face descriptor...")
    for bound in bounds:
        shape = predictor(input, bound)
        baseline = np.array(recognizer.compute_face_descriptor(input, shape, num_jitters=1))
elif args["load_encoding"] is 1:
    print("[INFO] Loading encoding...")
    input = open("baseline.dat", "rb")
    baseline = pickle.load(input)
    print("[INFO] Encoding loaded!")

# if baseline encoding should be saved
if args["save_encoding"] > 0:
    output = open("baseline.dat", "wb")
    print("[INFO] Dumping contents into baseline.dat.")
    pickle.dump (baseline, output)
    print("[INFO] Contents written!")
    output.close()

print("[INFO] Starting stream...")
stream = VideoStream().start()
print("[INF0] Warming up...")
time.sleep(2.0)

encodings = []

while True:
    frame = stream.read()
    frame = resize(frame, width=400)

    bounds = detector(frame, 0)

    for bound in bounds:
        shape = predictor(frame, bound)
        #encoding = [np.array(recognizer.compute_face_descriptor(frame, shape, num_jitters=1))]
        encodings = [np.array(recognizer.compute_face_descriptor(frame, shape, num_jitters=1))]

        shape = shape_to_numpy(shape)

        (x, y, w, h) = rect_to_bounding(bound)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

        for x, y in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    names = []
    for encoding in encodings:
        if np.linalg.norm(baseline - encoding) < 0.6:
            name = "Dwayne"
        else:
            name = "Unknown"

        names.append(name)

    for bound, name in zip(bounds, names):
        (x, y, w, h) = rect_to_bounding(bound)
        cv2.putText(frame, name, (x + 5, y - 5), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    #print("Number of encodings: {}".format(len(encodings)))
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord ("q"):
        break

cv2.destroyAllWindows()
stream.stop()
