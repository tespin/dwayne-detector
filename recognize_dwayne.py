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

images.extend([imDwayne, imUnknown])

encodings = []

# process each face
for index, image in enumerate(images):
    image = resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    if len(rects) is 0:
        print("No faces found in this image! Please supply an image with a detected face. Exiting...")
        exit()

    else:
        rois = []

        for rect in rects:
            shape = predictor(gray, rect)
            face_descriptor = recognizer.compute_face_descriptor(image, shape)
            print("Face Descriptor for Image {}: {}".format(index, face_descriptor))
            encodings.extend(face_descriptor)
            shape = shape_to_numpy(shape)

#    for encoding in encodings:
#        print("Encodings for image {}: {}".format(index, encoding))

#vector = encodings[0] - encodings[1]
#print("Distance: {}".format(distance))
#print(sqrt(np.dot(encodings[0], encodings[1])))
#print("Dot product: {}".format(np.dot(vector, vector)))
#print("Length: {}".format(sqrt(np.dot(vector, vector))))

distance = np.linalg.norm(encodings[0] - encodings[1])
print("Distance: {}".format(distance))

# compute face_descriptor for dwayne
# compute face_descriptor for other input
# subtrct dlib vectors to find euclidean distance
# if < 0.6, it is dwayne!
