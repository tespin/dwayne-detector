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

images = []

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
recognizer = dlib.face_recognition_model_v1(args["recognition_model"])

imDwaynePath = os.path.basename(args["dwayne"])
imUnknownPath = os.path.basename(args["unknown"])

#imDwayne = cv2.imread(args["dwayne"])
#imUnknown = cv2.imread(args["unknown"])
#images.extend([imDwayne, imUnknown])

imDwayne = cv2.imread(imDwaynePath)
imUnknown = cv2.imread(imUnknownPath)

images = {imDwaynePath: imDwayne, imUnknownPath: imUnknown}

encodings = []

for path, image in images.items():
    image = resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    if len(rects) is 0:
        print("[INFO] No faces found in image {}! Skipping...".format(path))
    else:
        rois = []

        for rect in rects:
            shape = predictor(gray, rect)

            face_descriptor = recognizer.compute_face_descriptor(image, shape)
            print("[INFO] Face descriptor for image {}.".format(path))



#for index, image in enumerate(images):
#    image = resize(image, width=500)
#    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#    
#    rects = detector(gray, 1)
#
#    if len(rects) is 0:
#        print("[INFO] No faces found in image {}! Skipping...".format(index))
#        continue
#    else:
#        rois = []
#
#        for rect in rects:
#            shape = predictor(gray, rect)
#
#            face_descriptor = recognizer.compute_face_descriptor(image, shape)
#            print("[INFO]Face descriptor for image{}
#
# take two images as input
# detect face
# align face
# compute face descriptors
# calculate Euclidean distance between two input faces (dwayne descriptor and other)
