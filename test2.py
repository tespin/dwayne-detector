import argparse
import dwayne_detector
from process import resize
from process import shapes_to_numpy
from process import shape_to_numpy
import dlib
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dwayne", required=True, help="path to dwayne")
args = vars(ap.parse_args())

dwayne = cv2.imread(args["dwayne"])
dwayne = resize(dwayne, width=400)

d_locations = dwayne_detector.face_locations(dwayne)
d_encodings = dwayne_detector.face_encodings(dwayne, d_locations)
d_shapes = shapes_to_numpy(dwayne_detector.raw_landmarks(dwayne, d_locations))
d_landmarks = dwayne_detector.raw_landmarks(dwayne, d_locations)
d_landmark_locations = [[(p.x, p.y) for p in d_landmark.parts()] for d_landmark in d_landmarks]

for (x, y, w, h) in d_locations:
    cv2.rectangle(dwayne, (x, y), (x + w, y + h), (0, 255, 0), 1)

cv2.imshow("Dwayne", dwayne)
cv2.waitKey(0)
#capture = cv2.VideoCapture(0)


