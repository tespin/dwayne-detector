import argparse
import dwayne_detector
from process import resize
from process import shapes_to_numpy
from process import shape_to_numpy
import dlib
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dwayne", required=True, help="path to dwayne")
ap.add_argument("-u", "--unknown", required=True, help="path to unknown")
args = vars(ap.parse_args())

dwayne = cv2.imread(args["dwayne"])
dwayne = resize(dwayne, width=800)

unknown = cv2.imread(args["unknown"])
unknown = resize(unknown, width=800)

dwayne_encoding = dwayne_detector.face_encodings(dwayne)[0]
unknown_encoding = dwayne_detector.face_encodings(unknown)[0]
#print(dwayne_encoding)

known_encodings = [
    dwayne_encoding
]

results = dwayne_detector.compare(known_encodings, unknown_encoding)

print("Is it Dwayne? {}".format(results[0]))

d_locations = dwayne_detector.face_locations(dwayne)
d_encodings = dwayne_detector.face_encodings(dwayne, d_locations)
d_landmarks = dwayne_detector.face_landmarks(dwayne)
#d_landmarks = dwayne_detector.face_landmarks(dwayne, d_locations)
d_shapes = shapes_to_numpy(dwayne_detector.raw_landmarks(dwayne, d_locations))

for (x, y, w, h) in d_locations:
#    print(x, y, w, h)
    cv2.rectangle(dwayne, (x, y), (x + w, y + h), (0, 255, 0), 1)

for d_landmark in d_landmarks:
    for coords in d_landmark:
        (x, y) = (coords[0], coords[1])
        cv2.circle(dwayne, (x, y), 1, (0, 0, 255), -1)

cv2.imshow("Dwayne", dwayne)
cv2.waitKey(0)
#capture = cv2.VideoCapture(0)


