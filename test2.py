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

locations = dwayne_detector.raw_locations(dwayne)
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#print("d_locations: {}".format(d_locations))
for d_location in d_locations:
    print("d_location: {}".format(d_location))
    print(type(d_location))
#print(type(d_locations))
print("-------")

for location in locations:
    print("location: {}".format(location))
    print(type(location))
    shape = predictor(dwayne, location)
    shape = shape_to_numpy(shape)

    for (x, y) in shape:
        cv2.circle(dwayne, (x, y), 1, (0, 0, 255), -1)

#print(type(d_landmark_locations))
#print(type(dwayne_detector.raw_landmarks(dwayne, d_locations)))

for (x, y, w, h) in d_locations:
    cv2.rectangle(dwayne, (x, y), (x + w, y + h), (0, 255, 0), 1)



#for shape in d_shapes:
#    for coord in shape:
#        (x, y) = (coord[0], coord[1])
#        cv2.circle(dwayne, (x, y), 1, (0, 0, 255), -1)

#for d_landmark_location in d_landmark_locations:
#    print(d_landmark_location)
#    for coords in d_landmark_location:
#        print(type(coords))
#       print(coords)
#        (x, y) = (coords[0], coords[1])
#        cv2.circle(dwayne, (x, y), 1, (0, 0, 255), -1)

cv2.imshow("Dwayne", dwayne)
cv2.waitKey(0)
#capture = cv2.VideoCapture(0)


