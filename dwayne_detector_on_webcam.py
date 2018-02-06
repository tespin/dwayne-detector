import argparse
import dwayne_detector
import dlib
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dwayne", required=True, help="path to Dwayne")
args = vars(ap.parse_args())

dwayne = cv2.imread(args["unknown"])
dwayne = dwayne_detector.resize(dwayne, width=800)

dwayne_encoding = dwayne_detector.face_encodings(dwayne)[0]

capture = cv2.VideoCapture(0)

while True:
    unknown = capture.read()
    unknown = dwayne_detector.resize(unknown, width=800)

    unknown_locations = dwayne_detector.face_locations(unknown)
    unknown_encodings = dwayne_detector.face_encodings(unknown)

    for (x, y, w, h), unknown_encoding in zip(unknown_locations, unknown_encodings):
        result = dwayne_detector.compare(dwayne_encoding, unknown_encoding)
        print(result)

        name = "Not Dwayne"

        if True in result:
            name = "Dwayne"

       cv2.rectangle(unknown, (x, y), (x + w, y + h), (0, 255, 0), 1)
       cv2.rectangle(unknown, (x, y + h), (x + w, y + h + 30), (0, 255, 0), cv2.Filled)       
       cv2.putText(unknown, name, (x + 6, y + h + 25), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
