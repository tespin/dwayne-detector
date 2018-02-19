import argparse
import dwayne_detector
import dlib
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dwayne", required=True, help="path to Dwayne")
ap.add_argument("-u", "--unknown", required=True, help="path to img of unknown person")
args = vars(ap.parse_args())

dwayne = cv2.imread(args["dwayne"])
dwayne = dwayne_detector.resize(dwayne, width=600)

unknown = cv2.imread(args["unknown"])
unknown = dwayne_detector.resize(unknown, width=600)

dwayne_encoding = dwayne_detector.face_encodings(dwayne)[0]

unknown_locations = dwayne_detector.face_locations(unknown)
unknown_encodings = dwayne_detector.face_encodings(unknown)

label_height = 60

for (x, y, w, h), unknown_encoding in zip(unknown_locations, unknown_encodings):
    result = dwayne_detector.compare(dwayne_encoding, unknown_encoding)
    print(result)

    name = "Not \nDwayne"

    if True in result:
        name = "Dwayne"
        label_height = 30

    cv2.rectangle(unknown, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.rectangle(unknown, (x, y + h), (x + w, y + h + label_height), (0, 255, 0), cv2.FILLED)
    #cv2.putText(unknown, name, (x + 6, y + h + 25), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
    for i, line in enumerate(name.split("\n")):
        #cv2.rectangle(unknown, (x, y + h), (x + w, y + h + box_height * (i+1)), (0, 255, 0), cv2.FILLED)
        cv2.putText(unknown, line, (x + 4, y + h + 25*(i+1)), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 1)
        print(line)

cv2.imshow("Unknown", unknown)
cv2.waitKey(0)

