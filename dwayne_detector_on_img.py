import argparse
import dwayne_detector
import dlib
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dwayne", required=True, help="path to Dwayne")
ap.add_argument("-u", "--unknown", required=True, help="path to img of unknown person")
args = vars(ap.parse_args())

dwayne = cv2.imread(args["dwayne"])
dwayne = dwayne_detector.resize(dwayne, width=800)

unknown = cv2.imread(args["unknown"])
unknown = dwayne_detector.resize(unknown, width=800)
