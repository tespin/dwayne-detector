import argparse
import dwayne_detector
import dlib
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dwayne", required=True, help="path to Dwayne")
args = vars(ap.parse_args())

dwayne = cv2.imread(args["unknown"])
dwayne = dwayne_detector.resize(dwayne, width=800)
