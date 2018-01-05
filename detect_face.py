import numpy as np
import argparse
from process import resize
from process import shape_to_numpy
from process import rect_to_bounding
import dlib
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

image = cv2.imread(args["image"])

image = resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 1)

for rect in rects:
    shape = predictor(gray, rect)
    shape = shape_to_numpy(shape)

    (x, y, w, h) = rect_to_bounding(rect)

    if y < 0:
        y = 0
    elif y > y + h:
        y = y + h
 
    if x < 0:
        x = 0
    elif x > x + w:
        x = x + w   

    roi = image[y:y + h, x:x + w]
    roiResized = resize(roi, width=256)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

cv2.imshow("ROI", roiResized)
cv2.imshow("Output", image)
cv2.waitKey(0)
