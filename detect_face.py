import numpy as np
import argparse
from process import resize
from process import shape_to_numpy
from process import rect_to_bounding
from process import list_images
from face_aligner import FaceAligner
import dlib
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
#ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-i", "--images", required=True, help="path to images")
args = vars(ap.parse_args())

# initialize dlib's face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
aligner = FaceAligner(predictor, desiredFaceWidth=256)

imagePaths = sorted(list(list_images(args["images"])))

for (index, imagePath) in enumerate(imagePaths):
    # load input image, resize, and convert to grayscale
    image = cv2.imread(imagePath)
    image = resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = shape_to_numpy(shape)

        (x, y, w, h) = rect_to_bounding(rect)
        roi = image[y:y + h, x:x + w].copy()
        roiResized = resize(roi, width=256)
        roiAligned = aligner.align(image, gray, rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

        cv2.imwrite("images/dwayne/00000{}.jpg".format(index), roiAligned)

print(type(rects))
cv2.imshow("Output", image)
cv2.imshow("ROI", roi)
cv2.imshow("Aligned", roiAligned)
cv2.waitKey(0)
