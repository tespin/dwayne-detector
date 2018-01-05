import numpy as np
import argparse
from process import resize
from process import shape_to_numpy
from process import rect_to_bounding
from process import list_images
from face_aligner import FaceAligner
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-i", "--images", required=True, help="path to images")
args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
aligner = FaceAligner(predictor, desiredFaceWidth=256)

imagePaths = sorted(list(list_images(args["images"])))

for (index, imagePath) in enumerate(imagePaths):
    
    image = cv2.imread(imagePath)
    image = resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)
