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
ap.add_argument("-r", "--recognition-model", required=True, help="path to facial recognition model")
ap.add_argument("-i", "--images", required=True, help="path to images")
args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
recognizer = dlib.face_recognition_model_v1(args["recognition_model"])
aligner = FaceAligner(predictor, desiredFaceWidth=256)

imagePaths = sorted(list(list_images(args["images"])))
index = 0

for imagePath in imagePaths:
    
    image = cv2.imread(imagePath) 
    image = resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    # TODO: Accomodate multiple faces. Debug if Dwayne isn't the detected face.

    if len(rects) is 0:
        print("[INFO] No faces found in image {}! Skipping...".format(imagePath))
        index -= 1
        continue
    else:
        rois = []
        index += 1

        for rect in rects:
            shape = predictor(gray, rect)

            face_descriptor = recognizer.compute_face_descriptor(image, shape)
            print("[INFO]Face descriptor for image{}: {}".format(imagePath, face_descriptor))
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

            roi = image[y:y + h, x:x + w].copy()
            roiAligned = aligner.align(image, gray, rect)
            roiAligned = resize(roiAligned, width=256)
            rois.append(roiAligned)

        cv2.imwrite("images/dwayne/{}.jpg".format(str(index).zfill(8)), rois[0])
        print("[INFO]Normalized image {}...".format(imagePath))
