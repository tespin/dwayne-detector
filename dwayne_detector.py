import numpy as np
from process import resize
from process import shape_to_numpy
from process import rect_to_bounding
import dlib
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def rect_to_tuple(rect):
    return rect.top(), rect.right(), rect.bottom(), rect.left()

def raw_locations(input, upsample=1):
    return detector(input, upsample)

def face_locations(input, upsample=1):
    #input = resize(input, width=400)

    for location in raw_locations(input, upsample):
        return [rect_to_bounding(location)]


