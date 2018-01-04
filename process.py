from collections import OrderedDict
import numpy as np
import cv2
import sys
import os

FACIAL_LANDMARKS_INDEXES = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])

def list_files(basePath, validExtensions=(".jpg", ".jpeg", ".png", ".bmp"), contains=None):
    for (root, directory, filenames) in os.walk(basePath):
        for filename in filenames:
            if contains is not None and filename.find(contains) == -1:
                continue

            extension = filename[filename.rfind("."):].lower()

            if extension.endswith(validExtensions):
                imagePath = os.path.join(root, filename).replace(" ", "\\ ")
                yield imagePath

def list_images(basePath, contains=None):
    return list_files(basePath, validExtensions=(".jpg", ".jpeg", ".png", ".bmp"), contains=contains)

def resize(image, width=None, height=None, interpolation=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        ratio = height / float(h)
        dimensions = (int(w * ratio), height)

    else:
        ratio = width / float(w)
        dimensions = (width, int(h * ratio))

    resized = cv2.resize(image, dimensions, interpolation=interpolation)

    return resized

def shape_to_numpy(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)

    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

def rect_to_bounding(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)
