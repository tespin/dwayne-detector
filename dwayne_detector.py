from collections import OrderedDict
import numpy as np
#from process import resize
#from process import shape_to_numpy
from process import rect_to_bounding
from process import bounding_to_rect
import dlib
import cv2

FACIAL_LANDMARK_INDEXES = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])

detector = dlib.get_frontal_face_detector()
recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

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

def shapes_to_numpy(shapes, dtypes="int"):
    coords = np.zeros((68, 2), dtype=dtype)

    for shape in shapes:
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return [coords]

def rect_to_tuple(rect):
    return rect.top(), rect.right(), rect.bottom(), rect.left()

def raw_locations(input, upsample=1):
    return detector(input, upsample)

def face_locations(input, upsample=1):
    #input = resize(input, width=400)

    #for location in raw_locations(input, upsample):
        #return [rect_to_bounding(location)]

    return [rect_to_bounding(location) for location in raw_locations(input, upsample)]

def raw_landmarks(input, locations = None):
    if locations is None:
        locations = raw_locations(input)
    else:
        locations = [bounding_to_rect(location) for location in locations]


    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    return [predictor(input, location) for location in locations]

def face_landmarks(input, locations=None):
    landmarks = raw_landmarks(input, locations)
    landmark_tuples = [[(part.x, part.y) for part in landmark.parts()] for landmark in landmarks]

    return landmark_tuples

def face_encodings(input, locations=None, num_jitters=1):
    landmarks = raw_landmarks(input, locations)

    return [np.array(recognizer.compute_face_descriptor(input, landmark, num_jitters)) for landmark in landmarks]

def face_distance(known_encodings, unknown_encodings):
    if len(known_encodings) == 0:
        return np.empty((0))

#    return [np.linalg.norm(known_encodings - unknown_encoding) for unknown_encoding in unknown_encodings]
#    return [np.linalg.norm(known_encodings - unknown_encodings for (known_encodings, unknown_encodings) in zip(known_encodings, unknown_encodings))]
    return np.linalg.norm(known_encodings - unknown_encodings)

def compare(known_encoding, unknown_encodings, tolerance=0.6):
    #distances = face_distance(known_encoding, unknown_encodings)

    #return [distance <= tolerance for distance in distances]
    return [face_distance(known_encoding, unknown_encodings) <= tolerance]
