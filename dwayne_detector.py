from collections import OrderedDict
import numpy as np
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
    """
    Given a path to a directory, list all the files in the directory that end with a valid extension.

    :param basePath: string containing path to a directory with images
    :param validExtensions: tuple containing relevant file extensions
    :param contains: string used to filter files that don't have what it specified
    :return: an iterable with paths to the images
    """
    for (root, directory, filenames) in os.walk(basePath):
        for filename in filenames:
            if contains is not None and filename.find(contains) == -1:
                continue

            extension = filename[filename.rfind("."):].lower()

            if extension.endswith(validExtensions):
                imagePath = os.path.join(root, filename).replace(" ", "\\ ")
                yield imagePath

def list_images(basePath, contains=None):
    """
    Given a path to a directory, list all the image files that contain what is specified.

    :param basePath: string containing path to a directory with images
    :param contains: string used to filter files that don't have what is specified
    :return: generator with image path for each file in directory
    """
    return list_files(basePath, validExtensions=(".jpg", ".jpeg", ".png", ".bmp"), contains=contains)

def resize(image, width=None, height=None, interpolation=cv2.INTER_AREA):
    """
    Resizes an image given a width or height.

    :param image: numpy ndarray containing rgb values for an image
    :param width: int representing desired width to resize to
    :param height: int representing desired height to resize to
    :param interpolation: resampling method -- INTER_AREA has good results with image decimation
    :return: numpy ndarray representing the resized image
    """

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
    """
    Converts a dlib full_object_detection's points into a numpy ndarray. We initialize
    coords as a matrix with 68 rows and 2 columns to represent the (x, y) coordinates
    of the 68 facial landmarks.

    :param shape: dlib full_object_detection containing rects for each detected object
                  and points for its shape
    :param dtype: data type for the array
    :return coords: 68 x 2 array representing the locations of each facial landmark
    """ 

    coords = np.zeros((68, 2), dtype=dtype)

    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

def shapes_to_numpy(shapes, dtypes="int"):
    """
    Converts each full_object_detection in the 'shapes' list into a numpy ndarray. 'Coords'
    is initialized as a (68, 2) matrix to represent the (x, y) coordinates of the 68 
    facial landmarks.

    :param shapes: list of dlib full_object_detection objects corresponding to each detected face
    :param dtype: data type for the array
    :return: list of coords for each facial landmark
    """

    coords = np.zeros((68, 2), dtype=dtype)

    for shape in shapes:
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return [coords]

def rect_to_bounding(rect):
    """
    Convert dlib rectangle into a tuple of (x, y, width, height).

    :param rect: dlib rectangle representing bounding box of detected faces
    :return: tuple of values representing location and dimension of bounding box
    """
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)

def bounding_to_rect(rect):
    """
    Convert tuple into dlib rectangle.

    :param rect: tuple containing values for the bounding box of detected faces
    :return: dlib.rectangle containing values for a bounding box
    """

    return dlib.rectangle(rect[0], rect[1], rect[2], rect[3])

#def rect_to_tuple(rect):
#    return rect.top(), rect.right(), rect.bottom(), rect.left()

def raw_locations(input, upsample=1):
    """
    Returns list of dlib rect objects of each face in image.

    :param input: input image as a numpy ndarray
    :param upsample: number of times to upsample image while looking for faces    
    :return: list of dlib rect objects
    """

    return detector(input, upsample)

def face_locations(input, upsample=1):
    """
    Returns list of found faces as tuples.

    :param input: input image as a numpy ndarray
    :param upsample: number of times to umsample image while looking for faces
    :return: list of found faces as tuples in (x, y, w, h) order
    """

    #input = resize(input, width=400)

    #for location in raw_locations(input, upsample):
        #return [rect_to_bounding(location)]

    return [rect_to_bounding(location) for location in raw_locations(input, upsample)]

def raw_landmarks(input, locations = None):
    """
    Returns list of 68 landmarks per detected face.

    :param input: input image as a numpy ndarray
    :param locations: dlib rect objects representing detected faces
    :return: list of landmarks for each found face 
    """

    if locations is None:
        locations = raw_locations(input)
    else:
        locations = [bounding_to_rect(location) for location in locations]


    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    return [predictor(input, location) for location in locations]

def face_landmarks(input, locations=None):
    """
    Returns list of dictionaries containing (x, y) coords of each face feature.

    :param input: input image as a numpy ndarray
    :param locations: dlib rect objects representing detected faces
    :return landmark_tuples: list of dictionaries with coords for each facial landmark
    """

    landmarks = raw_landmarks(input, locations)
    landmark_tuples = [[(part.x, part.y) for part in landmark.parts()] for landmark in landmarks]

    return landmark_tuples

def face_encodings(input, locations=None, num_jitters=1):
    """
    Returns 128-dim face encoding for each detected face.

    :param input: input image as a numpy ndarray
    :param locations: dlib rect objects representing detected faces
    :param num_jitters: number of times to re-sample detected faces when calculating encodings
    :return: list of 128-dim face encodings for each face found in input
    """

    landmarks = raw_landmarks(input, locations)

    return [np.array(recognizer.compute_face_descriptor(input, landmark, num_jitters)) for landmark in landmarks]

def face_distance(known_encodings, unknown_encodings):
    """
    Calculates Euclidean distance between two encodings.

    :param known_encodings: list of 128-dim face encodings for Dwayne
    :param unknown_encodings: list of 128-dim face encodings for faces found in input image
    :return: numpy ndarray with distance for each face
    """

    if len(known_encodings) == 0:
        return np.empty((0))

#    return [np.linalg.norm(known_encodings - unknown_encoding) for unknown_encoding in unknown_encodings]
#    return [np.linalg.norm(known_encodings - unknown_encodings for (known_encodings, unknown_encodings) in zip(known_encodings, unknown_encodings))]
    return np.linalg.norm(known_encodings - unknown_encodings)

def compare(known_encoding, unknown_encodings, tolerance=0.6):
    """
    Compares encodings by determining if each result is above or below a certain tolerance value.

    :param known_encoding: 128-dim face encoding for Dwayne
    :param unknown_encodings: list of encodings for faces that need to be compared to
    :param tolerance: distance between faces that can be considered a match
    """

    #distances = face_distance(known_encoding, unknown_encodings)

    #return [distance <= tolerance for distance in distances]
    return [face_distance(known_encoding, unknown_encodings) <= tolerance]
