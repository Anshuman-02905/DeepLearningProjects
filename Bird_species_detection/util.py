import numpy as np
import cv2

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)

    resized=np.array([resized])
    return resized

def convert_image(uploaded_file):
    print('hello')
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        return opencv_image
    return None


#resized = cv2.resize(image, dim, interpolation=inter)

#return resized
#img = np.array([image_resize(img, height=224)])
#image_resize(img, height=224)