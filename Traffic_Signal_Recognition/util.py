import json
from tensorflow import keras
import cv2
import numpy as np


def convert_image(uploaded_file):
    print('hello')
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        return opencv_image
    return None

def get_labels(lable_file):
    dct = dict()
    with open(lable_file) as json_file:
        dct = json.load(json_file)
    return dct

def get_model(model_file):
    model=keras.models.load_model(model_file)
    return model

def model_predict(model,img):
    img = cv2.resize(img, (30, 30), interpolation=cv2.INTER_AREA)
    img = np.array([img])
    prediction=str(np.argmax(model.predict(img)))
    return prediction
