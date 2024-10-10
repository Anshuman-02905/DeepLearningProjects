
import streamlit as st
from tensorflow import keras
import numpy as np
import util
import cv2




st.image('Bird_species_detection/BlueBird.jpg', channels="BGR")

st.title("Indian Bird Species Recognition")
st.header("About")
st.markdown("This bird detection deep learning app is a computer vision application that uses deep learning techniques to detect and classify birds in images . The app consists of a trained deep neural network model that can recognize different species of birds based on their appearance.")
st.header("Development Summary")
st.markdown("To develop a bird detection app, a large dataset of bird images is needed to train the neural network model. The model is typically based on a convolutional neural network (CNN) architecture, which has shown excellent performance in image recognition tasks. During training, the model learns to identify the distinguishing features of different bird species, such as their shape, size, color, and patterns. Once the model is trained, it can be integrated into the app, which typically includes a user interface that allows users to upload or capture images or videos. The app then applies the model to the input data to detect and classify birds, highlighting their locations and identifying their species.")

image_size = 224
model = keras.models.load_model('Bird_species_detection/model.h5')
st.header("Birds that are recognized in the current model")
st.markdown('Asian Green Bee-Eater , Brown-Headed Barbet , Cattle Egret , Common Kingfisher ,  Common Myna , Common Rosefinch , Common Tailorbird , Coppersmith Barbet , Forest Wagtail , Gray Wagtail , Hoopoe , House Crow , Indian Grey Hornbill , Indian Peacock , Indian Pitta , Indian Roller , Jungle Babbler, Northern Lapwing , Red-Wattled Lapwing , Ruddy Shelduck , Rufous Treepie , Sarus Crane , White Wagtail , White-Breasted Kingfisher , White-Breasted Waterhen')


st.header("NOTE")
st.markdown("The input image should be have equal resolution  that means the length and width of image should be same. If the image is not in 1:1 ratio then do the same from the following website ->https://www.img2go.com/crop-image ")

img = st.file_uploader("Choose a file")
if st.button("submit Image"):
    img=util.convert_image(img)
    st.image(img, channels="BGR")
    img=util.image_resize(img,height = 224)
    dct={0:'Asian Green Bee-Eater', 1:'Brown-Headed Barbet', 2:'Cattle Egret', 3:'Common Kingfisher', 4:'Common Myna', 5:'Common Rosefinch', 6:'Common Tailorbird', 7:'Coppersmith Barbet', 8:'Forest Wagtail', 9:'Gray Wagtail', 10:'Hoopoe', 11:'House Crow', 12:'Indian Grey Hornbill', 13:'Indian Peacock', 14:'Indian Pitta', 15:'Indian Roller', 16:'Jungle Babbler', 17:'Northern Lapwing', 18:'Red-Wattled Lapwing', 19:'Ruddy Shelduck', 20:'Rufous Treepie',21:'Sarus Crane',22:'White Wagtail',23:'White-Breasted Kingfisher',24:'White-Breasted Waterhen'}
    st.header("PREDICTION")
    st.markdown(dct[np.argmax(model.predict(img))])
