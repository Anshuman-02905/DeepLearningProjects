import util
import streamlit as st

st.title("Traffic Light Recognition")
st.image('Traffic_Signal_Recognition/Traffic_lights.jpg', channels="BGR")

dct=util.get_labels('Traffic_Signal_Recognition/Labels.json')
model=util.get_model('Traffic_Signal_Recognition/model_GTSRB')

st.write("I have developed a Traffic Sign Detection model using Keras and Convolutional Neural Network (CNN) techniques. The model has been trained on the GTSRB (German Traffic Sign Recognition Benchmark) dataset, which contains 50,000 images of traffic signs belonging to 40 different classes.")
st.write("To build the model, I started by preprocessing the dataset to normalize the images and resize them to a uniform size. I then split the data into training, validation, and testing sets to evaluate the model's performance.")
st.write("The CNN model architecture consists of several convolutional layers, followed by max-pooling and dropout layers to prevent overfitting. I also added several fully connected layers with softmax activation for multi-class classification.")
st.write("During training, the model learns to detect and classify the different traffic signs based on their visual features, such as shape, color, and symbol. I used the categorical cross-entropy loss function and the Adam optimizer to optimize the model's performance.")
st.write("After training, I evaluated the model on the testing set and achieved high accuracy in detecting and classifying the different traffic signs. I also performed some additional analysis, such as confusion matrix and classification report, to assess the model's performance for each class.")
st.write("The Traffic Sign Detection model has many practical applications, such as enhancing autonomous driving systems, improving road safety, and assisting traffic law enforcement.")
st.write("NOTE The input image should be have equal resolution that means the length and width of image should be same. If the image is not in 1:1 ratio then do the same from the following website ->https://www.img2go.com/crop-image")
img = st.file_uploader("Choose a file")
if st.button("submit Image"):
    img=util.convert_image(img)
    prediction = util.model_predict(model, img)
    st.header("PREDICTION")
    st.write(dct[prediction])

