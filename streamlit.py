import streamlit as st
import requests
import time
import os
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np
import keras.utils as np_utils
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing import image
from keras.utils import image_dataset_from_directory
from keras.callbacks import ModelCheckpoint

st.set_page_config(page_title="Image Classifier", page_icon="🐬", layout="wide")

header = st.container()
EDA = st.container()
DL = st.container()
matrix = st.container()
with header:
    st.title("Image classifier")

with EDA:
    st.header("Performing exploratory data analysis")
    st.write("I wanted to work with the category: marine life. I chose to scrape images of dolphins, clownfish, starfish, jellyfish and sea turtles.")
    st.write("I didn't want to use categories that were too similar, so that's why I chose those.")
    st.write("It's also easy to find many more categories so, I can always add more that would fit the marine life category.")
    st.write("I will start by showing the EDA that I did and a few sample images.")
    col3, col4 = st.columns(2)
with col3:
    with st.expander(":blue[# of training images]", expanded=False):
        dataset_train_dir = './TaskDL/datasets/train'
        classes_tr = os.listdir(dataset_train_dir)
        st.write("Training: \n")
        for class_tr in classes_tr:
            class_path = os.path.join(dataset_train_dir, class_tr)
            num_images = len(os.listdir(class_path))
            st.write(f"{class_tr} class, Number of Images: {num_images}")
with col4:
    with st.expander(":blue[# of testing images]", expanded=False):
        dataset_test_dir = './TaskDL/datasets/val'
        classes_te = os.listdir(dataset_test_dir)
        st.write("Test: \n")
        for class_te in classes_te:
            class_path = os.path.join(dataset_test_dir, class_te)
            num_images = len(os.listdir(class_path))
            st.write(f"{class_te} class, Number of Images: {num_images}")
with col3:
    with st.expander(":blue[Example image of the training set]", expanded=False):
        #load and display the image
        image_path = './TaskDL/datasets/train/sea+turtle/2.jpg'
        img = image.load_img(image_path)
        st.image(img)

with col4:
    with st.expander(":blue[Example image of the test set]", expanded=False):
        #load and display the image
        image_path = './TaskDL/datasets/val/dolphin/172.jpg'
        img = image.load_img(image_path)
        st.image(img)



train_val_datagen = ImageDataGenerator(validation_split=0.2,
                                   rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_val_datagen.flow_from_directory('./TaskDL/datasets/train',
                                                 subset='training',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

validation_set = train_val_datagen.flow_from_directory('./TaskDL/datasets/train',
                                                 subset='validation',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('./TaskDL/datasets/val',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


#initialize the CNN
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64,(3,3), activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128,(3,3), activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dropout(0.2))

model.add(Dense(activation='relu', units=128))
model.add(Dropout(0.2))

model.add(Dense(activation='softmax', units=5))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

with DL:
    with col3:
        st.header("Training the model")
        st.write("Here, you can change the number of epochs you want to train on my model.")
        number = col3.slider("Number of epochs", min_value=5, max_value=30, value=15, step=5)

        batches = col3.selectbox("Number of batches", ('16', '32', '64'), index=2)
        history = model.fit(training_set,
                validation_data = validation_set,
                batch_size = batches,
                epochs = number
                )

        with st.expander(":blue[Loss and accuracy curves]", expanded=True):
            # Create a figure and a grid of subplots with a single call
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

            # Plot the loss curves on the first subplot
            ax1.plot(history.history['loss'], label='training loss')
            ax1.plot(history.history['val_loss'], label='validation loss')
            ax1.set_title('Loss curves')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()

            # Plot the accuracy curves on the second subplot
            ax2.plot(history.history['accuracy'], label='training accuracy')
            ax2.plot(history.history['val_accuracy'], label='validation accuracy')
            ax2.set_title('Accuracy curves')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()

            # Adjust the spacing between subplots
            fig.tight_layout()
            # Show the figure
            st.pyplot(fig)






footer="""<style>

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}


</style>
<div class="footer">
<p>Iris Loret - 2023</p>
</div>


"""
st.markdown(footer,unsafe_allow_html=True)
    
