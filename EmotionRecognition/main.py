import os 
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import keras 
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

def create_dataset(img_folder):
    # dataset of images
    img_data_array=[]
    # list of facial emotions
    class_name=[]
    # each image is 48x48 pixels
    IMG_WIDTH = 48
    IMG_HEIGHT = 48
    # iterate through each emotion folder in training folder
    for dir in os.listdir(img_folder):
        # iterate through each image in emotion folder
        for file in os.listdir(os.path.join(img_folder, dir)):
            # get path of image
            image_path = os.path.join(img_folder, dir, file)
            # read in image and convert to RGB
            image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            # resize image to fit within pixel boundaries
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv2.INTER_AREA)
            # save pixel values into array
            image = np.array(image)
            # make each pixel value a float
            image = image.astype('float32')
            # each pixel value originally falls between 0-255, make it fall between 0-1
            image /= 255
            # add pixel array to list
            img_data_array.append(image)
            # add emotion to list
            class_name.append(dir)
    return img_data_array, class_name, IMG_WIDTH, IMG_HEIGHT

def create_target_dict(class_name):
    target_dict = {k: v for v, k in enumerate(np.unique(class_name))}
    target_val = [target_dict[class_name[i]] for i in range(len(class_name))]
    return target_val

def create_model(image_width, image_height):
    model = keras.Sequential(
        [
         keras.layers.InputLayer(input_shape=(image_height,image_height,3)), 
         keras.layers.Conv2D(filters=32, kerner_size=3, strides=(2,2), activation='relu'), 
         keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2,2), activation='relu'), 
         keras.layers.Flatten(), 
         keras.layers.Dense(6)
        ])
    encoder.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model 

def train_model(learning_model, img_data, target_val):
    history = learning_model.fit(
        x = np.array(img_data, np.float32), 
        y = np.array(list(map(int, target_val)), np.float32), epochs = 5
        )

if __name__ == '__main__':
    # img_data is dataset of length 28,709 (that's how many training images there are)
    # img_data[0].shape is (48,48) each image has 256 pixel values between 0 and 1
    img_data, class_name, image_width, image_height = create_dataset('../train')
    target_val = create_target_dict(class_name)
    learning_model = create_model(image_width, image_height)
    train_model(learning_model, img_data, target_val)