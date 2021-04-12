import os 
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Layer, Lambda
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
    return img_data_array, class_name

if __name__ == '__main__':
    #emotion = menu()
    #show_image(emotion)
    # img_data is dataset of length 28,709 (that's how many training images there are)
    # img_data[0].shape is (48,48) each image has 256 pixel values between 0 and 1
    img_data, class_name = create_dataset('../train')
