import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow.keras import layers, Input 
from tensorflow.keras.layers import Dense, InputLayer, Flatten
from tensorflow.keras.models import Sequential, Model
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg

def create_dataset(img_folder):
    img_data_array=[]
    class_name=[]
    IMG_WIDTH = 48
    IMG_HEIGHT = 48
    for dir in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir)):
            image_path = os.path.join(img_folder, dir, file)
            image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv2.INTER_AREA)
            image = np.array(image)
            image = image.astype('float32')
            image /= 255
            img_data_array.append(image)
        class_name.append(dir)
    return img_data_array, class_name


if __name__ == '__main__':
    #emotion = menu()
    #show_image(emotion)
    img_data, class_name = create_dataset('C:\\Users\\steve\\Documents\\GitHub\\EmotionRecognition\\train')
    print(img_data[0].shape)