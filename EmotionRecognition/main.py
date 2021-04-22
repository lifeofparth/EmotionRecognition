import os 
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
#import keras # program crashes with this import
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import random

# emotion Lables
emotionNames = ['angry','disgust','fear','happy','neutral','sad','surprise']


# extracts the labels and images for the emotions (angry, sad, etc...)
# and each images width and height 48 x 48
# used on both the train and test folders
def load_dataset(img_folder):

    # dataset of images as a numpy array as shown in the instructions
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
            # image /= 255

            # put all of the images into a numpy array
            npImage = np.array(image)

            # add pixel array to list
            img_data_array.append(npImage)
            # add emotion to list
            class_name.append(dir)

    # turns the image data arrays into an np array
    npImageDataArray = np.array(img_data_array)

    # gets the labels for the images in the arrays
    imageLabels = get_image_labels(class_name)
    
    return npImageDataArray, imageLabels, IMG_WIDTH, IMG_HEIGHT


# how gets the labels for each image (the Y_Train / Y_Test values)
def get_image_labels(class_name):
    target_dict = {k: v for v, k in enumerate(np.unique(class_name))}
    target_val = [target_dict[class_name[i]] for i in range(len(class_name))]
    
    # convert the array into numpy arrays
    npLabels = np.array(target_val)

    return npLabels


# displays a (48,48) image with its label
def showImage(image, label):
    imgplot = plt.imshow(image)
    plt.title("Emotion: {0:}".format(emotionNames[label]))
    plt.colorbar()
    plt.grid(False)
    plt.show()
    plt.close()


# using keras to develop the training model
def create_model(image_width, image_height):

    model = tf.keras.Sequential([ # makes the training model
    tf.keras.layers.Flatten(input_shape=(image_width, image_height)), # makes the imput shape 48 * 48
    tf.keras.layers.Dense(180, activation='relu'), # 128 nodes
    tf.keras.layers.Dense(7) # output of 7
    ])

    # for the compile, added settings for: loss function (accuracy during model training), 
    # optimizer (how model is updated),
    # and metrics (monitors training and testing steps)
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model 


if __name__ == '__main__':

    # there are 7 types of emotion labels (angry, sad, etc..)
    # img_data is dataset of length 28,709 (that's how many training images there are)
    # img_data[0].shape is (48,48) each image has 256 pixel values between 0 and 1
    
    # 1. extract the data for the train and test sets and have the data be in the numpy array format
    # where each image array (48,48) is in each emotion array that is in the train/ test array
    train_img_data, train_img_labels, image_width, image_height = load_dataset('../train')
    test_img_data, test_img_labels, image_width2, image_height2 = load_dataset('../test')

    # 2. test the dataset by comparing the labels with their images, 10 random images in the training set
    # to confirm the dataset was correclty extracted
    for i in range(10):
        randPos = random.randrange(0,len(train_img_data))
        showImage(train_img_data[randPos], train_img_labels[randPos])

    # 3. create the learning model
    learning_model = create_model(image_width, image_height)

    # 4. train the model
    learning_model.fit(train_img_data, train_img_labels, epochs = 10)

    # 5. evaluate the accuracy
    test_loss, test_acc = learning_model.evaluate(test_img_data,  test_img_labels, verbose=2)
    print('\nTest accuracy:', test_acc)