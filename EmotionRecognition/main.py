import os 
import time

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf

import keras # program crashes with this import
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from keras.utils import np_utils

import sklearn 
from sklearn.metrics import classification_report 
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import cv2
import random

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
            image = cv2.imread(image_path)
            # resize image to fit within pixel boundaries
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
            # save pixel values into array
            image = np.array(image)
            # make each pixel value a float
            image = image.astype('float32')
            # convert pixels to values between 0 and 1
            image /= 255
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
    return npImageDataArray, imageLabels


# how gets the labels for each image (the Y_Train / Y_Test values)
def get_image_labels(class_name):
    # convert into diction with key class name and value int label
    target_dict = {k: v for v, k in enumerate(np.unique(class_name))}
    # array of int label values
    target_val = [target_dict[class_name[i]] for i in range(len(class_name))]
    # convert the array into numpy arrays
    npLabels = np_utils.to_categorical(target_val, 7)
    return npLabels

# using keras to develop the training model
def create_simple_model(train_images):

    model = Sequential()

    model.add(Conv2D(32, 3, strides = 1, padding='same',
                     input_shape=(train_images.shape[1:])))
    model.add(Activation('relu'))

    model.add(Conv2D(64, 3, strides = 1, padding='same'))
    model.add(Activation('relu'))

    ## 2x2 max pooling reduces to 14 x 14 x 64
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, 3, strides = 1, padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(128, 3, strides = 1, padding='same'))
    model.add(Activation('relu'))

    ## 2x2 max pooling reduces to 14 x 14 x 64
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    ## Flatten turns 14x14x64 into 12,544
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(7))
    model.add(Activation('softmax'))

    print("------------------------------------------")
    print("MODEL SUMMARY")
    print("------------------------------------------")
    model.summary()
    return model

def train_simple_model(learning_model, train_data, train_label, test_data, test_label):
    
    batch_size = 32
    # initiate RMSprop optimizer
    opt = keras.optimizers.RMSprop(learning_rate=0.0005, decay=1e-6)
    # Let's train the model using RMSprop
    learning_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    history = learning_model.fit(train_data, train_label, batch_size=batch_size, epochs=5, validation_data=(test_data, test_label), shuffle=True)
    return history

def perform_simple_analytics(history, learning_model, test_data, test_label):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(5)

    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    predictions = learning_model.predict(test_data)
    y_pred = np.argmax(predictions, axis=1)

    rounded_labels = np.argmax(test_label, axis=1)
    report = classification_report(rounded_labels, y_pred)
    print(report)

def process_images(image, label):

    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (227,227))
    return image, label

def create_alex_model(train_images, train_labels, test_images, test_labels):

    # emotion Lables
    emotionNames = ['angry','disgust','fear','happy','neutral','sad','surprise']
    
    # validation set
    valid_images = train_images[:5000]
    valid_labels = train_labels[:5000]

    # training set
    train_images = train_images[5000:]
    train_labels = train_labels[5000:]

    train_data = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_data = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    valid_data = tf.data.Dataset.from_tensor_slices((valid_images, valid_labels))

    train_data_size = tf.data.experimental.cardinality(train_data).numpy()
    test_data_size = tf.data.experimental.cardinality(test_data).numpy()
    valid_data_size = tf.data.experimental.cardinality(valid_data).numpy()

    train_data = (train_data
                  .map(process_images)
                  .shuffle(buffer_size=train_data_size)
                  .batch(batch_size=32, drop_remainder=True))

    test_data = (test_data
                  .map(process_images)
                  .shuffle(buffer_size=train_data_size)
                  .batch(batch_size=32, drop_remainder=True))

    valid_data = (valid_data
                  .map(process_images)
                  .shuffle(buffer_size=train_data_size)
                  .batch(batch_size=32, drop_remainder=True))

    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(7, activation='softmax')    
    ])

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
    model.summary()

    model.fit(train_data, epochs=5, validation_data=valid_data, validation_freq=1)
    model.evaluate(test_data)

if __name__ == '__main__':

    print("Emotion Recognition Image Classifier.")
    print("------------------------------------------")

    print("Begin loading training dataset...")
    train_img_data, train_img_labels = load_dataset('../train')
    print("Training dataset has been loaded...")

    print("------------------------------------------")

    print("Begin loading testing dataset...")
    test_img_data, test_img_labels = load_dataset('../test')
    print("Testing dataset has been loaded...")

    print("------------------------------------------")

    #print("Creating simple model...")
    #learning_model_simple = create_simple_model(train_img_data)
    #print("Simple model created...")

    #print("------------------------------------------")

    #print("Training simple model...")
    #history = train_simple_model(learning_model_simple, train_img_data, train_img_labels, test_img_data, test_img_labels)
    #print("Simple model training finished...")

    #print("------------------------------------------")

    #print("Performing analytics on simple model...")
    #perform_simple_analytics(history, learning_model_simple, test_img_data, test_img_labels)

    print("Creating AlexNet model...")
    learning_model_alex = create_alex_model(train_img_data, train_img_labels, test_img_data, test_img_labels)