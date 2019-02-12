from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras

import numpy as np 
import matplotlib.pyplot as plt

import logging

logging.basicConfig(format="%(message)s", level=logging.DEBUG, filename="debug.log", filemode="w")


def main():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_img, train_label), (test_img, test_label) = fashion_mnist.load_data()
    # print(train_img)
    class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_images = train_img / 255.0
    test_images = test_img / 255.0 

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])  
    model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    model.fit(train_images, train_label, epochs=20)

    test_loss, test_acc = model.evaluate(test_images, test_label)

    print('Test accuracy:', test_acc, 'Test loss: ', test_loss)

main() 

