from __future__ import division, absolute_import, print_function, unicode_literals

# Common inputs
import numpy as np
import matplotlib.pyplot as plt

# Tensorflow and keras inputs
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

# Import the MNIST dataset
mnist_dataset = tf.keras.datasets.fashion_mnist

# Load the traing and test images and labes
(train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()

''' Loading the dataset results in a numpy array where the:
* train_images and train_labels are the traing set
* test_images and test_labels are the test set
the model (the train set) is tested against the test set '''

# The ten classes in the MNIST dataset are

class_cat = [
'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Explore the data
train_images.shape
len(train_labels)
train_labels

test_images.shape
len(test_labels)
test_labels

''' Preprocess the data '''

plt.figure()
plt.imshow(train_images[10])
plt.colorbar()
plt.grid(False)
plt.show()
