import os
import numpy as np
import tensorflow as tf
from utils import resize_image, print_section
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical


class CNNApproach:
	def __init__(self, name, model, image_size):
		self.name = name
		self.image_size = image_size

		self.model = model
		self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


	def preprocess_images(self, image_paths):
		images = list(map(lambda p: resize_image(p, self.image_size), image_paths))
		images = self.normalize(np.array(images))

		return images


	def normalize(self, x):
		return (x / 255) - 0.5


	def train(self, x, y):
		print_section('train model: ' + self.name)

		self.model.fit(self.preprocess_images(x), to_categorical(y), epochs=10)


	def test(self, x):
		print_section('test model: ' + self.name)

		return np.argmax(self.model.predict(self.preprocess_images(x)), axis=1)


class AlexNet(CNNApproach):
	def __init__(self, num_classes):
		title = 'AlexNet'
		image_size = 224
		model = create_alexnet(num_classes, image_size)
		super().__init__(title, model, image_size)


class AlexNetLarger(CNNApproach):
	def __init__(self, num_classes):
		title = 'AlexNet Larger'
		image_size = 680
		model = create_alexnet(num_classes, image_size)
		super().__init__(title, model, image_size)


def create_alexnet(num_classes, image_size):
	""" adapted from source: https://towardsdatascience.com/implementing-alexnet-cnn-architecture-using-tensorflow-2-0-and-keras-2113e090ad98 """
	model = Sequential([
		Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(image_size,image_size,3)),
		BatchNormalization(),
		MaxPooling2D(pool_size=(3,3), strides=(2,2)),
		Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
		BatchNormalization(),
		MaxPooling2D(pool_size=(3,3), strides=(2,2)),
		Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
		BatchNormalization(),
		Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
		BatchNormalization(),
		Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
		BatchNormalization(),
		MaxPooling2D(pool_size=(3,3), strides=(2,2)),
		Flatten(),
		Dense(4096, activation='relu'),
		Dropout(0.5),
		Dense(4096, activation='relu'),
		Dropout(0.5),
		Dense(num_classes, activation='softmax')
	])
	model.summary()

	return model
