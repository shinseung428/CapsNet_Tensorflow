import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

from glob import glob

import numpy as np
import os

from architecture import *
from DataReader import *

class normal_classifier():
	def __init__(self, args):
		self.input_width = args.input_width
		self.input_height = args.input_height
		self.input_channel = args.input_channel

		self.output_dim = args.output_dim

		self.batch_size = args.batch_size
		self.learning_rate = args.learning_rate
		self.momentum = args.momentum


		self.X, self.Y, self.data_count = load_data(args)
		self.build_model()
		self.build_loss()

	# def build_data_reader(self):
	# 	def create_label_data(path, label=0):
	# 		file_list = glob(path)
	# 		data_count = len(file_list)

	# 		one_hot = np.zeros((data_count, self.output_dim))
	# 		one_hot[:,label] = 1

	# 		return file_list, one_hot

	# 	cats_filename, cats_labels = create_label_data(self.cats_file_path, label=0)
	# 	dogs_filename, dogs_labels = create_label_data(self.dogs_file_path, label=1)

	# 	all_filepaths = cats_filename + dogs_filename
	# 	all_labels = np.concatenate((cats_labels, dogs_labels), 0)

	# 	all_images = tf.convert_to_tensor(all_filepaths, dtype=tf.string)
	# 	all_labels = tf.convert_to_tensor(all_labels, dtype=tf.int32)

	# 	train_input_queue = tf.train.slice_input_producer([all_images, all_labels],shuffle=True)

	# 	file_content = tf.read_file(train_input_queue[0])
	# 	train_images = tf.image.decode_jpeg(file_content, channels=self.input_channel)
	# 	train_labels = train_input_queue[1]


	# 	#set image shape and normalize image input
	# 	train_images = tf.image.resize_images(train_images,[self.input_width, self.input_height])
	# 	train_images.set_shape((self.input_width, self.input_height, self.input_channel))
	# 	train_images = tf.cast(train_images, tf.float32) / 255.0

	# 	self.train_image_batch, self.train_label_batch = tf.train.batch([train_images, train_labels],
	# 																	batch_size=self.batch_size
	# 																    )

	def build_model(self):
		self.classifier, self.layers = self.model(self.X, name="classifier")
		self.trainable_vars = tf.trainable_variables()

	def build_loss(self):
		def calculate_loss(labels, logits):
			return tf.losses.softmax_cross_entropy(labels, logits)
		self.loss = calculate_loss(self.train_label_batch, self.classifier)


	def model(self, input_image, name):
		layers = []
		with tf.variable_scope(name) as scope:
			input = tf.reshape(self.input_image, shape=(self.batch_size, -1))
			fc1 = tf.contrib.layers.fully_connected(input, num_outputs=128,
													weights_initializer=tf.contrib.layers.xavier_initializer(),
													activation_fn=tf.nn.relu
													)

			fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=64,
													weights_initializer=tf.contrib.layers.xavier_initializer(),
													activation_fn=tf.nn.relu
													)

			fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=self.output_dim,
													weights_initializer=tf.contrib.layers.xavier_initializer(),
													activation_fn = tf.nn.sigmoid
													)

			return output, layers