from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

from glob import glob

import numpy as np
import os

from architecture import *

class capsule_em():
	def __init__(self, sess, args):
		self.graph_path = args.graph_path

		self.input_width = args.input_width
		self.input_height = args.input_height
		self.input_channel = args.input_channel

		self.output_dim = args.output_dim

		self.epochs = args.epochs
		self.batch_size = args.batch_size
		self.learning_rate = args.learning_rate
		self.momentum = args.momentum

		self.mask_with_y = args.mask_with_y
		self.lambda_val = args.lambda_val
		self.m_plus = args.m_plus
		self.m_minus = args.m_minus
		self.reg_scale = args.reg_scale

		self.sess = sess

		self.cats_file_path = args.cats_train_path
		self.dogs_file_path = args.dogs_train_path
		if not args.is_train:
			self.cats_file_path = args.cats_test_path
			self.dogs_file_path = args.dogs_test_path
		self.data_loader()
		self.build_model()
		self.build_loss()

	def data_loader(self):
		def create_label_data(path, label=0):
			file_list = glob(path)
			data_count = len(file_list)

			one_hot = np.zeros((data_count, self.output_dim))
			one_hot[:,label] = 1

			return file_list, one_hot, data_count

		cats_filename, cats_labels, cats_count = create_label_data(self.cats_file_path, label=0)
		dogs_filename, dogs_labels, dogs_count = create_label_data(self.dogs_file_path, label=1)

		self.data_count = cats_count + dogs_count

		all_filepaths = cats_filename + dogs_filename
		all_labels = np.concatenate((cats_labels, dogs_labels), 0)

		all_images = tf.convert_to_tensor(all_filepaths, dtype=tf.string)
		all_labels = tf.convert_to_tensor(all_labels, dtype=tf.float32)

		train_input_queue = tf.train.slice_input_producer([all_images, all_labels],shuffle=True)

		file_content = tf.read_file(train_input_queue[0])
		train_images = tf.image.decode_jpeg(file_content, channels=self.input_channel)
		train_labels = train_input_queue[1]


		#set image shape and normalize image input
		train_images = tf.image.resize_images(train_images,[self.input_width, self.input_height])
		train_images.set_shape((self.input_width, self.input_height, self.input_channel))
		train_images = tf.cast(train_images, tf.float32) / 255.0

		self.X, self.Y = tf.train.batch([train_images, train_labels],
														batch_size=self.batch_size
														)

	def build_model(self):

		self.CapsNetwork(self.X, name="capsnet")


	def CapsNetwork(self, input, name="capsnet"):
		with tf.variable_scope(name) as scope:
			conv1 = tf.contrib.layers.conv2d(input, 32, 5, 2,
											 padding="VALID",
											 activation_fn=tf.nn.relu,
											 scope="conv1")


			#Make Primary Capsule 
			primaryCaps_pose = tf.contrib.layers.conv2d(conv1, 32*(4*4), 1, 1,
												   		padding="SAME",
												   		)
			primaryCaps_pose = tf.reshape(primaryCaps_pose, 
										   (self.batch_size, conv1.shape[1].value, conv1.shape[2].value, 32, 4*4))

			primaryCaps_actv = tf.contrib.layers.conv2d(conv1, 32, 1, 1,
												   		 padding="SAME",
												   	   )
			primaryCaps_actv = tf.reshape(primaryCaps_actv,
										  (self.batch_size, conv1.shape[1].value, conv1.shape[2].value, 32, 1))

			primaryCaps = tf.concat([primaryCaps_pose, primaryCaps_actv], axis=4)
			primaryCaps = tf.reshape(primaryCaps, (self.batch_size, conv1.shape[1].value, conv1.shape[2].value, -1))
			

			#Make Convolution Capsule Layer 1
			convCaps1_trans = tf.contrib.layers.conv2d(primaryCaps, , 3, 2,
													   padding="VALID")
			convCaps1_actv = tf.contrib.layers.conv2d(primaryCaps, , 3, 2,
													  padding="VALID")

			#Make Convolution Capsule Layer 2
			


	def build_loss(self):
		pass