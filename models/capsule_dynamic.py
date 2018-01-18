import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

from glob import glob

import numpy as np
import os

from architecture import *
from CapsLayer import CapsLayer
from data.DataReader import *

epsilon = 1e-9

class capsule_dynamic():
	def __init__(self, args):

		self.graph_path = args.graph_path

		self.input_width = args.input_width
		self.input_height = args.input_height
		self.input_channel = args.input_channel

		self.output_dim = args.output_dim

		self.epochs = args.epochs
		self.batch_size = args.batch_size
		self.learning_rate = args.learning_rate
		self.momentum = args.momentum

		self.lambda_val = args.lambda_val
		self.m_plus = args.m_plus
		self.m_minus = args.m_minus
		self.reg_scale = args.reg_scale

		self.X, self.Y, self.data_count = load_data(args)
		self.build_model()
		self.build_loss()

		#summary
		self.img_summary = tf.summary.image("input", self.X, max_outputs=5)
		self.loss_summary = tf.summary.scalar("loss", tf.reduce_mean(self.loss))
		self.rec_img_summary = tf.summary.image("rec_img", self.rec_img, max_outputs=5)
		self.acc_summary = tf.summary.scalar("acc", self.accuracy)


	def build_model(self):
		self.CapsNetwork(self.X, name="capsnet")

		self.trainable_vars = tf.trainable_variables()
		print "number of parameters: ", count_param(self.trainable_vars)

	#implementation of dynamic routing between capsules
	def CapsNetwork(self, input_image, name):
		with tf.variable_scope(name) as scope:
			conv1 = tf.contrib.layers.conv2d(input_image, 256,
											 9, 1, padding="VALID",
											 activation_fn=tf.nn.relu,
											 scope="conv1"
											)			
			
			capsule = CapsLayer(self.batch_size)
			caps1 = capsule.dm_primaryCaps(conv1, 
										   kernel=9,
										   stride=2,
										   num_outputs=32,
										   vec_length=8,
										   name="primarycaps")
		

			caps2 = capsule.dm_digitCaps(caps1,
									     num_outputs=self.output_dim,
									     vec_length=16,
									     routing=3,
									     name="digitcaps_1")

			last_caps = caps2

			#Mask out all but the activity vector of the correct digit capsule
			with tf.variable_scope("Mask"):
				#masked_v will later be used to reconstruct images
				self.masked_v = tf.multiply(tf.squeeze(last_caps), tf.reshape(self.Y, (-1, last_caps.shape[1].value, 1)))

				#argmax_idx stores the predicted label
				self.v_length = tf.sqrt(tf.reduce_sum(tf.square(last_caps), axis=2) + epsilon)
				self.softmax_v = tf.nn.softmax(self.v_length)
				self.argmax_idx = tf.to_int32(tf.argmax(self.softmax_v, axis=1))
				self.argmax_idx = tf.reshape(self.argmax_idx, shape=(self.batch_size, ))


			#Decoder uses a masked vector to reconstruct images
			with tf.variable_scope("Decoder"):
				#fc version
				vector_j = tf.reshape(self.masked_v, shape=(self.batch_size, -1))
				fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=512,
														weights_initializer=tf.contrib.layers.xavier_initializer()
														)
				

				fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024,
														weights_initializer=tf.contrib.layers.xavier_initializer()
														)
				

				self.decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=self.input_width*self.input_height*self.input_channel, 
																 weights_initializer=tf.contrib.layers.xavier_initializer(),
																 activation_fn=tf.sigmoid)			

				#deconv version(optional)
				# vector_j = tf.reshape(self.masked_v, shape=(self.batch_size, -1))
				# fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=self.input_width/2*self.input_height/2*3,
				# 										weights_initializer=tf.contrib.layers.xavier_initializer()
				# 										)
				# deconv_ = tf.reshape(fc1, shape=(self.batch_size, self.input_width/2, self.input_height/2, self.input_channel))
				# self.decoded = tf.layers.conv2d_transpose(deconv_, 
				# 									filters=3, 
				# 									kernel_size=5,
				# 									strides=(2,2),
				# 									padding='SAME',
				# 									activation=tf.sigmoid)

				self.rec_img = tf.reshape(self.decoded, shape=(self.batch_size, self.input_width, self.input_height, self.input_channel))
		

	def build_loss(self):
		#loss function as decribed in the paper
		max_l = tf.square(tf.maximum(0., self.m_plus - self.v_length))
		max_r = tf.square(tf.maximum(0., self.v_length - self.m_minus))

		T_c = self.Y
		L_c = T_c * max_l + self.lambda_val * (1 - T_c) * max_r

		self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

		#calculate reconstruction loss
		origin = tf.reshape(self.X, shape=(self.batch_size, -1))
		squared = tf.square(self.decoded - origin)
		self.reconstruction_err = tf.reduce_mean(squared)

		self.loss = self.margin_loss + self.reg_scale * self.reconstruction_err

		#to check accuracy		
		gt = tf.cast(tf.argmax(self.Y, axis=1), tf.int32)
		pred = self.argmax_idx
		correct_prediction = tf.equal(pred, gt)

		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
