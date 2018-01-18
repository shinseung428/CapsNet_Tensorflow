import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

from glob import glob

import numpy as np
import os

from architecture import *
from data.DataReader import *

class baseline_network():
	def __init__(self, args):
		self.input_width = args.input_width
		self.input_height = args.input_height
		self.input_channel = args.input_channel

		self.output_dim = args.output_dim

		self.batch_size = args.batch_size
		self.learning_rate = args.learning_rate
		self.momentum = args.momentum

		#load image data X and label data Y
		self.X, self.Y, self.data_count = load_data(args)
		self.build_model()
		self.build_loss()

		#summary
		self.img_summary = tf.summary.image("input", self.X, max_outputs=5)
		self.acc_summary = tf.summary.scalar("acc", self.accuracy)
		self.loss_summary = tf.summary.scalar("loss", self.loss)

	def build_model(self):
		self.classifier, self.classifier_logits = self.model(self.X, name="classifier")

		self.trainable_vars = tf.trainable_variables()
		print "number of parameters: ", count_param(self.trainable_vars)

	def build_loss(self):
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.classifier_logits))

		#to check accuracy
		gt = tf.argmax(self.Y, axis=1)
		pred = tf.argmax(self.classifier, axis=1)
		correct_prediction = tf.equal(pred, gt)
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


	#baseline network as described in the paper
	def model(self, input_image, name):
		with tf.variable_scope(name) as scope:
			conv1 = tf.contrib.layers.conv2d(input_image, 256,
											 5, 1, padding="VALID",
											 weights_initializer=tf.contrib.layers.xavier_initializer(),
											 activation_fn=tf.nn.relu,
											 scope="conv1"
											)

			conv1 = batch_norm(conv1, name="bn1")

			conv2 = tf.contrib.layers.conv2d(conv1, 256,
											 5, 1, padding="VALID",
											 weights_initializer=tf.contrib.layers.xavier_initializer(),
											 activation_fn=tf.nn.relu,
											 scope="conv2"
											)

			conv2 = batch_norm(conv2, name="bn2")

			conv3 = tf.contrib.layers.conv2d(conv2, 128,
											 5, 1, padding="VALID",
											 weights_initializer=tf.contrib.layers.xavier_initializer(),
											 activation_fn=tf.nn.relu,
											 scope="conv3"
											)	

			conv3 = batch_norm(conv3, name="bn3")
			
			flattened = tf.contrib.layers.flatten(conv3)
			fc1 = tf.contrib.layers.fully_connected(flattened, num_outputs=328,
													weights_initializer=tf.contrib.layers.xavier_initializer(),
													activation_fn=tf.nn.relu,
													scope="fc1")

			fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=192,
													weights_initializer=tf.contrib.layers.xavier_initializer(),
													activation_fn=tf.nn.relu,
													scope="fc2"
													)

			fc2 = tf.layers.dropout(fc2, rate=0.5)
			fc3 = tf.contrib.layers.fully_connected(fc2, num_outputs=self.output_dim,
													weights_initializer=tf.contrib.layers.xavier_initializer(),
													scope="fc3"
													)
			
			output = tf.nn.softmax(fc3)

			return output, fc3




