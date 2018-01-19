from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

from glob import glob

import numpy as np
import os

from architecture import *
from CapsLayer import CapsLayer
from data.DataReader import *

class capsule_em():
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
		self.CapsNetwork(self.X, name="capsnet_em")

		self.trainable_vars = tf.trainable_variables()
		print "number of parameters: ", count_param(self.trainable_vars)


	def CapsNetwork(self, input, name="capsnet"):
		with tf.variable_scope(name) as scope:
			#first layer before creating primary caps layer
			conv1 = tf.contrib.layers.conv2d(input, 32, 5, 2,
											 padding="VALID",
											 activation_fn=tf.nn.relu,
											 scope="conv1")


			capsule = CapsLayer(self.batch_size)

			#Make Primary Capsule 
			primary_pose, primary_actv = capsule.em_primaryCaps(conv1, kernel=1, stride=1, num_outputs=32, name="primaryCaps")

			#Make Convolution Capsule Layer 1 & 2
			convCaps1 = capsule.em_convCaps(primary_pose,
											primary_actv, 
											kernel=3, stride=2, 
											num_outputs=32, routing=3, 
											name="convCaps1")
			# convCaps2 = capsule.em_convCaps(convCaps1, kernel=3, stride=1, num_outputs=32, routing=3, name="convCaps1")
			
			
			input('done')

	def build_loss(self):
		pass