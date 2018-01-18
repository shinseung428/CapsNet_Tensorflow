import numpy as np
import tensorflow as tf

from architecture import *

epsilon = 1e-9
class CapsLayer(object):
	def __init__(self,batch_size):
		self.batch_size = batch_size


#Functions for dynamic routing
	def dm_primaryCaps(self,
					   input, 
					   kernel=9,
					   stride=2,
					   num_outputs=32,
					   vec_length=8,
					   name="primarycaps"):
		with tf.variable_scope(name) as scope:
			capsules = tf.contrib.layers.conv2d(input, num_outputs*vec_length,
												kernel, stride, padding="VALID",
												activation_fn=tf.nn.relu,
												scope="conv"
												)
			capsules = batch_norm(capsules, "batch_norm1")
			capsules = tf.reshape(capsules, (self.batch_size, -1, vec_length, 1))

		return capsules

	def dm_digitCaps(self,
				  	 caps,
				  	 num_outputs=10,
				  	 vec_length=16,
				  	 routing=3,
				  	 name="digitcaps"):
		with tf.variable_scope(name) as scope:
			self.input = tf.reshape(caps, shape=(self.batch_size, -1, 1, caps.shape[-2].value, 1))

			b_IJ = tf.zeros([self.batch_size, caps.shape[1].value, num_outputs, 1, 1], dtype=tf.float32)
			capsules = self.dm_routing(self.input, b_IJ, num_outputs, vec_length, routing)
			capsules = tf.squeeze(capsules, axis=[1,4])
			return capsules

	def dm_routing(self, input, b_IJ, out_num, vec_len, routing):
		with tf.variable_scope("routing") as scope:
			#8x16 weight matrices in the paper
			#shape of this matrix can be changed freely
			weight = tf.get_variable("Weight", shape=[1, input.shape[1].value, out_num, input.shape[3].value, vec_len], dtype=tf.float32,
									 initializer=tf.contrib.layers.xavier_initializer())
									 
			#tile tensors to match dimensions
			input = tf.tile(input, [1, 1, out_num, 1, 1])
			weight = tf.tile(weight, [self.batch_size, 1, 1, 1, 1])

			u_hat = tf.matmul(weight, input, transpose_a=True)
			u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

			for routenum in range(routing):
				with tf.variable_scope('iter' + str(routenum)):

					#calculate c_ij
					c_IJ = tf.nn.softmax(b_IJ, dim=2)

					if routenum == routing - 1:
						#calculate s_J
						s_J = tf.multiply(c_IJ, u_hat)
						s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
						v_J = self.dm_squash(s_J)

					elif routenum < routing - 1:
						s_J = tf.multiply(c_IJ, u_hat_stopped)
						s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
						v_J = self.dm_squash(s_J)

						v_J_tiled = tf.tile(v_J, [1, input.shape[1].value, 1, 1, 1])
						u_produce_v = tf.matmul(u_hat_stopped, v_J_tiled, transpose_a=True)
						b_IJ += u_produce_v

			return v_J


	def dm_squash(self, vec):

		vec_squared_norm = tf.reduce_sum(tf.square(vec), -2, keep_dims=True)
		scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
		squashed = scalar_factor * vec
		return squashed

	#functions for EM routing 
	def em_primaryCaps(self,
					input, 
					kernel=9,
					stride=2,
					num_outputs=32,
					name="primarycaps"):
		with tf.variable_scope(name) as scope:

			primaryCaps_pose = tf.contrib.layers.conv2d(input, num_outputs*(4*4), kernel, stride,
												   		padding="SAME"
												   		)
			primaryCaps_pose = tf.reshape(primaryCaps_pose, 
										   (self.batch_size, input.shape[1].value, input.shape[2].value, num_outputs, 4*4))

			primaryCaps_actv = tf.contrib.layers.conv2d(input, num_outputs, kernel, stride,
														activation_fn=tf.nn.sigmoid,
												   		padding="SAME",
												   	   )
			primaryCaps_actv = tf.reshape(primaryCaps_actv,
										  (self.batch_size, input.shape[1].value, input.shape[2].value, num_outputs, 1))


			#concatenate pose and activation as one capsule
			primaryCaps = tf.concat([primaryCaps_pose, primaryCaps_actv], axis=4)
			primaryCaps = tf.reshape(primaryCaps, (self.batch_size, input.shape[1].value, input.shape[2].value, -1))
		

		return primaryCaps

	def em_convCaps(self,
				 caps,
				 kernel=3,
				 stride=2,
				 num_outputs=32,
				 routing=3,
				 name="convCaps"):
		with tf.variable_scope(name) as scope:

			print caps
			caps_shape = caps.get_shape()
			filter_shape = [kernel, kernel, caps_shape[3], kernel*kernel]
			
			#conv_filter shape = [3, 3, 544, 9]
			conv_filter = np.zeros(shape = filter_shape, dtype=np.float32)
			
			for i in range(0,kernel):
				for j in range(0,kernel):
					conv_filter[i, j, :, i*kernel+j] = 1.0

			conv_filter_op = tf.constant(conv_filter, dtype=tf.float32)

			output = tf.nn.depthwise_conv2d(caps, conv_filter_op,
											strides=[1,stride,stride,1],
											padding="VALID"
											)
			print output
			out_shape = output.get_shape()
			output = tf.reshape(output, 
								shape=[int(out_shape[0]), int(out_shape[1]), int(out_shape[2]), int(caps_shape[3]), kernel * kernel])
			print output
			output = tf.transpose(output, perm=[0, 1, 2, 4, 3])
			print output


			input('Done')
			#======================================================================
			return capsules

	def em_routing(self, input, b_IJ, out_num, vec_len, routing):
		with tf.variable_scope("routing") as scope:
			weight = tf.get_variable("Weight", shape=[1, input.shape[1].value, out_num, input.shape[3].value, vec_len], dtype=tf.float32,
									 initializer=tf.contrib.layers.xavier_initializer())
									 # initializer=tf.random_normal_initializer(stddev=0.01))
									 

			input = tf.tile(input, [1, 1, out_num, 1, 1])
			weight = tf.tile(weight, [self.batch_size, 1, 1, 1, 1])

			u_hat = tf.matmul(weight, input, transpose_a=True)
			
			u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

			for routenum in range(routing):
				with tf.variable_scope('iter' + str(routenum)):

					#calculate c_ij
					c_IJ = tf.nn.softmax(b_IJ, dim=2)

					if routenum == iter_routing - 1:
						#calculate s_J
						s_J = tf.multiply(c_IJ, u_hat)
						s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
						v_J = self.squash(s_J)

					elif routenum < iter_routing - 1:
						s_J = tf.multiply(c_IJ, u_hat_stopped)
						s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
						v_J = self.squash(s_J)

						v_J_tiled = tf.tile(v_J, [1, input.shape[1].value, 1, 1, 1])
						u_produce_v = tf.matmul(u_hat_stopped, v_J_tiled, transpose_a=True)
						b_IJ += u_produce_v

			return v_J


	def em_squash(self, vec):

		vec_squared_norm = tf.reduce_sum(tf.square(vec), -2, keep_dims=True)
		scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
		squashed = scalar_factor * vec
		return squashed