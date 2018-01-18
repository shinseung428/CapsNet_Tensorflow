import numpy as np
import tensorflow as tf

from architecture import *

epsilon = 1e-9
class CapsLayer(object):
	def __init__(self,batch_size):
		self.batch_size = batch_size

	def primaryCaps(self,
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

	def digitCaps(self,
				  caps,
				  num_outputs=10,
				  vec_length=16,
				  iter_routing=3,
				  name="digitcaps"):
		with tf.variable_scope(name) as scope:
			self.input = tf.reshape(caps, shape=(self.batch_size, -1, 1, caps.shape[-2].value, 1))

			b_IJ = tf.zeros([self.batch_size, caps.shape[1].value, num_outputs, 1, 1], dtype=tf.float32)
			capsules = self.routing(self.input, b_IJ, num_outputs, vec_length, iter_routing)
			capsules = tf.squeeze(capsules, axis=[1,4])
			return capsules

	def routing(self, input, b_IJ, out_num, vec_len, iter_routing):
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

			for routenum in range(iter_routing):
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


	def squash(self, vec):

		vec_squared_norm = tf.reduce_sum(tf.square(vec), -2, keep_dims=True)
		scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
		squashed = scalar_factor * vec
		return squashed