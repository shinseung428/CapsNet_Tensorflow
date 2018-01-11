import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

from glob import glob

import numpy as np
import os

from architecture import *
from CapsLayer import CapsLayer
from DataReader import *

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

		self.mask_with_y = args.mask_with_y
		self.lambda_val = args.lambda_val
		self.m_plus = args.m_plus
		self.m_minus = args.m_minus
		self.reg_scale = args.reg_scale

		# self.sess = sess

		self.X, self.Y, self.data_count = load_data(args)
		self.build_model()
		self.build_loss()


	def build_model(self):
		self.CapsNetwork(self.X, name="capsnet")

		self.img_summary = tf.summary.image("input", self.X, max_outputs=5)

		self.trainable_vars = tf.trainable_variables()


	def CapsNetwork(self, input_image, name):
		with tf.variable_scope(name) as scope:
			conv1 = tf.contrib.layers.conv2d(input_image, 256,
											 9, 1, padding="VALID",
											 activation_fn=tf.nn.relu,
											 scope="conv1"
											)			
			conv1 = batch_norm(conv1,name="bn1")
			
			capsule = CapsLayer(self.batch_size)
			
			caps1 = capsule.primaryCaps(conv1, 
										kernel=9,
										stride=2,
										num_outputs=32,
										vec_length=8,
										name="primarycaps")
		


			caps2 = capsule.digitCaps(caps1,
									  num_outputs=10,
									  vec_length=16,
									  iter_routing=3,
									  name="digitcaps_1")

			last_caps = caps2

			with tf.variable_scope("Mask"):
				self.v_length = tf.sqrt(tf.reduce_sum(tf.square(last_caps), axis=2, keep_dims=True) + epsilon)
				self.softmax_v = tf.nn.softmax(self.v_length, dim=1)

				self.argmax_idx = tf.to_int32(tf.argmax(self.softmax_v, axis=1))
				self.argmax_idx = tf.reshape(self.argmax_idx, shape=(self.batch_size, ))

				# if not self.mask_with_y:
				# 	masked_v = []
				# 	for batch_size in range(self.batch_size):
				# 		v = last_caps[batch_size][self.argmax_idx[batch_size], :]
				# 		masked_v.append(tf.reshape(v, shape(1, 1, last_caps.shape[2].value, 1)))

				# 	self.masked_v = tf.concat(masked_v, axis=0)

				# else:
				self.masked_v = tf.multiply(tf.squeeze(last_caps), tf.reshape(self.Y, (-1, last_caps.shape[1].value, 1)))
				self.v_length = tf.sqrt(tf.reduce_sum(tf.square(last_caps), axis=2, keep_dims=True) + epsilon)


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

				#deconv version
				# vector_j = tf.reshape(self.masked_v, shape=(self.batch_size, -1))
				# fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=self.input_width/2*self.input_height/2*3,
				# 										weights_initializer=tf.contrib.layers.xavier_initializer()
				# 										)
				# deconv_ = tf.reshape(fc1, shape=(self.batch_size, self.input_width/2, self.input_height/2, self.input_channel))
				# deconv_ = batch_norm(deconv_,name="deconv_bn")
				# self.decoded = tf.layers.conv2d_transpose(deconv_, 
				# 									filters=3, 
				# 									kernel_size=5,
				# 									strides=(2,2),
				# 									padding='SAME',
				# 									activation=tf.sigmoid)


	def build_loss(self):

		max_l = tf.square(tf.maximum(0., self.m_plus - self.v_length))
		max_r = tf.square(tf.maximum(0., self.v_length - self.m_minus))

		max_l = tf.reshape(max_l, shape=(self.batch_size, -1))
		max_r = tf.reshape(max_r, shape=(self.batch_size, -1))

		T_c = self.Y
		L_c = T_c * max_l + self.lambda_val * (1 - T_c) * max_r

		self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

		origin = tf.reshape(self.X, shape=(self.batch_size, -1))
		squared = tf.square(self.decoded - origin)
		self.reconstruction_err = tf.reduce_mean(squared)

		self.loss = self.margin_loss + self.reg_scale * self.reconstruction_err

		#summary
		self.loss_summary = tf.summary.scalar("loss", tf.reduce_mean(self.loss))
		self.rec_img = tf.reshape(self.decoded, shape=(self.batch_size, self.input_width, self.input_height, self.input_channel))
		self.rec_img_summary = tf.summary.image("rec_img", self.rec_img, max_outputs=5)


		#to check accuracy
		self.pred_label = tf.one_hot(self.argmax_idx, self.output_dim, dtype=tf.float32)
		correct_prediction = tf.equal(self.pred_label, self.Y)
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



	# def train(self, args):

	# 	#optimizer
	# 	self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.momentum, name="AdamOptimizer").minimize(self.loss, var_list=self.trainable_vars)

	# 	epoch = 0
	# 	step = 0

	# 	#saver
	# 	saver = tf.train.Saver()		
	# 	if args.continue_training == "True":
	# 		last_ckpt = tf.train.latest_checkpoint(args.model_path)
	# 		saver.restore(self.sess, last_ckpt)
	# 		ckpt_name = str(last_ckpt)
	# 		print "Loaded model file from " + ckpt_name
	# 		epoch = int(last_ckpt[len(ckpt_name)-1])
	# 	else:
	# 		tf.global_variables_initializer().run()


	# 	coord = tf.train.Coordinator()
	# 	threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)


	# 	all_summary = tf.summary.merge([self.img_summary,
	# 									self.rec_img_summary,
	# 									self.loss_summary])

		
	# 	writer = tf.summary.FileWriter(self.graph_path, self.sess.graph)

	# 	while epoch < self.epochs:
	# 		summary, loss, acc, _ = self.sess.run([all_summary, self.loss, self.accuracy, self.optimizer])
	# 		writer.add_summary(summary, step)

	# 		print "Epoch [%d] step [%d] Training Loss: [%.2f] Accuracy: [%.2f]" % (epoch, step, loss, acc)

	# 		step += 1

	# 		if step*self.batch_size >= self.data_count:
	# 			saver.save(self.sess, args.checkpoint_path + "model", global_step=epoch)
	# 			print "Model saved at epoch %s" % str(epoch)				
	# 			epoch += 1
	# 			step = 0

	# 	coord.request_stop()
	# 	coord.join(threads)
	# 	self.sess.close()			
	# 	print("Done.")


		

	# def test(self, args):
	# 	# tf.global_variables_initializer().run()

	# 	saver = tf.train.Saver()		

	# 	last_ckpt = tf.train.latest_checkpoint(args.checkpoint_path)
	# 	saver.restore(self.sess, last_ckpt)
	# 	print "Loaded model file from " + str(last_ckpt)

	# 	coord = tf.train.Coordinator()
	# 	threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

	# 	ave_loss = 0 
	# 	epoch = 0
	# 	step = 0
	# 	while epoch < 1:
	# 		loss = self.sess.run(self.loss)
	# 		accuracy = self.sess.run(self.accuracy)
			
	# 		print "Epoch [%d] step [%d] Test Loss: [%.2f] Accuracy [%.2f]" % (epoch, step, loss, accuracy)
			
	# 		ave_loss += loss
	# 		step += 1
	# 		if step*self.batch_size > self.data_count:
	# 			epoch += 1

	# 	print "Accuracy: [%.2f]"%(1 - ave_loss/step)
	# 	coord.request_stop()
	# 	coord.join(threads)
	# 	self.sess.close()
	# 	print('Done')