
import tensorflow as tf

class Manager():
	def __init__(self, args):
		self.learning_rate = args.learning_rate
		self.momentum = args.momentum

		self.continue_training = args.continue_training
		self.checkpoints_path = args.checkpoints_path
		self.graph_path = args.graph_path
		self.epochs = args.epochs

		self.batch_size = args.batch_size


	def train(self, sess, model):

		#optimizer
		optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.momentum, name="AdamOptimizer").minimize(model.loss, var_list=model.trainable_vars)

		epoch = 0
		step = 0

		#saver
		saver = tf.train.Saver()		
		if self.continue_training == "True":
			last_ckpt = tf.train.latest_checkpoint(self.model_path)
			saver.restore(sess, last_ckpt)
			ckpt_name = str(last_ckpt)
			print "Loaded model file from " + ckpt_name
			epoch = int(last_ckpt[len(ckpt_name)-1])
		else:
			tf.global_variables_initializer().run()


		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)


		if hasattr(model, 'rec_img_summary'):
			all_summary = tf.summary.merge([model.img_summary,
											model.rec_img_summary,
											model.loss_summary,
											model.acc_summary])
		else:
			all_summary = tf.summary.merge([model.img_summary,
											model.loss_summary,
											model.acc_summary])
		
		writer = tf.summary.FileWriter(self.graph_path, sess.graph)

		while epoch < self.epochs:
			summary, loss, acc, _ = sess.run([all_summary, 
											  model.loss, 
											  model.accuracy, 
											  optimizer])
			writer.add_summary(summary, step)

			print "Epoch [%d] step [%d] Training Loss: [%.2f] Accuracy: [%.2f]" % (epoch, step, loss, acc)

			step += 1

			if step*self.batch_size >= model.data_count:
				saver.save(sess, args.checkpoint_path + "model", global_step=epoch)
				print "Model saved at epoch %s" % str(epoch)				
				epoch += 1
				step = 0

		coord.request_stop()
		coord.join(threads)
		sess.close()			
		print("Done.")


	def test(self, sess, model):
		# tf.global_variables_initializer().run()

		saver = tf.train.Saver()		

		last_ckpt = tf.train.latest_checkpoint(self.checkpoints_path)
		saver.restore(sess, last_ckpt)
		print "Loaded model file from " + str(last_ckpt)

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)

		ave_loss = 0 
		epoch = 0
		step = 0
		while epoch < 1:
			loss = sess.run(model.loss)
			accuracy = sess.run(model.accuracy)
			
			print "Epoch [%d] step [%d] Test Loss: [%.2f] Accuracy [%.2f]" % (epoch, step, loss, accuracy)
			
			ave_loss += loss
			step += 1
			if step*self.batch_size > model.data_count:
				epoch += 1

		print "Ave. Accuracy: [%.2f]"%(1 - ave_loss/step)
		coord.request_stop()
		coord.join(threads)
		sess.close()
		print('Done')