import numpy as np
import tensorflow as tf
from glob import glob
import os
import math

def one_hot(label, output_dim):
	one_hot = np.zeros((len(label), output_dim))
	
	for idx in range(0,len(label)):
		one_hot[idx, label[idx]] = 1
	
	return one_hot


def affnist_reader(args, path):

	f = open(os.path.join(path, 'train-images.idx3-ubyte'))
	loaded = np.fromfile(file=f, dtype=np.uint8)
	trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

	f = open(os.path.join(path, 'train-labels.idx1-ubyte'))
	loaded = np.fromfile(file=f, dtype=np.uint8)
	trainY = loaded[8:].reshape((60000)).astype(np.int32)
	trainY = one_hot(trainY, args.output_dim)

	f = open(os.path.join(path, 't10k-images.idx3-ubyte'))
	loaded = np.fromfile(file=f, dtype=np.uint8)
	testX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float32)

	f = open(os.path.join(path, 't10k-labels.idx1-ubyte'))
	loaded = np.fromfile(file=f, dtype=np.uint8)
	testY = loaded[8:].reshape((10000)).astype(np.int32)
	testY = one_hot(testY, args.output_dim)

	if args.is_train:
		X = tf.convert_to_tensor(trainX, dtype=tf.float32) / 255.
		Y = tf.convert_to_tensor(trainY, dtype=tf.float32)
		data_count = len(trainX)
	else:
		X = tf.convert_to_tensor(testX, dtype=tf.float32) / 255.
		Y = tf.convert_to_tensor(testY, dtype=tf.float32)
		data_count = len(testX)

	input_queue = tf.train.slice_input_producer([X, Y],shuffle=True)
	images = tf.image.resize_images(input_queue[0] ,[args.input_width, args.input_height])
	labels = input_queue[1]

	if args.rotate:
		angle = tf.random_uniform([1], minval=-60, maxval=60, dtype=tf.float32)
		radian = angle * math.pi / 180
		images = tf.contrib.image.rotate(images, radian)

	X, Y = tf.train.batch([images, labels],
						  batch_size=args.batch_size
						  )

	return X, Y, data_count


def fashion_mnist_reader(args, path):
	f = open(os.path.join(path, 'train-images-idx3-ubyte'))
	loaded = np.fromfile(file=f, dtype=np.uint8)
	trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

	f = open(os.path.join(path, 'train-labels-idx1-ubyte'))
	loaded = np.fromfile(file=f, dtype=np.uint8)
	trainY = loaded[8:].reshape((60000)).astype(np.int32)
	trainY = one_hot(trainY, args.output_dim)


	f = open(os.path.join(path, 't10k-images-idx3-ubyte'))
	loaded = np.fromfile(file=f, dtype=np.uint8)
	testX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float32)

	f = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
	loaded = np.fromfile(file=f, dtype=np.uint8)
	testY = loaded[8:].reshape((10000)).astype(np.int32)
	testY = one_hot(testY, args.output_dim)

	if args.is_train:
		X = tf.convert_to_tensor(trainX, dtype=tf.float32) / 255.
		Y = tf.convert_to_tensor(trainY, dtype=tf.float32)
		data_count = len(trainX)
	else:
		X = tf.convert_to_tensor(testX, dtype=tf.float32) / 255.
		Y = tf.convert_to_tensor(testY, dtype=tf.float32)
		data_count = len(testX)		

	input_queue = tf.train.slice_input_producer([X, Y],shuffle=True)
	images = tf.image.resize_images(input_queue[0] ,[args.input_width, args.input_height])
	labels = input_queue[1]

	if args.rotate:
		angle = tf.random_uniform([1], minval=-60, maxval=60, dtype=tf.float32)
		radian = angle * math.pi / 180
		images = tf.contrib.image.rotate(images, radian)

	X, Y = tf.train.batch([images, labels],
						  batch_size=args.batch_size
						  )
	return X, Y, data_count


def mnist_reader(args, path):

	f = open(os.path.join(path, 'train-images.idx3-ubyte'))
	loaded = np.fromfile(file=f, dtype=np.uint8)
	trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

	f = open(os.path.join(path, 'train-labels.idx1-ubyte'))
	loaded = np.fromfile(file=f, dtype=np.uint8)
	trainY = loaded[8:].reshape((60000)).astype(np.int32)
	trainY = one_hot(trainY, args.output_dim)

	f = open(os.path.join(path, 't10k-images.idx3-ubyte'))
	loaded = np.fromfile(file=f, dtype=np.uint8)
	testX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float32)

	f = open(os.path.join(path, 't10k-labels.idx1-ubyte'))
	loaded = np.fromfile(file=f, dtype=np.uint8)
	testY = loaded[8:].reshape((10000)).astype(np.int32)
	testY = one_hot(testY, args.output_dim)

	if args.is_train:
		X = tf.convert_to_tensor(trainX, dtype=tf.float32) / 255.
		Y = tf.convert_to_tensor(trainY, dtype=tf.float32)
		data_count = len(trainX)
	else:
		X = tf.convert_to_tensor(testX, dtype=tf.float32) / 255.
		Y = tf.convert_to_tensor(testY, dtype=tf.float32)
		data_count = len(testX)

	input_queue = tf.train.slice_input_producer([X, Y],shuffle=True)
	images = tf.image.resize_images(input_queue[0] ,[args.input_width, args.input_height])
	labels = input_queue[1]

	if args.rotate:
		angle = tf.random_uniform([1], minval=-60, maxval=60, dtype=tf.float32)
		radian = angle * math.pi / 180
		images = tf.contrib.image.rotate(images, radian)


	X, Y = tf.train.batch([images, labels],
						  batch_size=args.batch_size
						  )

	return X, Y, data_count


# currently not working
# def catsdogs_reader(args, path):
# 	def create_label_data(path, label=0):
# 		file_list = glob(path)
# 		data_count = len(file_list)

# 		one_hot = np.zeros((data_count, args.output_dim))
# 		one_hot[:,label] = 1

# 		return file_list, one_hot, data_count

# 	cats_file_paths = os.path.join(path,"*cat*")
# 	dogs_file_paths = os.path.join(path,"*dog*")

# 	cats_filename, cats_labels, cats_count = create_label_data(cats_file_paths, label=0)
# 	dogs_filename, dogs_labels, dogs_count = create_label_data(dogs_file_paths, label=1)

# 	data_count = cats_count + dogs_count

# 	all_filepaths = cats_filename + dogs_filename
# 	all_labels = np.concatenate((cats_labels, dogs_labels), 0)

# 	all_images = tf.convert_to_tensor(all_filepaths, dtype=tf.string)
# 	all_labels = tf.convert_to_tensor(all_labels, dtype=tf.float32)

# 	train_input_queue = tf.train.slice_input_producer([all_images, all_labels], shuffle=True)

# 	file_content = tf.read_file(train_input_queue[0])
# 	train_images = tf.image.decode_jpeg(file_content, channels=args.input_channel)
# 	train_labels = train_input_queue[1]

# 	#set image shape and normalize image input
# 	train_images = tf.image.resize_images(train_images,[args.input_width, args.input_height])
# 	train_images.set_shape((args.input_width, args.input_height, args.input_channel))
# 	train_images = tf.cast(train_images, tf.float32) / 255.0

# 	X, Y = tf.train.batch([train_images, train_labels],
# 						  batch_size=args.batch_size
# 						  )

# 	return X, Y, data_count



#load different datasets
def load_data(args):

	path = os.path.join(args.root_path, args.data)

	if args.data == "mnist":
		images, labels, data_count = mnist_reader(args, path)
	elif args.data == "fashion_mnist":
		images, labels, data_count = fashion_mnist_reader(args, path)
	if args.data == "affnist":
		images, labels, data_count = affnist_reader(args, path)					
	elif args.data == "catsdogs":#currently not working well using capsule_dynamic
		images, labels, data_count = catsdogs_reader(args, path)

	return images, labels, data_count 