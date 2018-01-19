import numpy as np
import tensorflow as tf
import scipy.io as sio
from glob import glob
import os
import math
import cv2
import cPickle

from norb_reader import *

def place_random(trainX):
	#randomly place 28x28 mnist image on 40x40 background
	trainX_new = []
	for img in trainX:
		x_len = maxX - minX
		y_len = maxY - minY

		img_new = np.zeros((40,40,1), dtype=np.float32)
		x = np.random.randint(12 , size=1)[0]
		y = np.random.randint(12 , size=1)[0]

		img_new[y:y+28, x:x+28, :] = img
		trainX_new.append(img_new)
	
	return np.array(trainX_new)

def one_hot(label, output_dim):
	one_hot = np.zeros((len(label), output_dim))
	
	for idx in range(0,len(label)):
		one_hot[idx, label[idx]] = 1
	
	return one_hot

def load_data_from_mat(path):
	data = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
	for key in data:
		if isinstance(data[key], sio.matlab.mio5_params.mat_struct):
			data[key] = _todict(data[key])
	return data

def _todict(matobj):
    #A recursive function which constructs from matobjects nested dictionaries
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

#============== Different Readers ==============

def affnist_reader(args, path):
	train_path = glob(os.path.join(path, "train/*.mat"))
	test_path = glob(os.path.join(path, "test/*.mat"))

	train_data = load_data_from_mat(train_path[0])

	trainX = train_data['affNISTdata']['image'].transpose()
	trainY = train_data['affNISTdata']['label_int']

	trainX = trainX.reshape((50000, 40, 40, 1)).astype(np.float32)
	trainY = trainY.reshape((50000)).astype(np.int32)
	trainY = one_hot(trainY, args.output_dim)

	test_data = load_data_from_mat(test_path[0])
	testX = test_data['affNISTdata']['image'].transpose()
	testY = test_data['affNISTdata']['label_int']

	testX = testX.reshape((10000, 40, 40, 1)).astype(np.float32)
	testY = testY.reshape((10000)).astype(np.int32)
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

def cifar_reader(args, path):
	def unpickle(file):
		with open(file, 'rb') as fo:
			dict = cPickle.load(fo)
		return dict

	train_path = glob(os.path.join(path, "data_batch_*"))	
	test_path = glob(os.path.join(path, "test_batch"))

	trainX = []
	trainY = []
	for p in train_path:
		extracted = unpickle(p)
		image = None
		for t in extracted['data']:
			r = t[:1024].reshape(32,32,1)
			g = t[1024:2048].reshape(32,32,1)
			b = t[2048:].reshape(32,32,1)
			image = np.concatenate([r,g,b], axis=2)
			trainX.append(image)
		trainY.append(extracted['labels'])

	trainX = np.array(trainX)	
	trainY = np.concatenate(np.array(trainY), axis=0)	
	trainY = one_hot(trainY, args.output_dim)

	testX = []
	testY = []
	extracted = unpickle(test_path[0])
	for t in extracted['data']:
		r = t[:1024].reshape(32,32,1)
		g = t[1024:2048].reshape(32,32,1)
		b = t[2048:].reshape(32,32,1)
		image = np.concatenate([r,g,b], axis=2)
		testX.append(image)
	testY.append(extracted['labels'])
	testX = np.array(testX)	
	testY = np.concatenate(np.array(testY), axis=0)		
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
		angle = tf.random_uniform([1], minval=-30, maxval=30, dtype=tf.float32)
		radian = angle * math.pi / 180
		images = tf.contrib.image.rotate(images, radian)

	X, Y = tf.train.batch([images, labels],
						  batch_size=args.batch_size
						  )

	return X, Y, data_count


def small_norb_reader(args, path):
	def extract_patch(dataset):
		extracted = []
		for img in train_dat:
			img_ = img[0].reshape(96,96,1)
			extracted.append(img_)
			img_ = img[1].reshape(96,96,1)
			extracted.append(img_)					
		return np.array(extracted)
	
	#Training Data
	file_handle = open(getPath('train','dat'))
	train_dat = parseNORBFile(file_handle)
	file_handle = open(getPath('train','cat'))
	train_cat = parseNORBFile(file_handle)

	#Test Data
	file_handle = open(getPath('test','dat'))
	test_dat = parseNORBFile(file_handle)
	file_handle = open(getPath('test','cat'))
	test_cat = parseNORBFile(file_handle)

	trainX = extract_patch(train_dat)
	trainY = np.repeat(train_cat, 2)
	trainY = one_hot(trainY, args.output_dim)

	testX = extract_patch(test_dat)
	testY = np.repeat(test_cat, 2)
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
		
	if args.is_train:	
		images = tf.image.resize_images(input_queue[0] ,[48, 48])
		images = tf.random_crop(images, [32, 32, 1])
	else:
		images = tf.image.resize_images(input_queue[0] ,[48, 48])
		images = tf.image.resize_image_with_crop_or_pad(images, 32, 32)

	labels = input_queue[1]

	if args.rotate:
		angle = tf.random_uniform([1], minval=-30, maxval=30, dtype=tf.float32)
		radian = angle * math.pi / 180
		images = tf.contrib.image.rotate(images, radian)

	X, Y = tf.train.batch([images, labels],
						  batch_size=args.batch_size
						  )

	return X, Y, data_count


def fashion_mnist_reader(args, path):
	#Training Data
	f = open(os.path.join(path, 'train-images-idx3-ubyte'))
	loaded = np.fromfile(file=f, dtype=np.uint8)
	trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

	f = open(os.path.join(path, 'train-labels-idx1-ubyte'))
	loaded = np.fromfile(file=f, dtype=np.uint8)
	trainY = loaded[8:].reshape((60000)).astype(np.int32)
	trainY = one_hot(trainY, args.output_dim)


	#Test Data
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
		angle = tf.random_uniform([1], minval=-30, maxval=30, dtype=tf.float32)
		radian = angle * math.pi / 180
		images = tf.contrib.image.rotate(images, radian)

	X, Y = tf.train.batch([images, labels],
						  batch_size=args.batch_size
						  )
	return X, Y, data_count


def mnist_reader(args, path):
	#Training Data
	f = open(os.path.join(path, 'train-images.idx3-ubyte'))
	loaded = np.fromfile(file=f, dtype=np.uint8)
	trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

	if args.random_pos:
		trainX = place_random(trainX)

	f = open(os.path.join(path, 'train-labels.idx1-ubyte'))
	loaded = np.fromfile(file=f, dtype=np.uint8)
	trainY = loaded[8:].reshape((60000)).astype(np.int32)
	trainY = one_hot(trainY, args.output_dim)


	#Test Data
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

	input_queue = tf.train.slice_input_producer([X, Y], shuffle=True)
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




#load different datasets
def load_data(args):

	path = os.path.join(args.root_path, args.data)

	if args.data == "mnist":
		images, labels, data_count = mnist_reader(args, path)
	elif args.data == "fashion_mnist":
		images, labels, data_count = fashion_mnist_reader(args, path)
	elif args.data == "affnist":
		images, labels, data_count = affnist_reader(args, path)					
	elif args.data == "small_norb":
		images, labels, data_count = small_norb_reader(args, path)
	elif args.data == "cifar10":
		images, labels, data_count = cifar_reader(args, path)		
	else:
		print "Invalid dataset name!!"

	return images, labels, data_count 