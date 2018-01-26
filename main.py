import tensorflow as tf
import os
import numpy as np
import scipy.misc
import argparse
import sys

sys.path.insert(0, './models')
from baseline_network import baseline_network
from capsule_dynamic import capsule_dynamic
from manager import Manager
from capsule_em import capsule_em
#============================================================================================

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', True):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', False):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='')

parser.add_argument('--model', dest='model', default="capsule_dynamic", help='model type')

#Image/Output setting
parser.add_argument('--input_width', dest='input_width', default=28, help='input image width')
parser.add_argument('--input_height', dest='input_height', default=28, help='input image height')
parser.add_argument('--input_channel', dest='input_channel', default=1, help='input image channel')
parser.add_argument('--output_dim', dest='output_dim', default=10, help='output dim')

#Training Settings
parser.add_argument('--data', dest='data', default='mnist', help='cats image train path')
parser.add_argument('--root_path', dest='root_path', default='./data/', help='cats image train path')

parser.add_argument('--epochs', dest='epochs', default=500, help='total number of epochs')
parser.add_argument('--batch_size', dest='batch_size', default=64, help='batch size')

parser.add_argument('--learning_rate', dest='learning_rate', default=1e-5, help='learning rate of the optimizer')
parser.add_argument('--momentum', dest='momentum', default=0.5, help='momentum of the optimizer')

parser.add_argument('--m_plus', dest='m_plus', default=0.9, help='m_plus')
parser.add_argument('--m_minus', dest='m_minus', default=0.1, help='m_minus')
parser.add_argument('--lambda_val', dest='lambda_val', default=0.5, help='lambda_val')
parser.add_argument('--reg_scale', dest='reg_scale', default=0.0005, help='reg_scale')

#Test Setting
parser.add_argument('--is_train', dest='is_train', default=True, type=str2bool, help='flag to train')
parser.add_argument('--continue_training', dest='continue_training', default=False, type=str2bool, help='flag to continue training')
parser.add_argument('--rotate', dest='rotate', default=False, type=str2bool, help='rotate image flag')
parser.add_argument('--random_pos', dest='random_pos', default=False, type=str2bool, help='randomly place image on 40 x 40 background')

#Extra folders setting
parser.add_argument('--checkpoints_path', dest='checkpoints_path', default='./checkpoints/', help='saved model checkpoint path')
parser.add_argument('--graph_path', dest='graph_path', default='./graphs/', help='tensorboard graph')

args = parser.parse_args()

#============================================================================================

def main(_):
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    
    with tf.Session(config=run_config) as sess:
        #print used dataset
        print "Dataset: %s"%args.data
        print "Model: %s"%args.model

        if args.model == "baseline_network":
            model = baseline_network(args)
        elif args.model == "capsule_dynamic":
            model = capsule_dynamic(args)
        elif args.model == "capsule_em":
            model = capsule_em(args)        


        #create graph and checkpoints folder if they don't exist
        if not os.path.exists(args.checkpoints_path):
            os.makedirs(args.checkpoints_path)
        if not os.path.exists(args.graph_path):
            os.makedirs(args.graph_path)
            
        #create a subfolder in checkpoints folder
        args.checkpoints_path = os.path.join(args.checkpoints_path, args.model + "/")
        if not os.path.exists(args.checkpoints_path):
            os.makedirs(args.checkpoints_path)
        args.graph_path = os.path.join(args.graph_path, args.model + "/")
        if not os.path.exists(args.graph_path):
            os.makedirs(args.graph_path)

        #manager performs all the training/testing
        manager = Manager(args)

        if args.is_train:
            print 'Start Training...'
            manager.train(sess, model)
        else:
            print 'Start Testing...'
            manager.test(sess, model)

main(args)

#Still Working....