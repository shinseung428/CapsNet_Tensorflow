import tensorflow as tf
import os
import numpy as np
import scipy.misc
import argparse
import sys

sys.path.insert(0, './models')
from capsule_dynamic import capsule_dynamic
from normal_classifier import normal_classifier

from manager import Manager
# from capsule_em import capsule_em
#============================================================================================

parser = argparse.ArgumentParser(description='')
#Training Settings
parser.add_argument('--data', dest='data', default='mnist', help='cats image train path')
parser.add_argument('--root_path', dest='root_path', default='./data/', help='cats image train path')

parser.add_argument('--input_width', dest='input_width', default=28, help='input image width')
parser.add_argument('--input_height', dest='input_height', default=28, help='input image height')
parser.add_argument('--input_channel', dest='input_channel', default=1, help='input image channel')
parser.add_argument('--output_dim', dest='output_dim', default=10, help='output dim')

#Training setting 
parser.add_argument('--epochs', dest='epochs', default=10, help='total number of epochs')
parser.add_argument('--is_train', dest='is_train', default=True, help='flag to train')
parser.add_argument('--continue_training', dest='continue_training', default=False, help='flag to continue training')
parser.add_argument('--batch_size', dest='batch_size', default=64, help='batch size')

parser.add_argument('--rotate', dest='rotate', default=False, help='rotate image flag')

parser.add_argument('--learning_rate', dest='learning_rate', default=0.01, help='learning rate of the optimizer')
parser.add_argument('--momentum', dest='momentum', default=0.5, help='momentum of the optimizer')

parser.add_argument('--checkpoints_path', dest='checkpoints_path', default='./checkpoints/', help='saved model checkpoint path')
parser.add_argument('--graph_path', dest='graph_path', default='./graphs/', help='graph')
parser.add_argument('--model_num', dest='model_num', default=None, help='model number to restore')


parser.add_argument('--mask_with_y', dest='mask_with_y', default=True, help='mask_with_y')
parser.add_argument('--m_plus', dest='m_plus', default=0.9, help='m_plus')
parser.add_argument('--m_minus', dest='m_minus', default=0.1, help='m_minus')
parser.add_argument('--lambda_val', dest='lambda_val', default=0.5, help='lambda_val')
parser.add_argument('--reg_scale', dest='reg_scale', default=0.0005, help='reg_scale')

args = parser.parse_args()
#============================================================================================

def main(_):
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    print 'Starting CapsNet Classifier...'
    with tf.Session(config=run_config) as sess:
        model = capsule_dynamic(args)
        manager = Manager(args)

        if args.is_train == True:
            print 'Start Training...'
            manager.train(sess, model)
        else:
            print 'Start Testing...'
            manager.test(sess, model)

main(args)

#Still Working....