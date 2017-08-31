import tensorflow as tf
import numpy as np 
from numpy.random import RandomState
from tensorflow.examples.tutorials.mnist import input_data

LENET_BATCH_SIZE = 32
LENET_PATCH_SIZE = 5
LENET_PATCH_DEPTH1 = 6
LENET_PATCH_DEPTH2 = 16
LENET_HID_OUT1 = 120
LENET_HID_OUT2 = 84

def flatten_tf_array(array):
    shape = array.get_shape().as_list()
    return tf.reshape(array, [shape[0], shape[1] * shape[2] * shape[3]])

def variables_lenet5():
    w1 = tf.Variable(tf.truncated_normal([LENET_PATCH_SIZE, LENET_PATCH_SIZE, 1, LENET_PATCH_DEPTH1], stddev=0.1))
    b1 = tf.Variable(tf.zeros([LENET_PATCH_DEPTH1]))
    w2 = tf.Variable(tf.truncated_normal([LENET_PATCH_SIZE, LENET_PATCH_SIZE, LENET_PATCH_DEPTH1, LENET_PATCH_DEPTH2], stddev=0.1))
    b2 = tf.Variable(tf.constant(0.1, shape=[LENET_PATCH_DEPTH2]))
    w3 = tf.Variable(tf.truncated_normal([LENET_PATCH_SIZE, LENET_PATCH_SIZE, LENET_PATCH_DEPTH2, LENET_HID_OUT1], stddev=0.1))
    b3 = tf.Variable(tf.constant(0.1, shape=[LENET_HID_OUT1]))
    w4 = tf.Variable(tf.truncated_normal([LENET_HID_OUT1, LENET_HID_OUT2], stddev=0.1))
    b4 = tf.Variable(tf.constant(0.1, shape=[LENET_HID_OUT2]))
    w5 = tf.Variable(tf.truncated_normal([LENET_HID_OUT2, 10], stddev=0.1))
    b5 = tf.Variable(tf.constant(0.1, shape=[10]))
    variables_dict = {
        'w1':w1, 'w2':w2, 'w3':w3, 'w4':w4, 'w5':w5,
        'b1':b1, 'b2':b2, 'b3':b3, 'b4':b4, 'b5':b5
    }
    return variables_dict

def model_lenet5(input_x, variables_dict):
    layer1_conv = tf.nn.conv2d(input_x, variables_dict['w1'], [1,1,1,1], 'SAME')
    layer1_actv = tf.nn.relu(layer1_conv + variables_dict['b1'])
    layer1_pool = tf.nn.avg_pool(layer1_actv, [1, 2, 2, 1], [1,2,2,1], 'SAME')
    layer2_conv = tf.nn.conv2d(layer1_pool, variables_dict['w2'], [1,1,1,1], 'VALID')
    layer2_actv = tf.nn.relu(layer2_conv + variables_dict['b2'])
    layer2_pool = tf.nn.avg_pool(layer2_actv, [1, 2, 2, 1], [1,2,2,1], 'SAME')
    layer3_flat = flatten_tf_array(layer2_pool)
    layer3_out = tf.matmul(layer3_flat, variables_dict['w3']) + variables_dict['b3']
    layer3_actv = tf.nn