import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def sampling(args):
    z_mean, z_log_var = args
    epislion = tf.random_normal(tf.shape(z_mean)[0], 32, mean=0.0, stddev=1.0)
    return z_mean + tf.exp(z_log_var/2)*epislion
class Encoder():

    def __init__(self):
        pass

    def Model(self):
        #encoder
        with tf.variable_scope('enocder'):
            input_image =  tf.placeholder(shape=[None,64,64,3],dtype=tf.float32)
            encoder_conv1 = slim.layers.conv2d(inputs=input_image, num_outputs=32,kernel_size=[4,4], stride=2,activation_fn=tf.nn.relu)
            encoder_conv2 = slim.layer.conv2d(inputs=encoder_conv1, num_outputs=64, kernel=[4,4], stride=2, activation_fn=tf.nn.relu)
            encoder_conv3 = slim.layers.conv2d(inputs=encoder_conv2, num_outputs=64, kernel_size=[4, 4], stride=2,activation_fn=tf.nn.relu)
            encoder_conv4 = slim.layers.conv2d(inputs=encoder_conv3, num_outputs=128, kernel_size=[4, 4], stride=2,activation_fn=tf.nn.relu)

            encoder_out_mean = slim.fully_connected(num_outputs=32, inputs=slim.flatten(encoder_conv4))
            encoder_out_log_var = slim.fully_connected(num_outputs=32, inputs=slim.flatten(encoder_conv4))
            encoder_out = sampling([encoder_out_mean,encoder_out_log_var])

        #decoder
        with tf.variable_scope('decoder'):
            decoder_input = slim.fully_connected(num_outputs=1024,inputs=encoder_out,)
            decoder_input = tf.reshape(decoder_input,[1,1,1024])
            decoder_conv1 =slim.conv2d_transpose(num_outputs=64, kernel_size=[5,5], stride=[2,2], activation_fn=tf.nn.relu, inputs=decoder_input)
            decoder_conv2 = slim.conv2d_transpose(num_outputs=64, kernel_size=[5, 5], stride=[2, 2], activation_fn=tf.nn.relu, inputs=decoder_conv1)
            decoder_conv3 = slim.conv2d_transpose(num_outputs=32, kernel_size=[6, 6], stride=[2, 2], activation_fn=tf.nn.relu, inputs=decoder_conv2)
            decoder_conv4 = slim.conv2d_transpose(num_outputs=3, kernel_size=[6, 6], stride=[2, 2],activation_fn=tf.nn.sigmoid, inputs=decoder_conv3)



