import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
from scipy import misc
sess  = tf.Session()

summary_file = tf.summary.FileWriter('./encoder_training')
def sampling(args):
    z_mean, z_log_var = args
    epislion = tf.random_normal(tf.shape(z_mean)[0], 32, mean=0.0, stddev=1.0)
    return z_mean + tf.exp(z_log_var/2)*epislion

class Encoder():

    def __init__(self):
        #encoder
        with tf.variable_scope('enocder'):
            self.input_image = tf.placeholder(shape=[None,32,32,3],dtype=tf.float32)
            encoder_conv1 = slim.conv2d(inputs=self.input_image, num_outputs=32,kernel_size=[4,4], stride=2,activation_fn=tf.nn.relu)
            encoder_conv2 = slim.conv2d(inputs=encoder_conv1, num_outputs=64, kernel_size=[4,4], stride=2, activation_fn=tf.nn.relu)
            encoder_conv3 = slim.conv2d(inputs=encoder_conv2, num_outputs=64, kernel_size=[4, 4], stride=2,activation_fn=tf.nn.relu)
            encoder_conv4 = slim.conv2d(inputs=encoder_conv3, num_outputs=128, kernel_size=[4, 4], stride=2,activation_fn=tf.nn.tanh)

            encoder_out_mean = slim.fully_connected(num_outputs=32, inputs=slim.flatten(encoder_conv4))
            encoder_out_log_var = slim.fully_connected(num_outputs=32, inputs=slim.flatten(encoder_conv4))

            #stocahastic
            episilon = tf.random_normal(shape=(tf.shape(encoder_out_mean)[0], 32), mean=0, stddev=1.0)
            self.encoder_out = tf.add(encoder_out_mean,(episilon * tf.exp(encoder_out_log_var/2)))

        #decoder
        with tf.variable_scope('decoder'):
            decoder_input = slim.fully_connected(num_outputs=1024,inputs=self.encoder_out,)
            decoder_input = tf.reshape(decoder_input,[-1,1,1,1024])
            decoder_conv1 =slim.conv2d_transpose(num_outputs=64, kernel_size=[5,5], stride=2, activation_fn=tf.nn.relu, inputs=decoder_input)
            decoder_conv2 = slim.conv2d_transpose(num_outputs=64, kernel_size=[5, 5], stride=2, activation_fn=tf.nn.relu, inputs=decoder_conv1)
            decoder_conv3 = slim.conv2d_transpose(num_outputs=32, kernel_size=[6, 6], stride= 2, activation_fn=tf.nn.relu, inputs=decoder_conv2)
            self.decoder_conv4 = slim.conv2d_transpose(num_outputs=12, kernel_size=[2, 2], stride=2,activation_fn=tf.nn.sigmoid, inputs=decoder_conv3)
            #self.decoder_conv4 = tf.reshape(self.decoder_conv4, shape=[-1,32,32,3])

        #loss
        x = slim.flatten(self.input_image)
        y = slim.flatten(self.decoder_conv4)
        r_loss = 10*tf.reduce_mean(tf.square(x-y))

        kl_loss = tf.reduce_mean(1+ encoder_out_log_var - tf.square(encoder_out_mean) - tf.exp(encoder_out_log_var))

        self.loss = tf.add(r_loss, kl_loss)

        #optimiser
        optimiser = tf.train.AdamOptimizer()
        var =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.op = optimiser.minimize(loss=self.loss, var_list=var)
        sess.run(tf.global_variables_initializer())

    def train(self, images):
        Batch_size = 5
        EPOCHS =  100
        train_size =  len(images)

        for i in range(EPOCHS):
            for start, end in zip(range(0,train_size,Batch_size), range(Batch_size,train_size+1, Batch_size)):
                ops = sess.run([self.loss, self.op], feed_dict = {self.input_image: images[start:end]})
                summary  = tf.Summary()
                summary.value.add(tag='Loss/loss', simple_value = ops[0] )
                summary_file.add_summary(summary, i)

    def encode(self, image):
        encoded_data = sess.run(self.encoder_out, feed_dict={self.input_image:image})
        return encoded_data

    def decode(self, encoded_data):
        decoded_image = sess.run(self.decoder_conv4, feed_dict={self.encoder_out: encoded_data})
        return decoded_image


encoder = Encoder()
img_names = os.listdir('pong-images')
images = []
for i in img_names[0:100]:
    img = misc.imread('pong-images/'+i)
    img = misc.imresize(img,[32,32,3])
    images.append(img)
print(images[0].shape)
encoder.train(images)
encoder_data = encoder.encode([images[0]])
print(encoder_data)