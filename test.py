import gym
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
env = gym.make("Breakout-v0")
env.reset()

# #defining CNN
# input = tf.placeholder(tf.float32, [None, 80, 80])
# conv1 = tf.nn.conv2d(input=input, filter=16, kernel_size=[8, 8], activation=tf.nn.relu)
# conv2 = tf.nn.conv2d(input=conv1, filter=32, kernel_size=[4, 4], activation=tf.nn.relu)
# dense = tf.layers.dense(input=conv2, activation=tf.nn.relu, units=125)
#
# #defining Pollicy Net
# policy = tf.layer.dense(unit=1, activation=tf.nn.sigmoid) #output the probaility of going up


def imgporcessing(img):
    img = img[25:, :, :]

def greyscale(img):
    grey_img = np.ndarray[64,64]
    for i in range(3):
        grey_img +=[:,:,i]



#playing a hundred episode
for _ in range(10):
    done = False

    while not done:
        obs, reward, done, _ = env.step(env.action_space.sample())
        plt.imshow(obs)
        plt.show()
