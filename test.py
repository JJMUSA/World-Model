import gym
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from scipy import misc


env = gym.make("Breakout-v0")
env.reset()

#defining CNN
input = tf.placeholder(tf.float32, [None, 80, 80])
conv1 = tf.nn.conv2d(input=input, filter=16, kernel_size=[8, 8], activation=tf.nn.relu)
conv2 = tf.nn.conv2d(input=conv1, filter=32, kernel_size=[4, 4], activation=tf.nn.relu)
dense = tf.layers.dense(input=conv2, activation=tf.nn.relu, units=125)

#defining Pollicy Net
policy = tf.layer.dense(input=dense, unit=1, activation=tf.nn.sigmoid) #output the probaility of moving right

#advantage


#defining policy_loss
#policy_loss =


def imgporcessing(img):
    img = greyscale(img)
    img = img[25:, :] #crop out irrelevant pixels
    img = misc.imresize(img, [80, 80]) #resize to 80X80

    return img



def greyscale(img):
    """takes a colored image and return a black and white img"""
    grey_img = np.zeros([210, 160])
    for i in range(3):
        grey_img =np.sum([grey_img,img[:,:,i]], 0)
    grey_img /= 3
    grey_img = grey_img.astype(np.uint8)
    return grey_img



#playing a hundred episode
for i in range(10):
    obs, reward, done, _ = env.step(env.action_space.sample())
    obs = imgporcessing(obs)

