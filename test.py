import gym,time
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from scipy import misc
sess = tf.Session()
gamma = 0.99


#defining CNN
input = tf.placeholder(dtype=tf.float32, shape=[None, 80, 80, 1])

conv1 = tf.layers.conv2d(inputs=input, filters=16, kernel_size=[8, 8], padding='VALID')
conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[4, 4], padding='VALID')
dense = tf.contrib.layers.fully_connected(inputs=tf.contrib.slim.flatten(conv2), activation_fn=tf.nn.relu, num_outputs=125)

#defining Pollicy Net
policy = tf.contrib.layers.fully_connected(inputs=dense, num_outputs=1, activation_fn=tf.nn.sigmoid) #outputs the probability of moving left

#defining loss function
advantage=tf.placeholder(dtype=tf.float32,shape=[None,1])
responsible_action=tf.placeholder(dtype=tf.float32,shape=[None,1])
loss= tf.reduce_sum(advantage*policy)
#defining optimiser
optimiser = tf.train.RMSPropOptimizer(learning_rate=1e-4)
vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
print(vars)
grad = optimiser.compute_gradients(loss,vars)


def imgporcessing(img):
    img = greyscale(img)
    img = img[25:, :] #crop out irrelevant pixels
    img = misc.imresize(img, [80, 80, 1]) #resize to 80X80
    img = np.array(img)
    return img



def greyscale(img):
    """takes a colored image and return a black and white img"""
    grey_img = np.zeros([210, 160])
    for i in range(3):
        grey_img =np.sum([grey_img, img[:, :, i]], 0)
    grey_img /= 3
    grey_img = grey_img.astype(np.uint8)
    return grey_img

def discounted_reward(rewards):
    discounted_rewards=np.zeros_like(rewards)
    runing_add=0
    for t in reversed(range(0,rewards.size)):
        if rewards[t]!=0 : runing_add=0
        runing_add = runing_add*gamma+ rewards[t]
        discounted_rewards[t]=runing_add
    return discounted_rewards







sess.run(tf.global_variables_initializer())

env = gym.make("Pong-v0")
obs_buffer, reward_buffer, actionProbability_buffer, hiddenState_buffer = [], [], [], []
#playing a hundred episode
for i in range(1):
    done = False
    new_obs = env.reset()
    while not done:
        # env.render()
        # time.sleep(0.4)
        obs = imgporcessing(new_obs)
        obs = np.reshape(obs, [-1, 80, 80, 1])
        #print(obs.shape)
        hidden_state, action_probability = sess.run([dense, policy], feed_dict={input: obs})
        action_probability = action_probability.flat[0]
        action = np.random.choice(a=[2, 3], p=np.array([action_probability , 1-action_probability ])) #use the probability from policy network to decide on action
        new_obs, reward, done, _ = env.step(action)


        responsibleAction_probability = action_probability if action == 2 else 1 - action_probability

        obs_buffer.append(obs)
        reward_buffer.append(reward)
        actionProbability_buffer.append(responsibleAction_probability)
        hiddenState_buffer.append(hidden_state)

        if done:
            epX = np.vstack(obs_buffer)
            epr = np.vstack(reward_buffer)
            eph = np.vstack(hiddenState_buffer)
            epa =np.vstack(actionProbability_buffer)

            obs_buffer, reward_buffer, actionProbability_buffer, hiddenState_buffer = [], [], [], []
            epr = discounted_reward(epr)


            feed_dict = {input:epX,
                         dense:eph,
                         advantage:epr,
                         responsible_action:epa
                         }

            g = sess.run(grad, feed_dict=feed_dict)
            print(g)


