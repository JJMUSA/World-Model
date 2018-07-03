import gym,time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from matplotlib import pyplot as plt
from scipy import misc
gamma = 0.99


gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_option))

# with tf.device("/gpu:0"):
#defining CNN
input = tf.placeholder(dtype=tf.float32, shape=[None,84,84,1])
imageIn = tf.reshape(input,shape=[-1,84,84,1])
conv1 = slim.conv2d(activation_fn=tf.nn.elu,inputs=imageIn,num_outputs=16,kernel_size=[8,8],stride=[4,4],padding='VALID')
conv2 = slim.conv2d(activation_fn=tf.nn.elu,inputs=conv1,num_outputs=32,kernel_size=[4,4],stride=[2,2],padding='VALID')
dense = tf.contrib.layers.fully_connected(inputs=conv2, activation_fn=tf.nn.relu, num_outputs=256)


#Recurrent NN for memory and time dependecies
lstm = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)
c_init = np.zeros_like((1,lstm.state_size.c), np.float32)
h_init = np.zeros_like((1,lstm.state_size.h),np.float32)
state_init = [c_init, h_init]
c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
state_in = (c_in, h_in)
rnn_in = tf.expand_dims(dense, [0])
step_size =tf.shape(imageIn)[:1]
# print(step_size)
state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm, rnn_in, initial_state=state_in, sequence_length=[1,0],time_major=False)
lstm_c, lstm_h = lstm_state
state_out = (lstm_c[:1, :], lstm_h[:1, :])
rnn_out = tf.reshape(lstm_outputs, [-1, 256])


#defining Pollicy Net
policy_logit = tf.contrib.layers.fully_connected(inputs=dense, num_outputs=1, activation_fn=None) #outputs the probability of moving left
policy = tf.nn.sigmoid(policy_logit)

#defining loss function
advantage = tf.placeholder(dtype=tf.float32,shape=[None,1])
responsible_action = tf.placeholder(dtype=tf.float32,shape=[None,1])
entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=responsible_action,logits=policy_logit)
loss = tf.reduce_sum(advantage*entropy)

#defining optimiser
optimiser = tf.train.RMSPropOptimizer(learning_rate=1e-4,momentum=0.01,centered=True)
vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
grad = optimiser.compute_gradients(loss = loss,var_list = vars)
gradient_buffer = [tf.Variable(tf.zeros_like(g[1])) for g in grad]
update = optimiser.apply_gradients(zip(gradient_buffer,vars))
#saver for saving model
saver= tf.train.Saver(max_to_keep=1)
ckpt = tf.train.get_checkpoint_state('./Model')
saver.restore(sess,ckpt.model_checkpoint_path)
def imgporcessing(img):
    img = greyscale(img)
    img = img[25:, :] #crop out irrelevant pixels
    img = misc.imresize(img, [80, 80, 1]) #resize to 80X80
    #img = np.reshape(img,[np.prod(img.shape)])
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


'''def weight_update(gradients):
    # functions that call weight update based on accumlated gradient
    update = optimiser.apply_gradients(zip(gradients,vars))
    sess.run(update)
'''

def train(epx,batch_rnn, epa,epr):
    # normalising reward to ensure after every episode there is positive and negative gradients

    epr -= np.mean(epr)
    epr /= np.std(epr)

    feed_dict = {input: epx,
                 advantage: epr,
                 responsible_action: epa,
                 state_in[0]: batch_rnn[0],
                 state_in[1]:batch_rnn[1]
                 }

    l, g = sess.run([loss, grad], feed_dict=feed_dict)
    return (l, g)


# sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])

env = gym.make("Pong-v0")
gb = sess.run(gradient_buffer)
print(gb)
obs_buffer, reward_buffer, actionProbability_buffer, hiddenState_buffer = [], [], [], []

summary_writer = tf.summary.FileWriter('./training_summary')
episode_count = 0
#playing a hundred episode
while True:
    done = False
    new_obs = env.reset()
    rnn_state = state_init
    batch_rnn =rnn_state
    while not done:
        # env.render()
        # time.sleep(0.4)
        p_obs, obs = imgporcessing(new_obs), new_obs
        p_obs = np.reshape(p_obs, [-1, 6400])
        #print(obs.shape)
        rnn_state, action_probability = sess.run([state_out, policy], feed_dict={input: p_obs, state_in[0]: rnn_state[0], state_in[1]:rnn_state[1]})
        action_probability = action_probability.flat[0]
        action = np.random.choice(a=[2, 3], p=np.array([action_probability , 1-action_probability ])) #use the probability from policy network to decide on action
        new_obs, reward, done, _ = env.step(action)
        new_obs = new_obs-obs       	
        

        responsibleAction_probability = 1 if action == 2 else 0

        obs_buffer.append(p_obs)
        reward_buffer.append(reward)
        actionProbability_buffer.append(responsibleAction_probability)


        if done:
            epX = np.vstack(obs_buffer)
            epr = np.vstack(reward_buffer)
            epa = np.vstack(actionProbability_buffer)
            epR = discounted_reward(epr)

            obs_buffer, reward_buffer, actionProbability_buffer = [], [], [], []
            l, _g = train(epX,batch_rnn, epa, epR)
            gradient_buffer = [tf.add(gradient_buffer[i], g[1]) for i, g in enumerate(grad)]



            #recording performance
            summary = tf.Summary()
            summary.value.add(tag='Perf/Reward', simple_value=float(np.mean(epr)))
            summary.value.add(tag='Losses/Policy_loss', simple_value=float(l))
            summary_writer.add_summary(summary, episode_count)

            #weight update after every 10 episode
            if episode_count % 5 == 0:
                sess.run([update])
                gradient_buffer = [tf.Variable(tf.zeros_like(gradient_buffer[i])) for i in range(len(gradient_buffer))] # resetting buffer
                saver.save(sess, './Model/model-' + str(episode_count) + '.cptk')  # saving_model
                print('updated and saved model')
        episode_count += 1

