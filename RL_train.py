from __future__ import print_function
from collections import deque

from pg_actor_critic import PolicyGradientActorCritic
import tensorflow as tf
import numpy as np
import gym

env_name = "Pong-v0"
env = gym.make(env_name)

sess = tf.Session()
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
writer = tf.summary.FileWriter("tf_board/{}-experiment-1".format(env_name))

state_dim   = 6400
state_dim2= 64
num_actions = env.action_space.n

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()



def share_network(states):
  # define policy neural network
 
  W1_s = tf.get_variable("ac_sw1", [state_dim, 128],
                      initializer=tf.random_normal_initializer())
  b1_s = tf.get_variable("ac_sb1", [128],
                      initializer=tf.constant_initializer(0))
  h1_s = tf.nn.tanh(tf.matmul(states, W1_s) + b1_s)
  W2_s = tf.get_variable("ac_sw2", [128, 64],
                      initializer=tf.random_normal_initializer(stddev=0.1))
  b2_s = tf.get_variable("ac_sb2", [64],
                      initializer=tf.constant_initializer(0))
  p_s = tf.matmul(h1_s, W2_s) + b2_s

  return p_s




  
def actor_network(states):

  # define policy neural network
  W1 = tf.get_variable("ac_W1", [64, 20],
                       initializer=tf.random_normal_initializer())
  b1 = tf.get_variable("ac_b1", [20],
                       initializer=tf.constant_initializer(0))
  h1 = tf.nn.tanh(tf.matmul( states , W1) + b1)
  W2 = tf.get_variable("ac_W2", [20, num_actions],
                       initializer=tf.random_normal_initializer(stddev=0.1))
  b2 = tf.get_variable("ac_b2", [num_actions],
                       initializer=tf.constant_initializer(0))
  p = tf.matmul(h1, W2) + b2
  return p

def critic_network(states):
  # define policy neural network

  W1 = tf.get_variable("ct_W1", [64, 20],
                       initializer=tf.random_normal_initializer())
  b1 = tf.get_variable("ct_b1", [20],
                       initializer=tf.constant_initializer(0))
  h1 = tf.nn.tanh(tf.matmul(states, W1) + b1)
  W2 = tf.get_variable("ct_W2", [20, 1],
                       initializer=tf.random_normal_initializer())
  b2 = tf.get_variable("ct_b2", [1],
                       initializer=tf.constant_initializer(0))
  v = tf.matmul(h1, W2) + b2
  return v


pg_reinforce = PolicyGradientActorCritic(sess,
                                         optimizer,
                                         actor_network,
                                         critic_network,
                                         share_network,
                                         state_dim,
                                         state_dim2,
                                         num_actions,
                                         summary_writer=writer)

MAX_EPISODES = 1000000
MAX_STEPS    = 1000

no_reward_since = 0

episode_history = deque(maxlen=100)
for i_episode in range(MAX_EPISODES):

  # initialize
  state = prepro(env.reset())
  total_rewards = 0

  for t in range(MAX_STEPS):
    #env.render()
    action = pg_reinforce.sampleAction( state[np.newaxis,:])
    next_state, reward, done, _ = env.step(action)

    total_rewards += reward
    #reward = 5.0 if done else -0.1
    pg_reinforce.storeRollout( state , action, reward)

    state = prepro( next_state )

    # if reward > 0 or reward < 0 :
    #   state = prepro(env.reset())
    #   print ('reset', reward)
    #   break
    
    if done: break

  # if we don't see rewards in consecutive episodes
  # it's likely that the model gets stuck in bad local optima
  # we simply reset the model and try again
  if total_rewards <= -500:
    no_reward_since += 1
    if no_reward_since >= 5:
      # create and initialize variables
      print('Resetting model... start anew!')
      pg_reinforce.resetModel()
      no_reward_since = 0
      continue
  else:
    no_reward_since = 0

  pg_reinforce.updateModel()

  episode_history.append(total_rewards)
  mean_rewards = np.mean(episode_history)

  


  print("Episode {}".format(i_episode))
  print("Finished after {} timesteps".format(t+1))
  print("Reward for this episode: {}".format(total_rewards))
  print("Average reward for last 100 episodes: {:.2f}".format(mean_rewards))

  if i_episode%100==99:
    episode_history = deque(maxlen=100)

  if mean_rewards >= -100.0 and len(episode_history) >= 100:
    print("Environment {} solved after {} episodes".format(env_name, i_episode+1))
    break