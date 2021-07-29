#  관련 라이브러리를 가져오기
import numpy as np 
import tensorflow as tf
from tensorflow.keras import Model, Input, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import gym
from datetime import datetime
import time


def mlp(x, hidden_layers, output_size, activation=tf.nn.relu, last_activation=None):
    '''
    Multi-layer perceptron
    '''
    inputs = x
    for l in hidden_layers:
        x = Dense(units=l, activation=activation)(x)
    outputs = Dense(units=output_size, activation=last_activation)(x)
    model = Model(inputs, outputs)
    return model

def softmax_entropy(logits):
    '''
    Softmax Entropy
    '''
    return tf.reduce_sum(tf.nn.softmax(logits, axis=-1) * tf.nn.log_softmax(logits, axis=-1), axis=-1)


def discounted_rewards(rews, gamma):
    '''
    Discounted reward to go 

    Parameters:
    ----------
    rews: list of rewards
    gamma: discount value 
    '''
    rtg = np.zeros_like(rews, dtype=np.float32)
    rtg[-1] = rews[-1]
    for i in reversed(range(len(rews)-1)):
        rtg[i] = rews[i] + gamma*rtg[i+1]
    return rtg

class Buffer():
    '''
    Buffer class to store the experience from a unique policy
    '''
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.obs = []
        self.act = []
        self.ret = []
        self.rtg = []

    def store(self, temp_traj):
        '''
        Add temp_traj values to the buffers and compute the advantage and reward to go

        Parameters:
        -----------
        temp_traj: list where each element is a list that contains: observation, reward, action, state-value
        '''
        # store only if the temp_traj list is not empty
        if len(temp_traj) > 0:
            self.obs.extend(temp_traj[:,0])
            rtg = discounted_rewards(temp_traj[:,1], self.gamma)
            # NEW
            self.ret.extend(rtg - temp_traj[:,3])
            self.rtg.extend(rtg)
            self.act.extend(temp_traj[:,2])

    def get_batch(self):
        # MODIFIED
        return self.obs, self.act, self.ret, self.rtg

    def __len__(self):
        assert(len(self.obs) == len(self.act) == len(self.ret) == len(self.rtg))
        return len(self.obs)


def REINFORCE_baseline(env_name, hidden_sizes=[32], p_lr=5e-3, vf_lr=8e-3, gamma=0.99, steps_per_epoch=100, num_epochs=1000):
    '''
    REINFORCE with baseline Algorithm

    Parameters:
    -----------
    env_name: Name of the environment
    hidden_size: list of the number of hidden units for each layer
    p_lr: policy learning rate
    vf_lr: value function learning rate
    gamma: discount factor
    steps_per_epoch: number of steps per epoch
    num_epochs: number train epochs (Note: they aren't properly epochs)
    '''
    # TF2.0에 대한 tf.reset_default_graph()
    tf.keras.backend.clear_session() 

    env = gym.make(env_name)    
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n 

    #####################################################
    ########### COMPUTE THE PG LOSS FUNCTIONS ###########
    #####################################################
    # policy
    x = Input([obs_dim[0]])
    p_logits_model = mlp(x, hidden_sizes, act_dim, activation=tf.tanh)
    p_opt = optimizers.Adam(lr=p_lr,)    

    #######################################
    ###########  VALUE FUNCTION ###########
    #######################################
    # value fun 계산(dim = 8)
    x =Input([obs_dim[0]])
    s_value_model = mlp(x, hidden_sizes, 1, activation=tf.tanh)
    v_opt = optimizers.Adam(lr=vf_lr, )

    # Time
    now = datetime.now()
    clock_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, now.second)
    print('Time:', clock_time)

    hyp_str = '-steps_{}-plr_{}-vflr_{}'.format(steps_per_epoch, p_lr, vf_lr)
    file_writer = tf.summary.create_file_writer('log_dir/{}/REINFORCE_baseline_{}_{}'.format(env_name, clock_time, hyp_str))

    # few variables
    step_count = 0
    train_rewards = []
    train_ep_len = []
    timer = time.time()


    with file_writer.as_default():
        # main cycle
        for ep in range(num_epochs):

            # initialize environment for the new epochs
            obs = env.reset()

            # intiaizlie buffer and other variables for the new epochs
            buffer = Buffer(gamma)
            env_buf = []
            ep_rews = []
            
            while len(buffer) < steps_per_epoch:

                tmp_obs = tf.convert_to_tensor(obs.reshape(-1, obs_dim[0]), dtype=tf.float32)
                p_logits = p_logits_model(tmp_obs)
                act = tf.squeeze(tf.random.categorical(p_logits, 1))

                s_values = tf.squeeze(s_value_model(tmp_obs))
                val = tf.squeeze(s_values)

                # take a step in the environment
                obs2, rew, done, _ = env.step(np.squeeze(act))

                # add the new transition
                env_buf.append([obs.copy(), rew, act, np.squeeze(val)])

                obs = obs2.copy()

                step_count += 1
                ep_rews.append(rew)

                if done:
                    # store the trajectory just completed
                    buffer.store(np.array(env_buf))
                    env_buf = []
                    # store additionl information about the episode
                    train_rewards.append(np.sum(ep_rews))
                    train_ep_len.append(len(ep_rews))
                    # reset the environment
                    obs = env.reset()
                    ep_rews = []

            # collect the episodes' information
            obs_batch, act_batch, ret_batch, rtg_batch = buffer.get_batch()
            
            ac_variable = p_logits_model.trainable_variables                
            with tf.GradientTape() as tape:
                # automatic differentiation에 대한 연산내용을 기록함
                tape.watch(ac_variable)            
                # p_loss
                actions_mask = tf.one_hot(act_batch, depth=act_dim)
                tmp_obs = tf.convert_to_tensor(obs_batch, dtype=tf.float32)            
                p_logits = p_logits_model(tmp_obs)
                p_log = tf.reduce_sum(actions_mask * tf.nn.log_softmax(p_logits), axis=1)
                # entropy useful to study the algorithms
                entropy = -tf.reduce_mean(softmax_entropy(p_logits))
                p_loss = -tf.reduce_mean(p_log * ret_batch)        
                old_p_loss = p_loss

            ac_grads = tape.gradient(p_loss, ac_variable)
            p_opt.apply_gradients(zip(ac_grads, ac_variable))

            tf.summary.scalar('p_loss', p_loss, step=step_count) 
            tf.summary.scalar('entropy', entropy, step=step_count) 
            tf.summary.scalar('p_log', p_log, step=step_count) 
            tf.summary.scalar('p_logits', p_logits, step=step_count) 
            tf.summary.scalar('old_p_loss', old_p_loss, step=step_count) 

            cr_variable = s_value_model.trainable_variables                
            with tf.GradientTape() as tape:
                tape.watch(cr_variable)   
                tmp_obs = tf.convert_to_tensor(obs_batch, dtype=tf.float32)
                s_values = tf.squeeze(s_value_model(tmp_obs))

                # v_loss
                v_loss = tf.reduce_mean((rtg_batch - s_values)**2)
                old_v_loss = v_loss

            cr_grads = tape.gradient(v_loss, cr_variable)
            v_opt.apply_gradients(zip(cr_grads, cr_variable))

            tf.summary.scalar('v_loss', v_loss, step=step_count) 
            tf.summary.scalar('s_values', s_values, step=step_count)
            tf.summary.scalar('old_v_loss', old_v_loss, step=step_count)
            file_writer.flush()

            # it's time to print some useful information
            if ep % 10 == 0:
                print('Ep:%d MnRew:%.2f MxRew:%.1f EpLen:%.1f Buffer:%d -- Step:%d -- Time:%d' % (ep, np.mean(train_rewards), np.max(train_rewards), np.mean(train_ep_len), len(buffer), step_count,time.time()-timer))
                timer = time.time()
                train_rewards = []
                train_ep_len = []
        env.close()
    file_writer.close()

if __name__ == '__main__':
    REINFORCE_baseline('LunarLander-v2', hidden_sizes=[64], p_lr=8e-3, vf_lr=7e-3, gamma=0.99, steps_per_epoch=1000, num_epochs=1000)
