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

def discounted_rewards(rews, last_sv, gamma):
    '''
    Discounted reward to go 

    Parameters:
    ----------
    rews: list of rewards
    last_sv: value of the last state
    gamma: discount value 
    '''
    rtg = np.zeros_like(rews, dtype=np.float32)
    rtg[-1] = rews[-1] + gamma*last_sv
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

    def store(self, temp_traj, last_sv):
        '''
        Add temp_traj values to the buffers and compute the advantage and reward to go

        Parameters:
        -----------
        temp_traj: list where each element is a list that contains: observation, reward, action, state-value
        last_sv: value of the last state (Used to Bootstrap)
        '''
        # store only if the temp_traj list is not empty
        if len(temp_traj) > 0:
            self.obs.extend(temp_traj[:,0])
            rtg = discounted_rewards(temp_traj[:,1], last_sv, self.gamma)
            self.ret.extend(rtg - temp_traj[:,3])
            self.rtg.extend(rtg)
            self.act.extend(temp_traj[:,2])

    def get_batch(self):
        return self.obs, self.act, self.ret, self.rtg

    def __len__(self):
        assert(len(self.obs) == len(self.act) == len(self.ret) == len(self.rtg))
        return len(self.obs)
 
def AC(env_name, hidden_sizes=[32], ac_lr=5e-3, cr_lr=8e-3, num_epochs=50, gamma=0.99, steps_per_epoch=100, steps_to_print=100):
    '''
    Actor-Critic Algorithms
    Parameters:
    -----------
    env_name: Name of the environment
    hidden_size: list of the number of hidden units for each layer
    ac_lr: actor learning rate
    cr_lr: critic learning rate
    num_epochs: number of training epochs
    gamma: discount factor
    steps_per_epoch: number of steps per epoch
    '''

    # tf.reset_default_graph()의 TF2 버전
    tf.keras.backend.clear_session()    

    # 환경설정, 차원사이즈 설정 ====
    env = gym.make(env_name)    
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n 

    #####################################################
    ########### COMPUTE THE PG LOSS FUNCTIONS ###########
    #####################################################
    # policy gradient loss fun 계산(dim = 8)
    x = Input([obs_dim[0]])
    p_logits_model = mlp(x, hidden_sizes, act_dim, activation=tf.tanh)
    p_opt = optimizers.Adam(lr=ac_lr, )

    #######################################
    ###########  VALUE FUNCTION ###########
    #######################################
    # value fun 계산(dim = 8)
    x =Input([obs_dim[0]])
    s_value_model = mlp(x, hidden_sizes, 1, activation=tf.tanh)
    v_opt = optimizers.Adam(lr=cr_lr, )

    # Time 출력
    now = datetime.now()
    clock_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, now.second)
    print('Time:', clock_time)

    # 텐서플로에 실행정보 출력하는 코드
    hyp_str = '-steps_{}-aclr_{}-crlr_{}'.format(steps_per_epoch, ac_lr, cr_lr)
    file_writer = tf.summary.create_file_writer('log_dir/{}/AC_{}_{}'.format(env_name, clock_time, hyp_str))

    # few variables
    step_count = 0
    train_rewards = []
    train_ep_len = []
    timer = time.time()
    last_print_step = 0

    #Reset the environment at the beginning of the cycle
    obs = env.reset()
    ep_rews = []

    with file_writer.as_default():
        # main cycle
        for ep in range(num_epochs):

            # intiaizlie buffer and other variables for the new epochs
            buffer = Buffer(gamma)
            env_buf = []
            
            #iterate always over a fixed number of iterations
            for _ in range(steps_per_epoch):
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
                last_print_step += 1
                ep_rews.append(rew)

                if done:
                    # store the trajectory just completed
                    # Changed from REINFORCE! The second parameter is the estimated value of the next state. Because the environment is done. 
                    # we pass a value of 0
                    buffer.store(np.array(env_buf), 0)
                    env_buf = []
                    # store additionl information about the episode
                    train_rewards.append(np.sum(ep_rews))
                    train_ep_len.append(len(ep_rews))
                    # reset the environment
                    obs = env.reset()
                    ep_rews = []


            # Bootstrap with the estimated state value of the next state!
            if len(env_buf) > 0:
                tmp_obs = tf.convert_to_tensor(obs.reshape(-1, obs_dim[0]), dtype=tf.float32)            
                last_sv = tf.squeeze(s_value_model(tmp_obs))
                buffer.store(np.array(env_buf), last_sv)

            # collect the episodes' information
            obs_batch, act_batch, ret_batch, rtg_batch = buffer.get_batch()
            
            # run pre_scalar_summary before the optimization phase
            ac_variable = p_logits_model.trainable_variables

            # Optimize the actor and the critic                          
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
            tf.summary.scalar('p_log', p_log, step=step_count)  

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

            # run train_summary to save the summary after the optimization
            # p_loss
            actions_mask = tf.one_hot(act_batch, depth=act_dim)
            p_log = tf.reduce_sum(actions_mask * tf.nn.log_softmax(p_logits), axis=1)
            # entropy useful to study the algorithms
            entropy = -tf.reduce_mean(softmax_entropy(p_logits))
            p_loss = -tf.reduce_mean(p_log * ret_batch)        
            new_p_loss = p_loss

            # v_loss
            v_loss = tf.reduce_mean((rtg_batch - s_values)**2)
            new_v_loss = v_loss

            tf.summary.scalar('p_logits', p_logits, step=step_count) 
            tf.summary.scalar('s_values', s_values, step=step_count) 
            tf.summary.scalar('entropy', entropy, step=step_count) 
            tf.summary.scalar('old_v_loss', old_v_loss, step=step_count)
            tf.summary.scalar('old_p_loss', old_p_loss, step=step_count)
            tf.summary.scalar('diff/p_loss', (old_p_loss - new_p_loss), step_count)  
            tf.summary.scalar('diff/v_loss', (old_v_loss - new_v_loss), step_count)  
            file_writer.flush()

            # it's time to print some useful information
            if last_print_step > steps_to_print:
                print('Ep:%d MnRew:%.2f MxRew:%.1f EpLen:%.1f Buffer:%d -- Step:%d -- Time:%d' % (ep, np.mean(train_rewards), np.max(train_rewards), np.mean(train_ep_len), len(buffer), step_count, time.time()-timer))

                tf.summary.scalar('supplementary/len', np.mean(train_ep_len), step_count)  
                tf.summary.scalar('supplementary/train_rew', np.mean(train_rewards), step_count)  
                file_writer.flush()
                timer = time.time()
                train_rewards = []
                train_ep_len = []
                last_print_step = 0
        env.close()
    file_writer.close()


if __name__ == '__main__':
    AC('LunarLander-v2', hidden_sizes=[64], ac_lr=4e-3, cr_lr=1.5e-2, gamma=0.99, steps_per_epoch=100, steps_to_print=5000, num_epochs=8000)
