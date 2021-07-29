import numpy as np 
import tensorflow as tf
from tensorflow.keras import Model, Input, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import gym
from datetime import datetime
import time
from collections import deque

current_milli_time = lambda: int(round(time.time() * 1000))

def mlp(x, hidden_layers, output_layer, activation=tf.tanh, last_activation=None):
    '''
    Multi-layer perceptron
    '''
    inputs = x
    for l in hidden_layers:
        x = Dense(units=l, activation=activation)(x)
    outputs = Dense(units=output_layer, activation=last_activation)(x)
    model = Model(inputs, outputs)
    return model

class ExperiencedBuffer():
    '''
    Experienced buffer
    '''
    def __init__(self, buffer_size):
        # Contains up to 'buffer_size' experience
        self.obs_buf = deque(maxlen=buffer_size)
        self.rew_buf = deque(maxlen=buffer_size)
        self.act_buf = deque(maxlen=buffer_size)
        self.obs2_buf = deque(maxlen=buffer_size)
        self.done_buf = deque(maxlen=buffer_size)

    def add(self, obs, rew, act, obs2, done):
        '''
        Add a new transition to the buffers
        '''
        self.obs_buf.append(obs)
        self.rew_buf.append(rew)
        self.act_buf.append(act)
        self.obs2_buf.append(obs2)
        self.done_buf.append(done)
        
    def sample_minibatch(self, batch_size):
        '''
        Sample a mini-batch of size 'batch_size'
        '''
        mb_indices = np.random.randint(len(self.obs_buf), size=batch_size)

        mb_obs = [self.obs_buf[i] for i in mb_indices]
        mb_rew = [self.rew_buf[i] for i in mb_indices]
        mb_act = [self.act_buf[i] for i in mb_indices]
        mb_obs2 = [self.obs2_buf[i] for i in mb_indices]
        mb_done = [self.done_buf[i] for i in mb_indices]

        return mb_obs, mb_rew, mb_act, mb_obs2, mb_done

    def __len__(self):
        return len(self.obs_buf)

def test_agent(env_test, agent_op, num_games=10):
    '''
    Test an agent 'agent_op', 'num_games' times
    Return mean and std
    '''
    games_r = []
    for _ in range(num_games):
        d = False
        game_r = 0
        o = env_test.reset()

        while not d:
            a_s = agent_op(o)
            o, r, d, _ = env_test.step(a_s)
            game_r += r
        games_r.append(game_r)
    return np.mean(games_r), np.std(games_r)


def TD3(env_name, hidden_sizes=[32], ac_lr=1e-2, cr_lr=1e-2, num_epochs=2000, buffer_size=5000, discount=0.99, render_cycle=10000, mean_summaries_steps=1000, 
        batch_size=128, min_buffer_size=5000, tau=0.005, target_noise=0.2, expl_noise=0.1, policy_update_freq=2):

    # Create an environment for training
    env = gym.make(env_name)
    # Create an environment for testing the actor
    env_test = gym.make(env_name)

    # tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    print('-- Observation space:', obs_dim, ' Action space:', act_dim, '--')

    tmp_obs_ph11=Input([obs_dim[0]])
    tmp_obs_ph12=Input([obs_dim[0]+act_dim[0]])
    tmp_act_ph=Input([act_dim[0]])
    tmp_obs_ph13=Input([obs_dim[0]+act_dim[0]])

    # Create an online deterministic actor and a double critic 
    with tf.compat.v1.variable_scope('online'):
        # p_onl, qd1_onl, qa1_onl, qa2_onl값 계산
        with tf.compat.v1.variable_scope('p_mlp'):
            p_onl_model = mlp(tmp_obs_ph11, hidden_sizes, act_dim[0], activation=tf.nn.relu, last_activation=tf.tanh) 
        with tf.compat.v1.variable_scope('q1_mlp'):
            qd1_onl_model = mlp(tmp_obs_ph12, hidden_sizes, 1, activation=tf.nn.relu, last_activation=None)
        with tf.compat.v1.variable_scope('q1_mlp', reuse=True):            
            qa1_onl_model = mlp(tmp_obs_ph13, hidden_sizes, 1,  activation=tf.nn.relu, last_activation=None)
        with tf.compat.v1.variable_scope('q2_mlp', reuse=True):            
            qa2_onl_model = mlp(tmp_obs_ph13, hidden_sizes, 1,  activation=tf.nn.relu, last_activation=None)

    qa1_onl_var = qa1_onl_model.trainable_variables
    qa2_onl_var = qa2_onl_model.trainable_variables

    # and a target actor and double critic
    with tf.compat.v1.variable_scope('target'):
        # p_tar, qa_tar, qa2_tar값 계산
        with tf.compat.v1.variable_scope('p_mlp'):
            p_tar_model = mlp(tmp_obs_ph11, hidden_sizes, act_dim[0], activation=tf.nn.relu, last_activation=tf.tanh) 
        with tf.compat.v1.variable_scope('q1_mlp', reuse=True):   
            qa1_tar_model = mlp(tmp_obs_ph13, hidden_sizes, 1,  activation=tf.nn.relu, last_activation=None)
        with tf.compat.v1.variable_scope('q2_mlp', reuse=True):             
            qa2_tar_model = mlp(tmp_obs_ph13, hidden_sizes, 1,  activation=tf.nn.relu, last_activation=None)

    qd1_onl_var = qd1_onl_model.trainable_variables

    def variables_in_scope(scope):
        '''
        Retrieve all the variables in the scope 'scope'
        '''
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope)

    # Copy all the online variables to the target networks i.e. target = online
    # Needed only at the beginning
    def get_init_target_op():
        init_target = [target_var.assign(online_var) for target_var, online_var in zip(variables_in_scope('target'), variables_in_scope('online'))]
        init_target_op = tf.group(*init_target)

    # Soft update
    def get_update_target_op():
        update_target = [target_var.assign(tau*online_var + (1-tau)*target_var) for target_var, online_var in zip(variables_in_scope('target'), variables_in_scope('online'))]
        update_target_op = tf.group(*update_target)
  
    # Optimize the critics
    q1_opt = optimizers.Adam(lr=cr_lr, )
    q2_opt = optimizers.Adam(lr=cr_lr, )

    # Optimize the actor
    p_opt = optimizers.Adam(lr=ac_lr, )    

    def add_normal_noise(x, scale, low_lim=-0.5, high_lim=0.5):
        return x + np.clip(np.random.normal(loc=0.0, scale=scale, size=x.shape), low_lim, high_lim)

    def agent_op(o):
        # =====================>        
        p_onl =  np.max(env.action_space.high) * p_onl_model.predict(o.reshape(-1, obs_dim[0])) 
        ac = np.squeeze(p_onl)

        # ac = np.squeeze(sess.run(p_onl, feed_dict={obs_ph:[o]}))
        return np.clip(ac, env.action_space.low, env.action_space.high)

    def agent_noisy_op(o, scale):
        ac = agent_op(o)
        return np.clip(add_normal_noise(ac, scale, env.action_space.low, env.action_space.high), env.action_space.low, env.action_space.high)


    # Time
    now = datetime.now()
    clock_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, int(now.second))
    print('Time:', clock_time)

    # Set TensorBoard
    # tf.summary.scalar('loss/q1', q1_loss)
    # tf.summary.scalar('loss/q2', q2_loss)
    # tf.summary.scalar('loss/p', p_loss)
    # scalar_summary = tf.summary.merge_all()

    hyp_str = '-aclr_'+str(ac_lr)+'-crlr_'+str(cr_lr)+'-tau_'+str(tau)
    file_writer = tf.summary.create_file_writer('log_dir/{}/REINFORCE_baseline_{}_{}'.format(env_name, clock_time, hyp_str))

    # Create a session and initialize the variables
    get_init_target_op()
    
    # Some useful variables..
    render_the_game = False
    step_count = 0
    last_q1_update_loss = []
    last_q2_update_loss = []
    last_p_update_loss = []
    ep_time = current_milli_time()
    batch_rew = []

    # Reset the environment
    obs = env.reset()
    # Initialize the buffer
    buffer = ExperiencedBuffer(buffer_size)

    with file_writer.as_default():
        for ep in range(num_epochs):
            g_rew = 0
            done = False

            while not done:
                # If not gathered enough experience yet, act randomly
                if len(buffer) < min_buffer_size:
                    act = env.action_space.sample()
                else:
                    act = agent_noisy_op(obs, expl_noise)

                # Take a step in the environment
                obs2, rew, done, _ = env.step(act)

                if render_the_game:
                    env.render()

                # Add the transition in the buffer
                buffer.add(obs.copy(), rew, act, obs2.copy(), done)

                obs = obs2
                g_rew += rew
                step_count += 1

                if len(buffer) > min_buffer_size:
                    
                    # sample a mini batch from the buffer
                    mb_obs, mb_rew, mb_act, mb_obs2, mb_done = buffer.sample_minibatch(batch_size)

                    tmp_mb_obs2 = tf.convert_to_tensor(mb_obs2,dtype=tf.float32)
                    double_actions = p_tar_model(tmp_mb_obs2)
                    # Target regularization
                    double_noisy_actions = np.clip(add_normal_noise(double_actions, target_noise), env.action_space.low, env.action_space.high)

                    # Clipped Double Q-learning           
                    q1_target_mb = qa1_tar_model(tf.concat([tmp_mb_obs2, double_noisy_actions], axis=-1)) 
                    q2_target_mb = qa2_tar_model(tf.concat([tmp_mb_obs2, double_noisy_actions], axis=-1)) 
                    q1_target_mb = tf.squeeze(q1_target_mb)
                    q2_target_mb = tf.squeeze(q2_target_mb)

                    # q1_target_mb, q2_target_mb = sess.run([qa1_tar,qa2_tar], feed_dict={obs_ph:mb_obs2, act_ph:double_noisy_actions})
                    q_target_mb = np.min([q1_target_mb, q2_target_mb], axis=0) 
                    assert(len(q1_target_mb) == len(q_target_mb))

                    # Compute the target values
                    y_r = np.array(mb_rew) + discount*(1-np.array(mb_done))*q_target_mb

                    tmp_mb_obs = tf.convert_to_tensor(mb_obs, dtype=tf.float32)               
                    tmp_mb_act = tf.convert_to_tensor(mb_act, dtype=tf.float32)   
                    with tf.GradientTape() as tape:  
                        # Optimize the critics
                        tmp_qa1_onl = qa1_onl_model(tf.concat([tmp_mb_obs, tmp_mb_act], axis=-1))
                        tmp_qa1_onl = tf.squeeze(tmp_qa1_onl)
                        q1_train_loss = tf.reduce_mean((tmp_qa1_onl - y_r)**2) 

                    train1_grads = tape.gradient(q1_train_loss, qa1_onl_var)
                    q1_opt.apply_gradients(zip(train1_grads, qa1_onl_var))

                    tmp_mb_obs = tf.convert_to_tensor(mb_obs, dtype=tf.float32)               
                    tmp_mb_act = tf.convert_to_tensor(mb_act, dtype=tf.float32)   
                    with tf.GradientTape() as tape:  
                        tmp_qa2_onl = qa2_onl_model(tf.concat([tmp_mb_obs, tmp_mb_act], axis=-1)) 
                        tmp_qa2_onl = tf.squeeze(tmp_qa2_onl)
                        q2_train_loss = tf.reduce_mean((tmp_qa2_onl - y_r)**2)

                    train2_grads = tape.gradient(q2_train_loss, qa2_onl_var)
                    q2_opt.apply_gradients(zip(train2_grads, qa2_onl_var))

                    # Delayed policy update
                    if step_count % policy_update_freq == 0:
                        # Optimize the policy
                        tmp_mb_obs = tf.convert_to_tensor(mb_obs, dtype=tf.float32)        
                        with tf.GradientTape() as tape: 
                            q_target_mb = p_tar_model(tmp_mb_obs)
                            tmp_qd1_onl = qd1_onl_model(tf.concat([tmp_mb_obs, q_target_mb], axis=-1))
                            tmp_qd1_onl = tf.squeeze(tmp_qd1_onl)
                            p_train_loss = -tf.reduce_mean(tmp_qd1_onl)    

                        train3_grads = tape.gradient(p_train_loss, qd1_onl_var)
                        p_opt.apply_gradients(zip(train3_grads, qd1_onl_var))                    

                        # Soft update of the target networks
                        get_update_target_op()

                        # file_writer.add_summary(train_summary, step_count)
                        tf.summary.scalar(tag='loss/q1', q1, step_count)
                        tf.summary.scalar(tag='loss/q2', q2, step_count)
                        tf.summary.scalar(tag='loss/p', p_loss, step_count)
                        file_writer.flush()

                        last_q1_update_loss.append(q1_train_loss)
                        last_q2_update_loss.append(q2_train_loss)
                        last_p_update_loss.append(p_train_loss)

                    # some 'mean' summaries to plot more smooth functions
                    if step_count % mean_summaries_steps == 0:
                        tf.summary.scalar(tag='loss/mean_q1', np.mean(last_q1_update_loss), step_count)
                        tf.summary.scalar(tag='loss/mean_q2', np.mean(last_q2_update_loss), step_count)
                        tf.summary.scalar(tag='loss/mean_p', np.mean(last_p_update_loss), step_count)
                        file_writer.flush()

                        last_q1_update_loss = []
                        last_q2_update_loss = []
                        last_p_update_loss = []


                if done:
                    obs = env.reset()
                    batch_rew.append(g_rew)
                    g_rew, render_the_game = 0, False

            # Test the actor every 10 epochs
            if ep % 10 == 0:
                test_mn_rw, test_std_rw = test_agent(env_test, agent_op)
                tf.summary.scalar(tag = 'test/reward', test_mn_rw, step_count) 
                file_writer.flush()

                ep_sec_time = int((current_milli_time()-ep_time) / 1000)
                print('Ep:%4d Rew:%4.2f -- Step:%5d -- Test:%4.2f %4.2f -- Time:%d' %  (ep,np.mean(batch_rew), step_count, test_mn_rw, test_std_rw, ep_sec_time))

                ep_time = current_milli_time()
                batch_rew = []
                    
            if ep % render_cycle == 0:
                render_the_game = True

        env.close()
        env_test.close()
    # close everything
    file_writer.close()


if __name__ == '__main__':
    TD3('BipedalWalker-v3', hidden_sizes=[64,64], ac_lr=4e-4, cr_lr=4e-4, buffer_size=200000, mean_summaries_steps=100, batch_size=64, 
        min_buffer_size=10000, tau=0.005, policy_update_freq=2, target_noise=0.1)
