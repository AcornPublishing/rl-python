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



def DDPG(env_name, hidden_sizes=[32], ac_lr=1e-2, cr_lr=1e-2, num_epochs=2000, buffer_size=5000, discount=0.99, render_cycle=100, mean_summaries_steps=1000, 
        batch_size=128, min_buffer_size=5000, tau=0.005):

    # Create an environment for training
    env = gym.make(env_name)
    # Create an environment for testing the actor
    env_test = gym.make(env_name)

    tf.keras.backend.clear_session()
    # tf.compat.v1.reset_default_graph()

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    print('-- Observation space:', obs_dim, ' Action space:', act_dim, '--')

    tmp_obs_ph11=Input([obs_dim[0]])
    tmp_obs_ph12=Input([obs_dim[0]+act_dim[0]])
    tmp_act_ph=Input([act_dim[0]])
    tmp_obs_ph13=Input([obs_dim[0]+act_dim[0]])

    # Create an online deterministic actor-critic 
    with tf.compat.v1.variable_scope('online'):
        with tf.compat.v1.variable_scope('p_mlp'):
            p_onl_model = mlp(tmp_obs_ph11, hidden_sizes, act_dim[0], activation=tf.nn.relu, last_activation=tf.tanh) 
        
        with tf.compat.v1.variable_scope('q_mlp'):
            qd_onl_model = mlp(tmp_obs_ph12, hidden_sizes, 1, activation=tf.nn.relu, last_activation=None)
            qa_onl_model = mlp(tmp_obs_ph13, hidden_sizes, 1, activation=tf.nn.relu, last_activation=None) 

    with tf.compat.v1.variable_scope('target'):
        qd_tar_model = mlp(tmp_obs_ph12, hidden_sizes, 1, activation=tf.nn.relu, last_activation=None)


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

    # Optimize the critic
    q_opt = optimizers.Adam(lr=cr_lr, )
    # Optimize the actor
    p_opt = optimizers.Adam(lr=ac_lr, ) 

    def agent_op(o):
        p_onl =  np.max(env.action_space.high) * p_onl_model.predict(o.reshape(-1, obs_dim[0])) 
        a = np.squeeze(p_onl)
        return np.clip(a, env.action_space.low, env.action_space.high)

    def agent_noisy_op(o, scale):
        action = agent_op(o)
        noisy_action = action + np.random.normal(loc=0.0, scale=scale, size=action.shape)
        return np.clip(noisy_action, env.action_space.low, env.action_space.high)

    # Time
    now = datetime.now()
    clock_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, int(now.second))
    print('Time:', clock_time)

    hyp_str = '-aclr_'+str(ac_lr)+'-crlr_'+str(cr_lr)+'-tau_'+str(tau)
    file_writer = tf.summary.create_file_writer('log_dir/'+env_name+'/DDPG_'+clock_time+'_'+hyp_str, tf.get_default_graph())

    get_init_target_op()
    
    # Some useful variables..
    render_the_game = False
    step_count = 0
    last_q_update_loss = []
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
                    act = agent_noisy_op(obs, 0.1)

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

                    # Compute the target values
                    # p_means를 만들기
                    tmp_mb_obs2 = tf.convert_to_tensor(mb_obs2,dtype=tf.float32)
                    tmp_q_target_mb = p_onl_model(tmp_mb_obs2)
                    # qd_tar_model을 실행한 후 결과값을 tf.squeeze실행
                    tmp_mb_obs = tf.convert_to_tensor(mb_obs,dtype=tf.float32)                    
                    q_target_mb = np.max(env.action_space.high) * tmp_q_target_mb

                    tmp_qd_tar = qd_tar_model(tf.concat([tmp_mb_obs, q_target_mb], axis=-1))    
                    tmp_qd_tar = tf.squeeze(tmp_qd_tar)        
                    y_r = np.array(mb_rew) + discount*(1-np.array(mb_done))*tmp_qd_tar

                    with tf.GradientTape() as tape:           
                        # Actor loss
                        tmp_qd_onl = qd_onl_model(tf.concat([tmp_mb_obs, q_target_mb], axis=-1))  #mb_obs.reshape([obs_dim[0]]))
                        tmp_qd_onl = tf.squeeze(tmp_qd_onl)
                        p_train_loss = -tf.reduce_mean(tmp_qd_onl)

                    # optimize the actor
                    # gradients를 계산, gradients를 update                
                    train1_grads = tape.gradient(p_train_loss, qd_onl_model.trainable_variables)
                    p_opt.apply_gradients(zip(train1_grads, qd_onl_model.trainable_variables))

                    with tf.GradientTape() as tape:  
                        # Critic loss (MSE)
                        tmp_mb_obs = tf.convert_to_tensor(mb_obs,dtype=tf.float32)
                        tmp_qa_onl = qa_onl_model(tf.concat([tmp_mb_obs, mb_act], axis=-1))
                        tmp_qa_onl = tf.squeeze(tmp_qa_onl)
                        q_train_loss = tf.reduce_mean((tmp_qa_onl - y_r)**2) 
                    
                    # optimize the critic
                    # gradients를 계산 및 update
                    train2_grads = tape.gradient(q_train_loss, qa_onl_model.trainable_variables)
                    q_opt.apply_gradients(zip(train2_grads, qa_onl_model.trainable_variables))

                    tf.summary.scalar('loss/q', q_loss, step=step_count)
                    tf.summary.scalar('loss/p', p_loss, step=step_count)

                    # Soft update of the target networks           
                    get_update_target_op()
                    # sess.run(update_target_op)

                    last_q_update_loss.append(q_train_loss)
                    last_p_update_loss.append(p_train_loss)

                    # some 'mean' summaries to plot more smooth functions
                    if step_count % mean_summaries_steps == 0:
                        # summary = tf.Summary()  
                        tf.summary.scalar(tag='loss/mean_q', np.mean(last_q_update_loss), step_count)
                        tf.summary.scalar(tag='loss/mean_p', np.mean(last_p_update_loss), step_count)
                        file_writer.flush()

                        last_q_update_loss = []
                        last_p_update_loss = []


                if done:
                    obs = env.reset()
                    batch_rew.append(g_rew)
                    g_rew, render_the_game = 0, False

            # Test the actor every 10 epochs
            if ep % 10 == 0:
                test_mn_rw, test_std_rw = test_agent(env_test, agent_op)
                tf.summary.scalar(tag='test/reward', test_mn_rw, step_count)
                file_writer.flush()

                ep_sec_time = int((current_milli_time()-ep_time) / 1000)
                print('Ep:%4d Rew:%4.2f -- Step:%5d -- Test:%4.2f %4.2f -- Time:%d' %  (ep,np.mean(batch_rew), step_count, test_mn_rw, test_std_rw, ep_sec_time))

                ep_time = current_milli_time()
                batch_rew = []
                    
            if ep % render_cycle == 0:
                render_the_game = True

        # close everything
        env.close()
        env_test.close()
    file_writer.close()

if __name__ == '__main__':
    DDPG('BipedalWalker-v3', hidden_sizes=[64,64], ac_lr=3e-4, cr_lr=4e-4, buffer_size=2000, mean_summaries_steps=100, batch_size=64, 
        min_buffer_size=1000, tau=0.003)
