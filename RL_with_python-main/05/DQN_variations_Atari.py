#  관련 라이브러리를 가져오기
import numpy as np 
import tensorflow as tf
from tensorflow.keras import Model, Input, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import gym
from datetime import datetime
from collections import deque
import time
import sys

# 아타리 게임 관련 정보를 갖고오기, 게임레벨 설정
from atari_wrappers import make_env
gym.logger.set_level(40)
current_milli_time = lambda: int(round(time.time() * 1000))

# MultiStepExperienceBuffer클래스(리플레이 메모리로 사용)
class MultiStepExperienceBuffer():
    '''
    Experience Replay Buffer for multi-step learning
    '''
    def __init__(self, buffer_size, n_step, gamma):
        self.obs_buf = deque(maxlen=buffer_size)
        self.act_buf = deque(maxlen=buffer_size)

        self.n_obs_buf = deque(maxlen=buffer_size)
        self.n_done_buf = deque(maxlen=buffer_size)
        self.n_rew_buf = deque(maxlen=buffer_size)

        self.n_step = n_step
        self.last_rews = deque(maxlen=self.n_step+1)
        self.gamma = gamma

    def add(self, obs, rew, act, obs2, done):
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        # the following buffers will be updated in the next n_step steps
        # their values are not known, yet
        self.n_obs_buf.append(None)
        self.n_rew_buf.append(None)
        self.n_done_buf.append(None)

        self.last_rews.append(rew)

        ln = len(self.obs_buf)
        len_rews = len(self.last_rews)

        # Update the indices of the buffer that are n_steps old
        if done:
            # In case it's the last step, update up to the n_steps indices fo the buffer
            # it cannot update more than len(last_rews), otherwise will update the previous traj
            for i in range(len_rews):
                self.n_obs_buf[ln-(len_rews-i-1)-1] = obs2
                self.n_done_buf[ln-(len_rews-i-1)-1] = done
                rgt = np.sum([(self.gamma**k)*r for k,r in enumerate(np.array(self.last_rews)[i:len_rews])])
                self.n_rew_buf[ln-(len_rews-i-1)-1] = rgt

            # reset the reward deque
            self.last_rews = deque(maxlen=self.n_step+1)
        else:
            # Update the elements of the buffer that has been added n_step steps ago
            # Add only if the multi-step values are updated
            if len(self.last_rews) >= (self.n_step+1):
                self.n_obs_buf[ln-self.n_step-1] = obs2
                self.n_done_buf[ln-self.n_step-1] = done
                rgt = np.sum([(self.gamma**k)*r for k,r in enumerate(np.array(self.last_rews)[:len_rews])])
                self.n_rew_buf[ln-self.n_step-1] = rgt
        

    def sample_minibatch(self, batch_size):
        # Sample a minibatch of size batch_size
        # Note: the samples should be at least of n_step steps ago
        mb_indices = np.random.randint(len(self.obs_buf)-self.n_step, size=batch_size)

        mb_obs = scale_frames([self.obs_buf[i] for i in mb_indices])
        mb_rew = [self.n_rew_buf[i] for i in mb_indices]
        mb_act = [self.act_buf[i] for i in mb_indices]
        mb_obs2 = scale_frames([self.n_obs_buf[i] for i in mb_indices])
        mb_done = [self.n_done_buf[i] for i in mb_indices]

        return mb_obs, mb_rew, mb_act, mb_obs2, mb_done

    def __len__(self):
        return len(self.obs_buf)

def greedy(action_values):
    '''
    Greedy policy
    '''
    return np.argmax(action_values)

def eps_greedy(action_values, eps=0.1):
    '''
    Eps-greedy policy
    '''
    if np.random.uniform(0,1) < eps:
        # Choose a uniform random action
        return np.random.randint(len(action_values))
    else:
        # Choose the greedy action
        return np.argmax(action_values)

def test_agent(env_test, agent_op, num_games=20):
    '''
    Test an agent
    '''
    games_r = []

    for _ in range(num_games):
        d = False
        game_r = 0
        o = env_test.reset()

        while not d:
            # Use an eps-greedy policy with eps=0.05 (to add stochasticity to the policy)
            # Needed because Atari envs are deterministic
            # If you would use a greedy policy, the results will be always the same
            a = eps_greedy(np.squeeze(agent_op(o)), eps=0.05)
            o, r, d, _ = env_test.step(a)

            game_r += r

        games_r.append(game_r)

    return games_r

def scale_frames(frames):
    '''
    Scale the frame with number between 0 and 1
    '''
    return np.array(frames, dtype=np.float32) / 255.0

# 모델링 개발===================
def cnn(x):
    '''
    Convolutional neural network
    '''
    inputs = x
    x = Conv2D(filters=16, kernel_size=8, strides=4, padding='valid', activation=tf.nn.relu)(inputs)
    x = Conv2D(filters=32, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)(x)
    x = Conv2D(filters=32, kernel_size=3, strides=1, padding='valid', activation=tf.nn.relu)(x)
    return(x)

def fnn(x, hidden_layers, output_layer, activation=tf.nn.relu, last_activation= None):
    '''
    Feed-forward neural network
    '''
    for l in hidden_layers:
        x = Dense(units=l, activation=activation)(x)
    x = Dense(units=output_layer, activation=last_activation)(x)  
    return(x)

# qnet으로 함수명을 변경하도록 함
# 함수형 모델(케라스)을 적용한 코드
def qnet(x, hidden_layers, output_size, activation=tf.nn.relu, last_activation= None):
    '''
    Deep Q network: CNN followed by FNN
    '''
    inputs=x
    x = cnn(inputs)
    x = Flatten()(x) 
    outputs = fnn(x, hidden_layers, output_size, activation, last_activation)
    model = Model(inputs, outputs)
    return model

def dueling_qnet(x, hidden_layers, output_size, activation=tf.nn.relu, last_activation= None):
    '''
    Dueling neural network
    '''
    inputs = x
    x = cnn(inputs)
    x = Flatten()(x)
    qf = fnn(x, hidden_layers, 1, activation, last_activation)
    aaqf = fnn(x, hidden_layers, output_size, activation, last_activation)
    outputs = qf + aaqf - tf.reduce_mean(aaqf)
    model = Model(inputs, outputs)
    return model


def DQN_with_variations(env_name, extensions_hyp, hidden_sizes=[32], lr=1e-2, num_epochs=2000, buffer_size=100000, discount=0.99, render_cycle=100, update_target_net=1000, 
        batch_size=64, update_freq=4, frames_num=2, min_buffer_size=5000, test_frequency=20, start_explor=1, end_explor=0.1, explor_steps=100000):

    # Create the environment both for train and test
    env = make_env(env_name, frames_num=frames_num, skip_frames=True, noop_num=20)
    env_test = make_env(env_name, frames_num=frames_num, skip_frames=True, noop_num=20)
    # Add a monitor to the test env to store the videos
    env_test = gym.wrappers.Monitor(env_test, "VIDEOS/TEST_VIDEOS"+env_name+str(current_milli_time()),force=True, video_callable=lambda x: x%20==0)

    tf.keras.backend.clear_session() 

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n 


    # Train(Online)모델 생성 -> Target 모델 복사
    x = Input([obs_dim[0], obs_dim[1], obs_dim[2]])        
    # Train모델 생성
    train_model = dueling_qnet(x, hidden_sizes, act_dim)  #, lr)    
    # Target모델 복사
    target_model = tf.keras.models.clone_model(train_model)

    def agent_op(o):
        '''
        Forward pass to obtain the Q-values from the online network of a single observation
        '''
        # Scale the frames
        o = scale_frames(o)
        o = o.reshape(1, obs_dim[0], obs_dim[1], obs_dim[2])
        temp = train_model([o])        
        return temp

    # Time
    now = datetime.now()
    clock_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, int(now.second))
    print('Time:', clock_time)

    mr_v = tf.Variable(0.0)
    ml_v = tf.Variable(0.0)


    # 텐서플로에 실행정보 출력하는 코드
    LOG_DIR = 'log_dir/'+env_name
    hyp_str = "-lr_{}-upTN_{}-upF_{}-frms_{}" .format(lr, update_target_net, update_freq, frames_num)

    # 텐서보드 서머리에 관련 데이터 저장을 위한 File Writer초기화
    file_writer = tf.summary.create_file_writer(LOG_DIR+'/DQN_'+clock_time+'_'+hyp_str)
   
    render_the_game = False
    step_count = 0
    last_update_loss = []
    ep_time = current_milli_time()
    batch_rew = []
    old_step_count = 0
    opt = optimizers.Adam(lr=lr, )

    obs = env.reset()
    # Initialize the experience buffer
    buffer = MultiStepExperienceBuffer(buffer_size, extensions_hyp['multi_step'], discount)
    target_model.set_weights(train_model.get_weights())

    ########## EXPLORATION INITIALIZATION ######
    eps = start_explor
    eps_decay = (start_explor - end_explor) / explor_steps

    for ep in range(num_epochs):
        g_rew = 0
        done = False

        # Until the environment does not end..
        while not done:   
            # Epsilon decay
            if eps > end_explor:
                eps -= eps_decay

            # 탐욕정책에 의해 액션을 선택함
            act = eps_greedy(np.squeeze(agent_op(obs)), eps=eps)  

            # execute the action in the environment
            obs2, rew, done, _ = env.step(act)

            # Render the game if you want to
            if render_the_game:
                env.render()

            # Add the transition to the replay buffer
            buffer.add(obs, rew, act, obs2, done)

            obs = obs2
            g_rew += rew
            step_count += 1

            ################ TRAINING ###############
            # If it's time to train the network:
            if len(buffer) > min_buffer_size and (step_count % update_freq == 0):
                
                # sample a minibatch from the buffer
                mb_obs, mb_rew, mb_act, mb_obs2, mb_done = buffer.sample_minibatch(batch_size)

                dqn_variable = train_model.trainable_variables
                with tf.GradientTape() as tape:
                    # automatic differentiation에 대한 연산내용을 기록함
                    tape.watch(dqn_variable)

                    rewards = tf.convert_to_tensor(mb_rew, dtype=tf.float32)
                    actions = tf.convert_to_tensor(mb_act, dtype=tf.int32)
                    dones = tf.convert_to_tensor(mb_done, dtype=tf.float32)

                    # ======================================================================================
                    # DDQN의 경우에는 온라인 모델과 타겟 모델에 대한 Q값을 구하여 
                    # 폴리시를 업데이트하는 로직을 구현해야 함 
                    # 아래 extensions_hyp['DDQN']의 값이 True이면 이에 해당 하는 기능을 수행함
                    # 아닌 경우에는 DQN과 같이 그냥 해당 기능을 수행함

                    if extensions_hyp['DDQN']:
                        next_states = mb_obs2.reshape(batch_size, obs_dim[0], obs_dim[1], obs_dim[2])
                        target_q = target_model(tf.convert_to_tensor(next_states, dtype=tf.float32))
                        online_q = train_model(tf.convert_to_tensor(next_states, dtype=tf.float32))

                        next_action=tf.argmax(online_q, axis=1)
                        target_value = tf.reduce_sum(tf.one_hot(next_action, act_dim) * target_q, axis = 1)
                        # 게임종료시에는 reward를 target_value으로 설정하고
                        # 진행중인 경우에는 gamma * target_value + rewards값으로 설정함

                        target_value = (1-dones) * discount * target_value + rewards
                        states = mb_obs.reshape(batch_size, obs_dim[0], obs_dim[1], obs_dim[2])
                        main_q = train_model(tf.convert_to_tensor(states, dtype=tf.float32))
                        main_value = tf.reduce_sum(tf.one_hot(actions, act_dim) * main_q, axis = 1)
                    else:
                        next_states = mb_obs2.reshape(batch_size, obs_dim[0], obs_dim[1], obs_dim[2])
                        target_q = target_model(tf.convert_to_tensor(next_states, dtype=tf.float32))

                        next_action=tf.argmax(target_q, axis=1)
                        target_value = tf.reduce_sum(tf.one_hot(next_action, act_dim) * target_q, axis = 1)
                        # 게임종료시에는 reward를 target_value으로 설정하고
                        # 진행중인 경우에는 gamma * target_value + rewards값으로 설정함

                        target_value = (1-dones) * discount * target_value + rewards
                        states = mb_obs.reshape(batch_size, obs_dim[0], obs_dim[1], obs_dim[2])
                        main_q = train_model(tf.convert_to_tensor(states, dtype=tf.float32))
                        main_value = tf.reduce_sum(tf.one_hot(actions, act_dim) * main_q, axis = 1)

                    # Loss값 계산 : RMSE(Root Mean Square Error)
                    error = tf.square(main_value - target_value) * 0.5
                    error = tf.reduce_mean(error)

                dqn_grads = tape.gradient(error, dqn_variable)
                opt.apply_gradients(zip(dqn_grads, dqn_variable))

                # Add the train summary to the file_writer
                with file_writer.as_default():
                    tf.summary.scalar('loss', np.mean(error), step_count)  
                    tf.summary.scalar('reward', np.mean(g_rew), step_count)  
                    
            # Every update_target_net steps, update the target network
            if (len(buffer) > min_buffer_size) and (step_count % update_target_net == 0):

                target_model.set_weights(train_model.get_weights())                

                with file_writer.as_default():
                    tf.summary.scalar('loss', np.mean(error), step_count)  
                    tf.summary.scalar('reward', np.mean(g_rew), step_count)  

            # If the environment is ended, reset it and initialize the variables
            if done:
                obs = env.reset()
                batch_rew.append(g_rew)
                g_rew, render_the_game = 0, False

        # every test_frequency episodes, test the agent and write some stats in TensorBoard
        if ep % test_frequency == 0:
            # Test the agent to 10 games
            test_rw = test_agent(env_test, agent_op, num_games=10)

            # Run the test stats and add them to the file_writer
            # 텐서보드에 관련 정보를 출력하기
            with file_writer.as_default():
                tf.summary.scalar('test_rew', np.mean(test_rw), step_count)   

            # Print some useful stats
            ep_sec_time = int((current_milli_time()-ep_time) / 1000)
            print('Ep:%4d Rew:%4.2f, Eps:%2.2f -- Step:%5d -- Test:%4.2f %4.2f -- Time:%d -- Ep_Steps:%d' %
                        (ep,np.mean(batch_rew), eps, step_count, np.mean(test_rw), np.std(test_rw), ep_sec_time, (step_count-old_step_count)/test_frequency))

            ep_time = current_milli_time()
            batch_rew = []
            old_step_count = step_count
                            
        if ep % render_cycle == 0:
            render_the_game = False

    file_writer.close()
    env.close()


if __name__ == '__main__':

    extensions_hyp={
        'DDQN':False,
        'dueling':False,
        'multi_step':1
    }
    DQN_with_variations('PongNoFrameskip-v4', extensions_hyp, hidden_sizes=[128],
        lr=2e-4, buffer_size=100000, 
        update_target_net=1000, batch_size=32, 
        update_freq=2, frames_num=2, 
        min_buffer_size=10000, render_cycle=10000)
