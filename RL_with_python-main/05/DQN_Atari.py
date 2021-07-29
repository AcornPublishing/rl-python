# 관련 라이브러리를 가져오기
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

# ExperienceBuffer클래스(리플레이 메모리로 사용)
class ExperienceBuffer():
    '''
    Experience Replay Buffer
    '''
    # deque : 양쪽 끝에서 삽입과 삭제가 모두 가능한 자료 구조의 한 형태
    def __init__(self, buffer_size):
        self.obs_buf = deque(maxlen=buffer_size)
        self.rew_buf = deque(maxlen=buffer_size)
        self.act_buf = deque(maxlen=buffer_size)
        self.obs2_buf = deque(maxlen=buffer_size)
        self.done_buf = deque(maxlen=buffer_size)

    def add(self, obs, rew, act, obs2, done):
        # 신규 전이(transition)데이터를 버퍼에 추가
        self.obs_buf.append(obs)
        self.rew_buf.append(rew)
        self.act_buf.append(act)
        self.obs2_buf.append(obs2)
        self.done_buf.append(done)
    
    def sample_minibatch(self, batch_size):
        # batch_size크기의 미니배치를 샘플링
        mb_indices = np.random.randint(len(self.obs_buf), size=batch_size)
        mb_obs = scale_frames([self.obs_buf[i] for i in mb_indices])
        mb_rew = [self.rew_buf[i] for i in mb_indices]
        mb_act = [self.act_buf[i] for i in mb_indices]
        mb_obs2 = scale_frames([self.obs2_buf[i] for i in mb_indices])
        mb_done = [self.done_buf[i] for i in mb_indices]
        return mb_obs, mb_rew, mb_act, mb_obs2, mb_done

    def __len__(self):
        return len(self.obs_buf)

def greedy(action_values):
    '''
    Greedy policy(탐욕정책)
    '''
    return np.argmax(action_values)

def eps_greedy(action_values, eps=0.1):
    '''
    Eps-greedy policy(엡실론 탐욕정책)
    '''
    if np.random.uniform(0,1) < eps:
        # 유니폼한 랜덤액션을 선택함
        return np.random.randint(len(action_values))
    else:
        # 탐역액션을 선택함
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
            # epsilon을 이용한 랜덤탐색 전략을 수행하며 탐욕정책을 적용함
            # 폴리시에 확률적 데이터를 가미함
            # 초기 epsilon값을 0.05로 설정함
            a = eps_greedy(np.squeeze(agent_op(o)), eps=0.05)
            o, r, d, _ = env_test.step(a)
            game_r += r
        games_r.append(game_r)
    return games_r

def scale_frames(frames):
    '''
    값을 0에서 1사이의 값을 갖도록 정규화
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
def qnet(x, hidden_layers, output_size, activation=tf.nn.relu, last_activation=None):
    '''
    Deep Q network: CNN followed by FNN
    '''
    inputs=x
    x = cnn(inputs)
    x = Flatten()(x) 
    outputs = fnn(x, hidden_layers, output_size, activation, last_activation)
    model = Model(inputs, outputs)
    return model


def DQN(env_name, hidden_sizes=[32], lr=1e-2, num_epochs=2000, buffer_size=100000, discount=0.99, render_cycle=100, update_target_net=1000, 
        batch_size=64, update_freq=4, frames_num=2, min_buffer_size=5000, test_frequency=20, start_explor=1, end_explor=0.1, explor_steps=100000):
    # 훈련(train)과 테스트(test)를 위한 환경 생성
    env = make_env(env_name, frames_num=frames_num, skip_frames=True, noop_num=20)
    env_test = make_env(env_name, frames_num=frames_num, skip_frames=True, noop_num=20)
    # Add a monitor to the test env to store the videos
    env_test = gym.wrappers.Monitor(env_test, "VIDEOS/TEST_VIDEOS"+env_name+str(current_milli_time()),force=True, video_callable=lambda x: x%20==0)

    # TF2.0에 대한 tf.reset_default_graph()
    tf.compat.v1.reset_default_graph()

    # 차원 사이즈 설정
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n 

    # Train(Online)모델 생성 -> Target 모델 복사
    x = Input([obs_dim[0], obs_dim[1], obs_dim[2]])        
    # Train모델 생성
    train_model = qnet(x, hidden_sizes, act_dim) #, lr)    
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
        # temp = train_model.predict([o])
        return temp

    # Time : 시간정보 설정
    now = datetime.now()
    clock_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, int(now.second))
    print('Time:', clock_time)

    # 텐서플로에 실행정보 출력하는 코드
    LOG_DIR = 'log_dir/'+env_name
    hyp_str = "-lr_{}-upTN_{}-upF_{}-frms_{}" .format(lr, update_target_net, update_freq, frames_num)

    # 텐서보드 서머리에 관련 데이터 저장을 위한 File Writer초기화
    file_writer = tf.summary.create_file_writer(LOG_DIR+'/DQN_'+clock_time+'_'+hyp_str)

    # 하이퍼파라미터 초기화
    render_the_game = False
    step_count = 0
    last_update_loss = []
    ep_time = current_milli_time()
    batch_rew = []
    old_step_count = 0
    opt = optimizers.Adam(lr=lr, )

    # 환경 초기화 및 경험버퍼 초기화(buffer_size)
    obs = env.reset()
    buffer = ExperienceBuffer(buffer_size)
    # 온라인 모델의 가중치를 이용하여 타겟모델의 가중치 초기화
    target_model.set_weights(train_model.get_weights())

    ########## EXPLORATION INITIALIZATION ######
    # 엡실론 초기화 및 소멸비율 설정
    eps = start_explor
    eps_decay = (start_explor - end_explor) / explor_steps

    with file_writer.as_default():
        for ep in range(num_epochs):      
            g_rew = 0
            done = False

            # 게임환경이 종료될때까지 진행
            while not done:
                # 엡실론 소멸 정책을 실행
                if eps > end_explor:
                    eps -= eps_decay

                # 탐욕정책에 의해 액션을 선택함
                act = eps_greedy(np.squeeze(agent_op(obs)), eps=eps)            

                # 환경상에서 액션을 실행하고 결과값을 받아옮
                obs2, rew, done, _ = env.step(act)

                # 게임을 화면에 출력함(설정값이 True인 경우)
                if render_the_game:
                    env.render()

                # 확보한 결과 데이터를 버퍼에 추가함
                # obs : 현상태, rew : 리워드, act : 액션, obs2 : 신규상태, done : 종료여부
                buffer.add(obs, rew, act, obs2, done)

                obs = obs2
                g_rew += rew
                step_count += 1

                ################ 훈련(training) ###############
                # 버퍼사이즈 초과 & step_count를 update_freq(4)로 나누어 0이 되는 경우
                # Train Model을 업데이트 함
                if (len(buffer) > min_buffer_size) and (step_count % update_freq == 0):

                    # replay 버퍼에서 미니배치 샘플을 추출 : off폴리시 
                    # mb_obs : state, mb_rew : reward, mb_act : action, mb_obs2 : next state, done : mb_done
                    mb_obs, mb_rew, mb_act, mb_obs2, mb_done = buffer.sample_minibatch(batch_size)
                    # Train대상 variable를 가져옮
                    dqn_variable = train_model.trainable_variables                
                    with tf.GradientTape() as tape:
                        # automatic differentiation에 대한 연산내용을 기록함
                        tape.watch(dqn_variable)

                        rewards = tf.convert_to_tensor(mb_rew, dtype=tf.float32)
                        actions = tf.convert_to_tensor(mb_act, dtype=tf.int32)
                        dones = tf.convert_to_tensor(mb_done, dtype=tf.float32)

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

                    tf.summary.scalar('loss', np.mean(error), step_count)  
                    tf.summary.scalar('reward', np.mean(g_rew), step_count)  
                    file_writer.flush()
                # 버퍼사이즈 초과 & step_count를 update_target_net(1000)으로 나누어 0이 되는 경우
                # Target Model의 가중치를 Train Model의 가중치로 업데이트 함
                if (len(buffer) > min_buffer_size) and (step_count % update_target_net == 0):
                    target_model.set_weights(train_model.get_weights())                

                    tf.summary.scalar('loss', np.mean(error), step_count)  
                    tf.summary.scalar('reward', np.mean(g_rew), step_count)  
                    file_writer.flush()
                # 환경이 종료된 경우, 환경을 초기화(reset)하고 해당 파라미터를 초기화함
                if done:
                    obs = env.reset()
                    batch_rew.append(g_rew)
                    g_rew, render_the_game = 0, False             

            # 테스트 실행 : test_frequency 에피소드별로, 에이전트를 테스트하고 해당 결과를 텐서보드에 출력함
            if ep % test_frequency == 0:
                # Test the agent to 10 games
                test_rw = test_agent(env_test, agent_op, num_games=10)

                # Run the test stats and add them to the file_writer
                # 텐서보드에 관련 정보를 출력하기
                tf.summary.scalar('test_rew', np.mean(test_rw), step_count)            
                file_writer.flush()
                
                # Print some useful stats
                ep_sec_time = int((current_milli_time()-ep_time) / 1000)
                print('Ep:%4d Rew:%4.2f, Eps:%2.2f -- Step:%5d -- Test:%4.2f %4.2f -- Time:%d -- Ep_Steps:%d' %
                            (ep, np.mean(batch_rew), eps, step_count, np.mean(test_rw), np.std(test_rw), ep_sec_time, (step_count-old_step_count)/test_frequency))

                ep_time = current_milli_time()
                batch_rew = []
                old_step_count = step_count

            if ep % render_cycle == 0:
                render_the_game = False 
        env.close()

if __name__ == '__main__':
    DQN('PongNoFrameskip-v4', hidden_sizes=[128], 
        lr=2e-4, buffer_size=100000, 
        update_target_net=1000, batch_size=32, 
        update_freq=2, frames_num=2, 
        min_buffer_size=10000, render_cycle=10000)    
