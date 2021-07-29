import numpy as np 
import tensorflow as tf
from tensorflow.keras import Model, Input, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

import gym
from datetime import datetime
import time
import pybullet_envs

# 신경망 모델링
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

# 소프트맥스 엔트로피
def softmax_entropy(logits):
    '''
    Softmax Entropy
    '''
    return -tf.reduce_sum(tf.nn.softmax(logits, axis=-1) * tf.nn.log_softmax(logits, axis=-1), axis=-1)

# 가우시안 로그우도(likelihood)
def gaussian_log_likelihood(ac, mean, log_std):
    '''
    Gaussian Log Likelihood 
    '''
    log_p = ((ac-mean)**2 / (tf.exp(log_std)**2+1e-9) + 2*log_std) + np.log(2*np.pi)
    return -0.5 * tf.reduce_sum(log_p, axis=-1)

# 컨주게이트 그레디언트
def conjugate_gradient(A, b, x=None, iters=10):
    '''
    Conjugate gradient method: approximate the solution of Ax=b
    It solve Ax=b without forming the full matrix, just compute the matrix-vector product (The Fisher-vector product)
    
    NB: A is not the full matrix but is a useful matrix-vector product between the averaged Fisher information matrix and arbitrary vectors 
    Descibed in Appendix C.1 of the TRPO paper
    '''
    if x is None:
        x = np.zeros_like(b)
        
    r = A(x) - b
    p = -r
    for _ in range(iters):
        a = np.dot(r, r) / (np.dot(p, A(p))+1e-8)
        x += a*p
        r_n = r + a*A(p)
        b = np.dot(r_n, r_n) / (np.dot(r, r)+1e-8)
        p = -r_n + b*p
        r = r_n
    return x

# 가우시안 KL(Kullback Leibler Divergence)
def gaussian_DKL(mu_q, log_std_q, mu_p, log_std_p):
    '''
    Gaussian KL divergence in case of a diagonal covariance matrix
    '''
    return tf.reduce_mean(tf.reduce_sum(0.5 * (log_std_p - log_std_q + tf.exp(log_std_q - log_std_p) + (mu_q - mu_p)**2 / tf.exp(log_std_p) - 1), axis=1))
    # return temp

def backtracking_line_search(DKL, delta, old_loss, p=0.8):
    '''
    Backtracking line searc. It look for a coefficient s.t. the constraint on the DKL is satisfied
    It has both to
     - improve the non-linear objective
     - satisfy the constraint

    '''
    ## Explained in Appendix C of the TRPO paper
    a = 1
    it = 0
 
    new_dkl, new_loss = DKL(a) 
    while (new_dkl > delta) or (new_loss > old_loss):
        a *= p
        it += 1
        new_dkl, new_loss = DKL(a)
    return a

def GAE(rews, v, v_last, gamma=0.99, lam=0.95):
    '''
    Generalized Advantage Estimation
    '''
    # rews와 v의 길이가 같은지를 확인함
    assert len(rews) == len(v)
    vs = np.append(v, v_last)
    d = np.array(rews) + gamma*vs[1:] - vs[:-1]
    gae_advantage = discounted_rewards(d, 0, gamma*lam)
    return gae_advantage

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

def flatten_list(tensor_list):
    '''
    Flatten a list of tensors
    '''
    return tf.concat([flatten(t) for t in tensor_list], axis=0)

def flatten(tensor):
    '''
    Flatten a tensor
    '''
    return tf.reshape(tensor, shape=(-1,))


class Buffer():
    '''
    Class to store the experience from a unique policy
    '''
    def __init__(self, gamma=0.99, lam=0.95):
        self.gamma = gamma
        self.lam = lam
        self.adv = []
        self.ob = []
        self.ac = []
        self.rtg = []

    def store(self, temp_traj, last_sv):
        '''
        Add temp_traj values to the buffers and compute the advantage and reward to go

        Parameters:
        -----------
        temp_traj: list where each element is a list that contains: observation, reward, action, state-value
        last_sv: value of the last state (Used to Bootstrap)
        '''
        # store only if there are temporary trajectories
        if len(temp_traj) > 0:
            self.ob.extend(temp_traj[:,0])
            rtg = discounted_rewards(temp_traj[:,1], last_sv, self.gamma)
            self.adv.extend(GAE(temp_traj[:,1], temp_traj[:,3], last_sv, self.gamma, self.lam))
            self.rtg.extend(rtg)
            self.ac.extend(temp_traj[:,2])

    def get_batch(self):
        # standardize the advantage values
        norm_adv = (self.adv - np.mean(self.adv)) / (np.std(self.adv) + 1e-10)
        return np.array(self.ob), np.array(self.ac), np.array(norm_adv), np.array(self.rtg)

    def __len__(self):
        assert(len(self.adv) == len(self.ob) == len(self.ac) == len(self.rtg))
        return len(self.ob)

class StructEnv(gym.Wrapper):
    '''
    Gym Wrapper to store information like number of steps and total reward of the last espisode.
    '''
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.n_obs = self.env.reset()
        self.total_rew = 0
        self.len_episode = 0

    def reset(self, **kwargs):
        self.n_obs = self.env.reset(**kwargs)
        self.total_rew = 0
        self.len_episode = 0
        return self.n_obs.copy()
        
    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.total_rew += reward
        self.len_episode += 1 
        return ob, reward, done, info

    def get_episode_reward(self):
        return self.total_rew

    def get_episode_length(self):
        return self.len_episode

# TRPO 메소드
def TRPO(env_name, hidden_sizes=[32], cr_lr=5e-3, num_epochs=50, gamma=0.99, lam=0.95, number_envs=1, 
        critic_iter=10, steps_per_env=100, delta=0.002, algorithm='TRPO', conj_iters=10, minibatch_size=1000):
    '''
    Trust Region Policy Optimization

    Parameters:
    -----------
    env_name: Name of the environment
    hidden_sizes: list of the number of hidden units for each layer
    cr_lr: critic learning rate
    num_epochs: number of training epochs
    gamma: discount factor
    lam: lambda parameter for computing the GAE
    number_envs: number of "parallel" synchronous environments
        # NB: it isn't distributed across multiple CPUs
    critic_iter: NUmber of SGD iterations on the critic per epoch
    steps_per_env: number of steps per environment
            # NB: the total number of steps per epoch will be: steps_per_env*number_envs
    delta: Maximum KL divergence between two policies. Scalar value
    algorithm: type of algorithm. Either 'TRPO' or 'NPO'
    conj_iters: number of conjugate gradient iterations
    minibatch_size: Batch size used to train the critic
    '''

    # clears the default graph stack and resets the global default graph
    tf.compat.v1.reset_default_graph()

    # Create a few environments to collect the trajectories
    envs = [StructEnv(gym.make(env_name)) for _ in range(number_envs)]

    low_action_space = envs[0].action_space.low
    high_action_space = envs[0].action_space.high

    obs_dim = envs[0].observation_space.shape
    act_dim = envs[0].action_space.shape[0]

    # Time 출력 ===
    now = datetime.now()
    clock_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, now.second)
    print('Time:', clock_time)
    #  =======

    # 텐서플로에 실행정보 출력하는 코드
    hyp_str = '-steps_{}-crlr_{}'.format(steps_per_env, cr_lr)
    file_writer = tf.summary.create_file_writer('log_dir/{}/TRPO_{}_{}'.format(env_name, clock_time, hyp_str))

    # p_means_model에 대한 입력
    # log_std에 대한 생성
    x=Input([obs_dim[0]])
    p_means_model = mlp(x, hidden_sizes, act_dim, activation=tf.tanh, last_activation=tf.tanh)
    log_std = tf.compat.v1.get_variable(name='log_std', initializer=np.zeros(act_dim, dtype=np.float32)-0.5) 

    # Neural network that represent the value function
    x=Input([obs_dim[0]]) # osb_ph를 x값으로 사용함(s_values : 결과값, s_values_model : 모델)
    s_values_model = mlp(x, hidden_sizes, 1, activation=tf.tanh, last_activation=None)

    # 변수 생성(p_variables)
    p_variables = p_means_model.trainable_variables
    p_variables.append(log_std)
    v_opt = optimizers.Adam(lr=cr_lr) 

    def get_act_smp(p_means):
        # Add "noise" to the predicted mean following the Guassian distribution with standard deviation e^(log_std)
        p_noisy = p_means + tf.random.normal(shape = tf.shape(p_means), mean=0.0, stddev=1.0) * tf.exp(log_std)
        # Clip the noisy actions
        a_sampl = tf.clip_by_value(p_noisy, low_action_space, high_action_space)
        return(a_sampl)

    # p_loss를 계산함
    def get_ploss(old_mu_ph, old_log_std_ph, obs_ph, act_ph, adv_ph, old_p_log_ph, step_count):
        # Compute the gaussian log likelihood
        p_means = p_means_model(tf.convert_to_tensor(obs_ph, tf.dtypes.float32))

        # => clip_by_value포함 여부를 체크해야 함
        # p_means = tf.clip_by_value(p_means, low_action_space, high_action_space)        
        p_log = gaussian_log_likelihood(act_ph, p_means, log_std)        
        # Measure the divergence
        # diverg = tf.reduce_mean(tf.exp(old_p_log_ph - p_log))
        # ratio
        ratio_new_old = tf.exp(p_log - old_p_log_ph)

        # TRPO surrogate loss function
        p_loss = - tf.reduce_mean(ratio_new_old * adv_ph)

        with file_writer.as_default():
            tf.summary.histogram('p_log', p_log, step_count)        
        return(p_loss)



    # p_grads 계산후 flatten을 실행함
    def p_grads_flatten(obs_ph, act_ph, adv_ph, old_p_log_ph, step_count):
        with tf.GradientTape() as tape: 
            # Compute the gaussian log likelihood
            p_means = p_means_model(tf.convert_to_tensor(obs_ph, tf.dtypes.float32))
            p_log = gaussian_log_likelihood(act_ph, p_means, log_std)        
            # ratio
            ratio_new_old = tf.exp(p_log - old_p_log_ph)
            # TRPO surrogate loss function
            p_loss = - tf.reduce_mean(ratio_new_old * adv_ph)
        p_grads = tape.gradient(p_loss, p_variables)
        p_grads_flatten = flatten_list(p_grads)   

        with file_writer.as_default():
            tf.summary.scalar('ratio_new_old', tf.reduce_mean(ratio_new_old).numpy(), step_count)
       
        return(p_grads_flatten) 

    # Actor Parameters를 Restore함
    def get_restore_params(p_old_variables):      
        # variable used as index for restoring the actor's parameters
        it_v1 = tf.Variable(0, trainable=False)
        restore_params = []

        for p_v in p_variables:
            upd_rsh = tf.reshape(p_old_variables[it_v1 : it_v1+tf.reduce_prod(p_v.shape)], shape=p_v.shape)
            restore_params.append(p_v.assign(upd_rsh)) 
            it_v1 = it_v1 + tf.reduce_prod(p_v.shape)

        restore_params = tf.group(*restore_params)

    # dkl_diverg를 계산함
    def get_dkl_diverg(old_mu_ph, old_log_std_ph, obs_batch):
        p_means = p_means_model(tf.convert_to_tensor(obs_batch, tf.dtypes.float32))         
        temp = gaussian_DKL(old_mu_ph, old_log_std_ph, p_means, log_std)
        return(temp)


    # 피셔-벡터 Product를 계산함
    def get_Fx(old_mu_ph, old_log_std_ph, p_ph, obs_ph):
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                # gaussian KL divergence of the two policies 
                dkl_diverg = get_dkl_diverg(old_mu_ph, old_log_std_ph, obs_ph)
            # Jacobian of the KL divergence (Needed for the Fisher matrix-vector product)
            dkl_diverg_grad = t1.gradient(dkl_diverg, p_variables) 
            dkl_matrix_product = tf.reduce_sum(flatten_list(dkl_diverg_grad) * p_ph)
            # Fisher vector product
            # The Fisher-vector product is a way to compute the A matrix without the need of the full A
        Fx = flatten_list(t2.gradient(dkl_matrix_product, p_variables))
        return(Fx)

    # Policy를 업데이트함
    def do_p_opt(beta_ph, alpha, cg_ph):
        # Apply the updates to the policy
        npg_update = beta_ph * cg_ph
        alpha = tf.Variable(1., trainable=False)
        trpo_update = alpha * npg_update

        ####################   POLICY UPDATE  ###################
        # variable used as an index
        it_v = tf.Variable(0, trainable=False)
        p_opt = []

        for p_v in p_variables:
            upd_rsh = tf.reshape(trpo_update[it_v : it_v+tf.reduce_prod(p_v.shape)], shape=p_v.shape)
            p_opt.append(p_v.assign_sub(upd_rsh))
            it_v = it_v + tf.reduce_prod(p_v.shape)

        # p_opt = tf.group(*p_opt)
        # return(p_opt)


    # 시간을 출력함
    # Time
    now = datetime.now()
    clock_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, now.second)
    print('Time:', clock_time)

    hyp_str = '-spe_'+str(steps_per_env)+'-envs_'+str(number_envs)+'-cr_lr'+str(cr_lr)+'-crit_it_'+str(critic_iter)+'-delta_'+str(delta)+'-conj_iters_'+str(conj_iters)
    file_writer = tf.summary.create_file_writer('log_dir/'+env_name+'/'+algorithm+'_'+clock_time+'_'+hyp_str, tf.get_default_graph())

    # variable to store the total number of steps
    step_count = 0
    print('Env batch size:',steps_per_env, ' Batch size:',steps_per_env*number_envs)

    for ep in range(num_epochs):
        # Create the buffer that will contain the trajectories (full or partial) 
        # run with the last policy
        buffer = Buffer(gamma, lam)
        # lists to store rewards and length of the trajectories completed
        batch_rew = []
        batch_len = []

        # Execute in serial the environment, storing temporarily the trajectories.
        for env in envs:
            temp_buf = []

            # iterate over a fixed number of steps
            for _ in range(steps_per_env):
                # run the policy
                tmp_obs_ph = env.n_obs.reshape(-1, len(env.n_obs))
                act = np.squeeze(get_act_smp(p_means_model(tmp_obs_ph)))
                val = tf.squeeze(s_values_model(tmp_obs_ph)).numpy()

                # take a step in the  environment
                obs2, rew, done, _ = env.step(act)

                # add the new transition to the temporary buffer
                temp_buf.append([env.n_obs.copy(), rew, act, np.squeeze(val)])
                env.n_obs = obs2.copy()
                step_count += 1

                if done:
                    # Store the full trajectory in the buffer 
                    # (the value of the last state is 0 as the trajectory is completed)
                    buffer.store(np.array(temp_buf), 0)
                    # Empty temporary buffer
                    temp_buf = []

                    batch_rew.append(env.get_episode_reward())
                    batch_len.append(env.get_episode_length())

                    env.reset()

            # Bootstrap with the estimated state value of the next state!     
            temp = env.n_obs.reshape(-1, len(env.n_obs))
            lsv = tf.squeeze(s_values_model(temp))
            buffer.store(np.array(temp_buf), np.squeeze(lsv))

        # Get the entire batch from the buffer
        # NB: all the batch is used and deleted after the optimization. This is because PPO is on-policy
        obs_batch, act_batch, adv_batch, rtg_batch = buffer.get_batch()
        # log probabilities, logits and log std of the "old" policy
        # "old" policy refer to the policy to optimize and that has been used to sample from the environment
        old_p_means = p_means_model(tf.convert_to_tensor(obs_batch, tf.dtypes.float32))
        old_log_std = log_std
        old_p_log = gaussian_log_likelihood(act_batch, old_p_means, old_log_std)            

        # Gather and flatten the actor parameters   
        old_actor_params = flatten_list(p_variables)
        old_p_loss = get_ploss(old_p_means, old_log_std, obs_batch, act_batch, adv_batch, old_p_log, step_count)

        def H_f(p):
            '''
            Run the Fisher-Vector product on 'p' to approximate the Hessian of the DKL
            '''
            temp = get_Fx(old_p_means, old_log_std, p, obs_batch)
            return(temp)

        # p_var_flatten
        g_f = p_grads_flatten(obs_batch, act_batch, adv_batch, old_p_log, step_count)
        # Compute the Conjugate Gradient so to obtain an approximation of H^(-1)*g
        # Where H in reality isn't the true Hessian of the KL divergence but an approximation of it computed via Fisher-Vector Product (F)
        conj_grad = conjugate_gradient(H_f, g_f, iters=conj_iters)
        # Compute the step length
        beta_np = np.sqrt(2*delta / np.sum(conj_grad * H_f(conj_grad)))


        def DKL(alpha_v):
            '''
            Compute the KL divergence.
            It optimize the function to compute the DKL. Afterwards it restore the old parameters.
            '''
            # dkl_diverg
            temp1 = get_dkl_diverg(old_p_means, old_log_std, obs_batch)   
            # p_loss
            temp2 = get_ploss(old_p_means, old_log_std, obs_batch, act_batch, adv_batch, old_p_log, step_count)        
            a_res = [temp1, temp2]
            get_restore_params(old_actor_params)   

            with file_writer.as_default():
                tf.summary.scalar('dkl_diverg', np.mean(temp1), step_count)
                tf.summary.scalar('p_loss', np.mean(temp2), step_count)
            return a_res

        # Actor optimization step
        # Different for TRPO or NPG
        if algorithm=='TRPO':
            # Backtracing line search to find the maximum alpha coefficient s.t. the constraint is valid
            best_alpha = backtracking_line_search(DKL, delta, old_p_loss, p=0.8)
            do_p_opt(beta_np, best_alpha, conj_grad)
        elif algorithm=='NPG':
            # In case of NPG, no line search
            best_alpha = 1
            do_p_opt(beta_np, 1, conj_grad)

        lb = len(buffer)
        shuffled_batch = np.arange(lb)
        np.random.shuffle(shuffled_batch)

        with file_writer.as_default():
            # Value function optimization steps
            for _ in range(critic_iter):
                # shuffle the batch on every iteration
                np.random.shuffle(shuffled_batch)
                for idx in range(0,lb, minibatch_size):
                    minib = shuffled_batch[idx:min(idx+minibatch_size,lb)]

                    with tf.GradientTape() as tape:  
                        tmp_s_values = s_values_model(tf.convert_to_tensor(obs_batch[minib], tf.dtypes.float32))
                        tmp_s_values = tf.squeeze(tmp_s_values)

                        # MSE loss function        
                        v_loss = tf.reduce_mean((rtg_batch[minib] - tmp_s_values)**2)
                        # Critic optimization

                    grads = tape.gradient(v_loss, s_values_model.trainable_variables)
                    # Processing aggregated gradients.      
                    v_opt.apply_gradients(zip(grads, s_values_model.trainable_variables))

            tf.summary.scalar('alpha', np.mean(best_alpha), step_count) 
            tf.summary.scalar('beta', np.mean(beta_np), step_count) 
            tf.summary.scalar('old_p_loss', np.mean(old_p_loss), step_count)
            tf.summary.scalar('s_values_mn', tf.reduce_mean(tmp_s_values).numpy(), step_count)
            tf.summary.scalar('old_v_loss', np.mean(v_loss), step_count)                 
            tf.summary.scalar('p_std_mn', tf.reduce_mean(tf.exp(log_std)), step_count)
            tf.summary.scalar('v_loss', np.mean(v_loss), step_count)

            tf.summary.histogram('p_means', old_p_means, step_count)
            tf.summary.histogram('s_values', tmp_s_values.numpy(), step_count)
            tf.summary.histogram('adv_ph',adv_batch, step_count)
            tf.summary.histogram('log_std',log_std,step_count)

            # print some statistics and run the summary for visualizing it on TB
            if len(batch_rew) > 0:
                    tf.summary.scalar('supplementary/performance', np.mean(batch_rew), step_count)  
                    tf.summary.scalar('supplementary/len', np.mean(batch_len), step_count)    
                    file_writer.flush()             
                print('Ep:%d Rew:%.2f -- Step:%d' % (ep, np.mean(batch_rew), step_count))

            # closing environments..
            for env in envs:
                env.close()

            file_writer.close()

#'RoboschoolWalker2d-v1'
if __name__ == '__main__':
    TRPO('Walker2DBulletEnv-v0', hidden_sizes=[64,64], cr_lr=2e-3, gamma=0.99, lam=0.95, num_epochs=1000, steps_per_env=6000, 
         number_envs=1, critic_iter=10, delta=0.01, algorithm='TRPO', conj_iters=10, minibatch_size=1000)
