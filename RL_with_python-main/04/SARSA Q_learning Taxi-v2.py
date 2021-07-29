import numpy as np 
import gym

def eps_greedy(Q, s, eps=0.1):
    '''
    Epsilon greedy policy
    '''
    if np.random.uniform(0,1) < eps:
        # 랜덤액션 선택
        return np.random.randint(Q.shape[1])
    else:
        # 탐욕 폴리시의 액션을 선택
        return greedy(Q, s)


def greedy(Q, s):
    '''
    Greedy policy

    return the index corresponding to the maximum action-state value
    '''
    return np.argmax(Q[s])


def run_episodes(env, Q, num_episodes=100, to_print=False):
    '''
    Run some episodes to test the policy
    '''
    tot_rew = []
    state = env.reset()

    for _ in range(num_episodes):
        done = False
        game_rew = 0

        while not done:
            # 탐욕액션을 선택
            next_state, rew, done, _ = env.step(greedy(Q, state))

            state = next_state
            game_rew += rew 
            if done:
                state = env.reset()
                tot_rew.append(game_rew)

    if to_print:
        print('Mean score: %.3f of %i games!'%(np.mean(tot_rew), num_episodes))

    return np.mean(tot_rew)

def Q_learning(env, lr=0.01, num_episodes=10000, eps=0.3, gamma=0.95, eps_decay=0.00005):
    nA = env.action_space.n
    nS = env.observation_space.n

    # Q 매트릭스 초기화
    # Q: nS*nA행렬, 행은 상태를 표시하고, 열은 행동을 표시함
    Q = np.zeros((nS, nA))
    games_reward = []
    test_rewards = []

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        tot_rew = 0
        
        # 임계값이 0.01에 도달할 때까지 엡실론 값을 소멸시킴
        if eps > 0.01:
            eps -= eps_decay

        # 환경이 멈출때까지 메인 부분을 반복 수행함
        while not done:
            # esp-탐욕 폴리시를 따르는 행동을 선택함
            action = eps_greedy(Q, state, eps)

            next_state, rew, done, _ = env.step(action) # 해당 환경에서 1스텝 진행

            # Q러닝을 이용하여 상태-행동 가치(state-action value)를 업데이트
            # (다음 상태에 대해 최대 Q값을 가져옮)
            Q[state][action] = Q[state][action] + lr*(rew + gamma*np.max(Q[next_state]) - Q[state][action])

            state = next_state
            tot_rew += rew
            if done:
                games_reward.append(tot_rew)

        # 300 에피소드마다 폴리시를 테스트하고 결과를 출력함
        if (ep % 300) == 0:
            test_rew = run_episodes(env, Q, 1000)
            print("Episode:{:5d}  Eps:{:2.4f}  Rew:{:2.4f}".format(ep, eps, test_rew))
            test_rewards.append(test_rew)
            
    return Q


def SARSA(env, lr=0.01, num_episodes=10000, eps=0.3, gamma=0.95, eps_decay=0.00005):
    nA = env.action_space.n
    nS = env.observation_space.n

    # Q행렬 초기화
    # Q : 행렬 nS*nA
    # 개별 행은 상태를 표시하고, 개별 열은 다양한 행동을 표시함

    Q = np.zeros((nS, nA))
    games_reward = []
    test_rewards = []

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        tot_rew = 0

        # 0.01값에 도달할 때까지 엡실론을 소멸시킴
        if eps > 0.01:
            eps -= eps_decay


        action = eps_greedy(Q, state, eps) 

        # 환경이 멈출때까지 메인 부분을 반복수행
        while not done:
            next_state, rew, done, _ = env.step(action) # 환경에서 스텝 하나를 취함

            # 다음 행동을 선택(SARSA 업데이트를 위해 필요)
            next_action = eps_greedy(Q, next_state, eps) 
            # SARSA 업데이트
            Q[state][action] = Q[state][action] + lr*(rew + gamma*Q[next_state][next_action] - Q[state][action])

            state = next_state
            action = next_action
            tot_rew += rew
            if done:
                games_reward.append(tot_rew)

        # 매 300에피소드마다 폴리시를 테스트하고 결과를 출력함
        if (ep % 300) == 0:
            test_rew = run_episodes(env, Q, 1000)
            print("Episode:{:5d}  Eps:{:2.4f}  Rew:{:2.4f}".format(ep, eps, test_rew))
            test_rewards.append(test_rew)

    return Q


if __name__ == '__main__':
    env = gym.make('Taxi-v3')
    Q_qlearning = Q_learning(env, lr=.1, num_episodes=5000, eps=0.4, gamma=0.95, eps_decay=0.001)
    Q_sarsa = SARSA(env, lr=.1, num_episodes=5000, eps=0.4, gamma=0.95, eps_decay=0.001)
