import numpy as np
import gym

def eval_state_action(V, s, a, gamma=0.99):
    return np.sum([p * (rew + gamma*V[next_s]) for p, next_s, rew, _ in env.P[s][a]])

def policy_evaluation(V, policy, eps=0.0001):
    '''
    Policy evaluation. Update the value function until it reach a steady state
    '''
    while True:
        delta = 0
        # 모든 상태에 대해 반복 Loop
        for s in range(nS):
            old_v = V[s]
            # 벨만방정식을 이용하여 V[s]를 업데이트
            V[s] = eval_state_action(V, s, policy[s])
            delta = max(delta, np.abs(old_v - V[s]))

        if delta < eps:
            break

def policy_improvement(V, policy):
    '''
    Policy improvement. Update the policy based on the value function
    '''
    policy_stable = True
    for s in range(nS):
        old_a = policy[s]
        # 최고의 상태가치를 제공하는 행동으로 폴리시를 업데이트함
        policy[s] = np.argmax([eval_state_action(V, s, a) for a in range(nA)])
        if old_a != policy[s]: 
            policy_stable = False

    return policy_stable


def run_episodes(env, policy, num_games=100):
    '''
    Run some games to test a policy
    '''
    tot_rew = 0
    state = env.reset()

    for _ in range(num_games):
        done = False
        while not done:
            # 폴리시에 따라 행동을 선택함
            next_state, reward, done, _ = env.step(policy[state])
                
            state = next_state
            tot_rew += reward 
            if done:
                state = env.reset()

    print('Won %i of %i games!'%(tot_rew, num_games))

            
if __name__ == '__main__':
    # 환경을 생성함
    env = gym.make('FrozenLake-v0')
    # 추가 정보를 갖도록 래핑을 수행함
    env = env.unwrapped

    # 공간차원 설정
    nA = env.action_space.n
    nS = env.observation_space.n
    
    # 가치함수와 폴리시를 초기화함
    V = np.zeros(nS)
    policy = np.zeros(nS)

    # 유용한 변수 설정
    policy_stable = False
    it = 0

    while not policy_stable:
        policy_evaluation(V, policy)
        policy_stable = policy_improvement(V, policy)
        it += 1

    print('Converged after %i policy iterations'%(it))
    run_episodes(env, policy)
    print(V.reshape((4,4)))
    print(policy.reshape((4,4)))
