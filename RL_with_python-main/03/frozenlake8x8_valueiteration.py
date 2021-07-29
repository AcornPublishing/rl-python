import numpy as np
import gym

def eval_state_action(V, s, a, gamma=0.99):
    return np.sum([p * (rew + gamma*V[next_s]) for p, next_s, rew, _ in env.P[s][a]])

def value_iteration(eps=0.0001):
    '''
    Value iteration algorithm
    '''
    V = np.zeros(nS)
    it = 0

    while True:
        delta = 0
        # max 연산자로
        # 각 상태의 가치를 업데이트 함
        # update the value of each state using as "policy" the max operator
        for s in range(nS):
            old_v = V[s]
            V[s] = np.max([eval_state_action(V, s, a) for a in range(nA)])
            delta = max(delta, np.abs(old_v - V[s]))

        if delta < eps:
            break
        else:
            print('Iter:', it, ' delta:', np.round(delta, 5))
        it += 1

    return V

def run_episodes(env, V, num_games=100):
    '''
    Run some test games
    '''
    tot_rew = 0
    state = env.reset()

    for _ in range(num_games):
        done = False
        while not done:
            action = np.argmax([eval_state_action(V, state, a) for a in range(nA)])
            next_state, reward, done, _ = env.step(action)

            state = next_state
            tot_rew += reward 
            if done:
                state = env.reset()

    print('Won %i of %i games!'%(tot_rew, num_games))

            
if __name__ == '__main__':
    # 환경을 생성함
    env = gym.make('FrozenLake-v0')
    # 추가 정보를 갖도록 하기 위해 래핑함
    env = env.unwrapped

    # 공간차원을 설정
    nA = env.action_space.n
    nS = env.observation_space.n

    # 가치 이터레이션
    V = value_iteration(eps=0.0001)
    # 100회의 게임 에피소드에 대해 가치함수를 테스트
    run_episodes(env, V, 100)
    # 상태 가치를 출력
    print(V.reshape((4,4)))

