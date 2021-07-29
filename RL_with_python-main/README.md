# Reinforcement Learning Algorithms with Python

이 코드는 Reinforcement Learning Algorithms with Python 책에 나오는 코드를
TF2.0으로 변경하여 작성한 내용입니다.
현재 편집진행중에 있으며 번역본은 출간예정인 내용입니다.
[Reinforcement Learning Algorithms with Python](https://github.com/sabumjung/RL_with_python/)

## What is this book about?

강화학습(RL)은 인공지능의 인기있고 유망한 분야로  변경되는 요구사항에 대응하여 이상적인 행동을 자동으로 결정하는 에이전트와 스마트한 모델을 만드는 알고리즘이다.
이 책은 여러분이 강화학습 알고리즘을 마스터하고 자가학습(self-learning)하는 에이전트를 개발할 때 구현방법을 이해할 수 있도록 도와준다.

강화학습에서 작동하는데 필요한 툴, 라이브러리, 설정사항에 대한 소개를 시작으로 하여
이 책은 강화학습의 빌딩블록을 다루며 Q-러닝과 SARSA알고리즘과 같은 가치기반 방법에 대한 내용을 상세히 다룬다.

여러분은 복잡한 문제를 해결하기 위해 Q러닝과 신경망의 조합을 이용하는 방법을 배우게 될 것이다. 
게다가 성능과 안정성을 향상시키기 위해 DDPC와 TD3와 같은 결정적인 알고리즘을 학습하기 전에 폴리시 그레디언트 메소드, TRPO, PPO를 학습하게 된다.

또한, 이 책은 모방학습 기술이 작동하는 방법과 Dagger가 에이전트가 작동하도록 가르치는 방법을 다룬다. 
여러분은 진화학습 전략과 블랙-박스 최적화 기술에 대해서도 배우게 될 것이다. 
마지막으로 여러분은 UCB와 UCB1과 같은 탐색적 접근을 학습하고, ESBAS와 같은 메타-알고리즘을 학습한다.

이 책의 마지막 챕터에서 여러분은 
주요 강화학습 알고리즘을 현실에 적용할 때 발생하는 애로사항을 극복하기 위한 방법을 학습하게 될 것이다.


*여러분이 배우게 될 것
    1) OpenAI Gym인터페이스를 이용하여 CartPole게임을 하는 에이전트 개발방법
    2) 모델기반 강화학습 패러다임
    3) 동적프로그래밍으로 Frozen Lake 문제를 해결하는 방법
    4) Q러닝과 SARAS를 이용하여 택시게임을 하는 방법
    5) 딥-Q러닝(DQN)을 이용하여 Atari 게임을 하는 방법
    6) Actor-Critic와 REINFORCE를 이용하여 폴리시 그레디언트 알고리즘을 학습하는 방법
    7) PPO와 TRPO를 연속형 로코모션 환경에 사용하는 방법
    8) 달착륙 문제를 해결하는데 진화전략을 사용하는 방법


## Instructions and Navigations
All of the code is organized into folders. For example, Chapter02.

The code will look like the following:
```
import gym

# create the environment 
env = gym.make("CartPole-v1")
# reset the environment before starting
env.reset()

# loop 10 times
for i in range(10):
    # take a random action
    env.step(env.action_space.sample())
    # render the game
   env.render()

# close the environment
env.close()
```

**Following is what you need for this book:**
If you are an AI researcher, deep learning user, or anyone who wants to learn reinforcement learning from scratch, this book is for you. You’ll also find this reinforcement learning book useful if you want to learn about the advancements in the field. Working knowledge of Python is necessary.	


With the following software and hardware list you can run all code files present in the book (Chapter 1-11).
### Software and Hardware List
| Chapter | Software required | OS required |
| -------- | ------------------------------------ | ----------------------------------- |
| All | Python 3.6 or higher | Windows, Mac OS X, and Linux (Any) |
| All | TensorFlow 2.x or higher | Windows, Mac OS X, and Linux (Any) |


### Related products
* Hands-On Reinforcement Learning with Python [[Packt]](https://www.packtpub.com/big-data-and-business-intelligence/hands-reinforcement-learning-python) [[Amazon]](https://www.amazon.com/Hands-Reinforcement-Learning-Python-reinforcement-ebook/dp/B079Q3WLM4/)

* Python Reinforcement Learning Projects [[Packt]](https://www.packtpub.com/big-data-and-business-intelligence/python-reinforcement-learning-projects) [[Amazon]](https://www.amazon.com/Python-Reinforcement-Learning-Projects-hands-ebook/dp/B07F2S82W3/)


## 본 원서의 저자에 대한 소개
**Andrea Lonza** is a deep learning engineer with a great passion for artificial intelligence and a desire to create machines that act intelligently. He has acquired expert knowledge in reinforcement learning, natural language processing, and computer vision through academic and industrial machine learning projects. He has also participated in several Kaggle competitions, achieving high results. He is always looking for compelling challenges and loves to prove himself.

