# 2장 코드
#%%===== 예제1
import gym

# 환경을 생성하고 초기화
env = gym.make('CartPole-v1')
env.reset()

# 게임을 10회 실행
for i in range(10):
    # 랜덤액션을 선택
    env.step(env.action_space.sample())
    # 게임을 렌더링
    env.render()

# 환경을 닫기
env.close()


#%%===== 예제2
import gym

# 환경을 생성하고 초기화
env = gym.make('CartPole-v1')
env.reset()

# 게임을 10회 실행
for i in range(10):
    # 변수를 초기화
    done = False
    game_rew = 0

    while not done:
        # 랜덤액션을 선택
        action = env.action_space.sample()
        # 환경내에서 한개의 스텝을 취함
        new_obs, rew, done, info = env.step(action)
        game_rew += rew
    
        # 실행완료시 게임 누적보상을 출력하고 환경을 재설정
        if done:
            print('Episode %d finished, reward:%d' % (i, game_rew))
            env.reset()


#%%===== 예제3
import gym

env = gym.make('CartPole-v1')
print(env.observation_space)

print(env.action_space)

print(env.action_space.sample())
print(env.action_space.sample())
print(env.action_space.sample())

print(env.observation_space.low)
print(env.observation_space.high)


#%%===== 예제4
import tensorflow as tf
import timeit

cell = tf.keras.layers.LSTMCell(100)

@tf.function
def fn(input, state):
    return cell(input, state)

input = tf.zeros([100, 100])
state = [tf.zeros([100, 100])] * 2

cell(input, state)
fn(input, state)

graph_time = timeit.timeit(lambda: cell(input, state), number = 100)
auto_graph_time = timeit.timeit(lambda: fn(input, state), number = 100)
print('graph_time:', graph_time)
print('auto_graph_time:', auto_graph_time)



#%%===== 예제5
import tensorflow as tf

# 두 상수 a와 b를 생성
a = tf.constant(4)
b = tf.constant(3)

# 연산을 실행
c = a + b

# 데코레이터 설정
@tf.function
def return_c(x):
    return x

# 합을 계산후 출력
print(return_c(c))


#%%===== 예제6
# 상수
a = tf.constant(1)
print(a.shape)

# 5개 요소의 행렬
b = tf.constant([1,2,3,4,5])
print(b.shape)

a = tf.constant([1,2,3,4,5])
first_three_elem = a[:3]
fourth_elem = a[3]

@tf.function
def return_var(x):
    return(x)

print(return_var(first_three_elem))
print(return_var(fourth_elem))

a = tf.constant([1.0, 1.1, 2.1, 3.1], dtype=tf.float32, name='a_const')
print(a)


#%%===== 예제7
import tensorflow as tf
import numpy as np

# 랜덤변수는 tf.random.uniform을 사용하여 생성
# 0에서 1사이의 범위를 갖는 랜덤변수를 3개 생성후 변수var에 설정
var = tf.random.uniform([1,3], 0, 1, seed=0)

# 상수값으로 초기화한 변수int_val에 설정
int_val = np.array([4, 5])
var2 = tf.Variable(int_val, dtype=tf.int32, name='second_variable')


#%%===== 예제8
import tensorflow as tf
import numpy as np

tf.compat.v1.reset_default_graph()
tf.keras.backend.clear_session()

const1 = tf.constant(3.0, name='constant1')
var1 = tf.Variable(tf.ones([1,2]), dtype=tf.float32, name='variable1')
var2 = tf.Variable(tf.ones([1,2]), dtype=tf.float32, name='variable2', trainable=False)

@tf.function
def my_func(c1, x1, y1):
    op1 = c1 * x1
    op2 = op1 + y1
    return(tf.reduce_mean(op2))

# 연산그래프를 저장할 폴더를 설정
logdir = 'd:\\tmp'
writer = tf.summary.create_file_writer(logdir)
tf.summary.trace_on(graph=True, profiler=True)

op3 = my_func(const1, var1, var2)

with writer.as_default():
    tf.summary.trace_export(name='my_func', step=0, profiler_outdir=logdir)


# %%===== 예제9
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 기존에 생성된 Graph를 모두 삭제하고 Reset 시켜 중복되는 것을 방지한다.
tf.keras.backend.clear_session()

# 랜덤시드를 설정한다.
np.random.seed(10)
tf.random.set_seed(10)

# 모델 클래스를 생성
class Model(object):
  def __init__(self):
    # 변수를 (5.0, 0.0)으로 초기화
    self.W = tf.Variable(5.0)
    self.b = tf.Variable(0.0)

  def __call__(self, x):
    return self.W * x + self.b

model = Model()

# 손실함수를 정의
def loss(predicted_y, desired_y):
  return tf.reduce_mean(tf.square(predicted_y - desired_y))

# True Weight와 Bias를 설정
TRUE_W = 0.5
TRUE_b = 1.4
NUM_EXAMPLES = 100

# 랜덤데이터를 생성하여 Inputs와 Noise로 설정한 후 Outputs값을 생성함
inputs  = tf.random.normal(shape=[NUM_EXAMPLES])
noise   = tf.random.normal(shape=[NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise

# 생성한 데이터 분포를 표시함(scatter plot)
plt.scatter(inputs, outputs, c = 'b')
plt.scatter(inputs, model(inputs), c = 'r')
plt.show()

# 현재 손실 표시
print('현재 손실: '),
print(loss(model(inputs), outputs).numpy())

# 훈련 루프 정의
def train(model, inputs, outputs, learning_rate):
  with tf.GradientTape() as t:
    current_loss = loss(model(inputs), outputs)

  dW, db = t.gradient(current_loss, [model.W, model.b])
  model.W.assign_sub(learning_rate * dW)
  model.b.assign_sub(learning_rate * db)

# 텐서보드를 통하여 추적하려는 parameter와 저장폴더 설정
logdir = 'd:/tmp'
writer = tf.summary.create_file_writer(logdir)
tf.summary.trace_on(graph=True, profiler=True)

@tf.function
def my_func(step, tmpval1, tmpval2, tmpval3):
  with writer.as_default():
    tf.summary.scalar('weight', tmpval1, step=step)
    tf.summary.scalar('bias', tmpval2, step=step)
    tf.summary.scalar('loss', tmpval3, step=step)
    tf.summary.histogram('model_weight', tmpval1, step=step)
    tf.summary.histogram('model_bias', tmpval2, step=step)
    tf.summary.histogram('model_loss', tmpval3, step=step)

model = Model()
# 도식화를 위해 W값과 b값의 변화를 저장함
Ws, bs = [], []
epochs = range(100)
for epoch in epochs:
  Ws.append(model.W.numpy())
  bs.append(model.b.numpy())
  current_loss = loss(model(inputs), outputs)

  train(model, inputs, outputs, learning_rate=0.1)
  print('에포크 %2d: W=%1.2f b=%1.2f, 손실=%2.5f' %
        (epoch+1, Ws[-1], bs[-1], current_loss))

  # TensorBoard에서 조회하려는 parameter를 출력/저장
  my_func(epoch, Ws[-1], bs[-1], current_loss) 
  writer.flush()

# 저장된 값들을 도식화
plt.plot(epochs, Ws, 'r',
         epochs, bs, 'b')
plt.plot([TRUE_W] * len(epochs), 'r--',
         [TRUE_b] * len(epochs), 'b--')
plt.legend(['W', 'b', 'true W', 'true_b'])
plt.show()


# %%
