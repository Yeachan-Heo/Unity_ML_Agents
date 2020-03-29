# 라이브러리 불러오기
import numpy as np
import random
import datetime
import time
import tensorflow as tf
from collections import deque
from gym_unity.envs import UnityEnvironment

# DQN을 위한 파라미터 값 세팅 
state_size = [84, 84, 3]
action_size = 5 

load_model = False
train_mode = True

batch_size = 32
mem_maxlen = 50000
discount_factor = 0.9
learning_rate = 0.00025

run_episode = 25000
test_episode = 1000

start_train_episode = 1000

target_update_step = 10000
print_interval = 100
save_interval = 5000

epsilon_init = 1.0
epsilon_min = 0.1

# 소코반 환경 설정 (게임판 크기=5, 초록색 +의 수=1, 박스의 수=1)
env_config = {"gridSize": 5, "numGoals": 1, "numBoxes": 1}

date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")

# 유니티 환경 경로 
game = "Sokoban"
env_name = "../env/" + game + "/Mac/" + game

# 모델 저장 및 불러오기 경로
save_path = "../saved_models/" + game + "/" + date_time + "_DQN"
load_path = "../saved_models/" + game + "/20190828-10-42-45_DQN/model/model"

# Model 클래스 -> 함성곱 신경망 정의 및 손실함수 설정, 네트워크 최적화 알고리즘 결정
class Model(tf.keras.models.Model):
    def __init__(self, model_name):
        self.state_input = tf.keras.layers.Input(shape=[state_size[0], state_size[1],
                                           state_size[2]], dtype=tf.float32)
        # 입력을 -1 ~ 1까지 값을 가지도록 정규화
        self.input_normalize = (self.state_input - (255.0 / 2)) / (255.0 / 2)

        # CNN Network 구축 -> 3개의 Convolutional layer와 2개의 Fully connected layer
        self.conv1 = tf.keras.layers.Conv2D(filters=32,
                                          activation=tf.nn.relu, kernel_size=[8,8], 
                                          strides=[4,4], padding="SAME")(self.input_normalize)
        self.conv2 = tf.keras.layers.Conv2D(filters=64,
                                          activation=tf.nn.relu, kernel_size=[4,4],
                                          strides=[2,2],padding="SAME")(self.conv1)
        self.conv3 = tf.keras.layers.Conv2D(filters=64,
                                          activation=tf.nn.relu, kernel_size=[3,3],
                                          strides=[1,1],padding="SAME")(self.conv2)
 
        self.flat = tf.keras.layers.Flatten()(self.conv3)

        self.fc1 = tf.keras.layers.Dense(512,activation=tf.nn.relu)(self.flat)
        self.Q_Out = tf.keras.layers.Dense(action_size, activation=None)(self.fc1)
        super().__init__(inputs=self.state_input, outputs=self.Q_Out)

        # 손실함수 값 계산 및 네트워크 학습 수행 
        self.loss = tf.keras.losses.Huber()
        self.optimizer = tf.optimizers.Adam(learning_rate)

    def predict(self,
              x,
              batch_size=None,
              verbose=0,
              steps=None,
              callbacks=None,
              max_queue_size=10,
              workers=1,
              use_multiprocessing=False):
        return tf.argmax(self(x), 1)

    def UpdateModel(self, state_input, target_q):
        with tf.GradientTape() as tape:
            loss = self.loss(target_q, self(state_input))
        grad = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.trainable_weights))
        return loss

# DQNAgent 클래스 -> DQN 알고리즘을 위한 다양한 함수 정의 
class DQNAgent():
    def __init__(self):
        
        # 클래스의 함수들을 위한 값 설정 
        self.model = Model("Q")
        self.target_model = Model("target")

        self.memory = deque(maxlen=mem_maxlen)

        self.epsilon = epsilon_init

        self.Summary, self.metrics_loss, self.metrics_reward = self.Make_Summary()

        self.update_target()

        if load_model == True:
            self.model.load_weights(save_path)

    # Epsilon greedy 기법에 따라 행동 결정
    def get_action(self, state):
        if self.epsilon > np.random.rand():
            # 랜덤하게 행동 결정
            return np.random.randint(0, action_size)
        else:
            # 네트워크 연산에 따라 행동 결정
            predict = self.model.predict(state).numpy()
            return predict.item()

    # 리플레이 메모리에 데이터 추가 (상태, 행동, 보상, 다음 상태, 게임 종료 여부)
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state[0], action, reward, next_state[0], done))

    # 네트워크 모델 저장 
    def save_model(self):
        self.model.save(save_path + "/model/model")

    # 학습 수행 
    def train_model(self, done):
        # Epsilon 값 감소 
        if done:
            if self.epsilon > epsilon_min:
                self.epsilon -= 1 / (run_episode - start_train_episode)

        # 학습을 위한 미니 배치 데이터 샘플링
        mini_batch = random.sample(self.memory, batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for i in range(batch_size):
            states.append(mini_batch[i][0])
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states.append(mini_batch[i][3])
            dones.append(mini_batch[i][4])

        # 타겟값 계산 
        target = self.model(states).numpy()
        target_val = self.target_model(next_states).numpy()

        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + discount_factor * np.amax(target_val[i])

        # 학습 수행 및 손실함수 값 계산 
        loss = self.model.UpdateModel(states, target)
        return loss.numpy().mean()

    # 타겟 네트워크 업데이트 
    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    # 텐서보드에 기록할 값 설정 및 데이터 기록 
    def Make_Summary(self):
        metrics_loss = tf.keras.metrics.Mean("loss", dtype=tf.float32)
        metrics_reward = tf.keras.metrics.Mean("reward", dtype=tf.float32)
        Summary = tf.summary.create_file_writer(save_path)
        return Summary, metrics_loss, metrics_reward
    
    def Write_Summray(self, reward, loss, episode):
        self.metrics_loss(loss)
        self.metrics_reward(reward)
        with self.Summary.as_default():
            tf.summary.scalar("loss", self.metrics_loss.result())
            tf.summary.scalar("reward", self.metrics_reward.result())

# Main 함수 -> 전체적으로 DQN 알고리즘을 진행 
if __name__ == '__main__':
    # 유니티 환경 경로 설정 (file_name)
    env = UnityEnvironment(file_name=env_name)

    # 유니티 브레인 설정 
    default_brain = env.brain_names[0]
    brain = env.brains[default_brain]

    # DQNAgent 클래스를 agent로 정의 
    agent = DQNAgent()

    step = 0
    rewards = []
    losses = []

    # 환경 설정 (env_config)에 따라 유니티 환경 리셋 및 학습 모드 설정  
    env_info = env.reset(train_mode=train_mode, config=env_config)[default_brain]

    # 게임 진행 반복문 
    for episode in range(run_episode + test_episode):
        if episode > run_episode:
            train_mode = False
            env_info = env.reset(train_mode=train_mode)[default_brain]
        
        # 상태, episode_rewards, done 초기화 
        state = np.uint8(255 * np.array(env_info.visual_observations[0]))
        episode_rewards = 0
        done = False

        # 한 에피소드를 진행하는 반복문 
        while not done:
            step += 1

            # 행동 결정 및 유니티 환경에 행동 적용 
            action = agent.get_action(state)
            env_info = env.step(action)[default_brain]

            # 다음 상태, 보상, 게임 종료 정보 취득 
            next_state = np.uint8(255 * np.array(env_info.visual_observations[0]))
            reward = env_info.rewards[0]
            episode_rewards += reward
            done = env_info.local_done[0]

            # 학습 모드인 경우 리플레이 메모리에 데이터 저장 
            if train_mode:
                agent.append_sample(state, action, reward, next_state, done)
            else:
                time.sleep(0.01) 
                agent.epsilon = 0.05

            # 상태 정보 업데이트 
            state = next_state

            if episode > start_train_episode and train_mode:
                # 학습 수행 
                loss = agent.train_model(done)
                losses.append(loss)

                # 타겟 네트워크 업데이트 
                if step % (target_update_step) == 0:
                    agent.update_target()

        rewards.append(episode_rewards)

        # 게임 진행 상황 출력 및 텐서 보드에 보상과 손실함수 값 기록 
        if episode % print_interval == 0 and episode != 0:
            print("step: {} / episode: {} / reward: {:.2f} / loss: {:.4f} / epsilon: {:.3f}".format
                  (step, episode, np.mean(rewards), np.mean(losses), agent.epsilon))
            agent.Write_Summray(np.mean(rewards), np.mean(losses), episode)
            rewards = []
            losses = []

        # 네트워크 모델 저장 
        if episode % save_interval == 0 and episode != 0:
            agent.save_model()
            print("Save Model {}".format(episode))

    env.close()




