# 라이브러리 불러오기
import tensorflow as tf
import numpy as np
import datetime
import os

# 파라미터 설정하기
algorithm = 'ANN'

data_size = 13

load_model = False

batch_size = 32
num_epoch = 500

learning_rate = 1e-3

date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")

save_path = "../saved_models/" + date_time + "_" + algorithm
load_path = "../saved_models/20190312_11_12_35_ANN/model/model " 

# boston_housing 데이터셋 불러오기 ((x_train, y_train), (x_test, y_test))
boston_housing = tf.keras.datasets.boston_housing.load_data(path='boston_housing.npz')

x_train = boston_housing[0][0]
y_train = np.reshape(boston_housing[0][1],(-1,1))

x_train, x_valid = x_train[:len(x_train)*8//10], x_train[len(x_train)*8//10:]
y_train, y_valid = y_train[:len(y_train)*8//10], y_train[len(y_train)*8//10:]

x_test = boston_housing[1][0]
y_test = np.reshape(boston_housing[1][1],(-1,1))

# 네트워크 구조 정의, 손실 함수 정의 및 학습 수행 
class Model(tf.keras.models.Model):
    def __init__(self):
        # 네트워크
        self.x_input = tf.keras.layers.Input(shape=[data_size, ])
        self.fc1 = tf.keras.layers.Dense(128, activation="relu")(self.x_input)
        self.fc2 = tf.keras.layers.Dense(128, activation="relu")(self.fc1)
        self.fc3 = tf.keras.layers.Dense(128, activation="relu")(self.fc2)
        self.out = tf.keras.layers.Dense(1)(self.fc3)
        super().__init__(inputs=self.x_input, outputs=self.out)
        # 옵티마이
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    #손실 함
    @tf.function
    def loss(self, x_input, y_target):
        return tf.losses.mean_squared_error(y_target, self(x_input))
    
    @tf.function
    def UpdateModel(self, x_input, y_target):
        with tf.GradientTape() as tape:
            loss = self.loss(x_input, y_target)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return loss

# 인공 신경망 학습을 위한 다양한 함수들 
class ANN():
    def __init__(self):
        self.model = Model()

        self.Train_Summary, self.Val_Summary, self.Train_metric, self.Val_metric = self.Make_Summary()

        # 모델 불러오기
        if (load_model == True and os.path.exists(save_path + "/model/model.h5")):
            self.model.load_weights(save_path + "/model/model.h5")

    # 모델 학습
    def train_model(self, data_x, data_y, batch_idx):
        len_data = data_x.shape[0]

        if batch_idx + batch_size < len_data:
            batch_x = data_x[batch_idx : batch_idx + batch_size, :]
            batch_y = data_y[batch_idx : batch_idx + batch_size, :]
        else:
            batch_x = data_x[batch_idx : len_data, :]
            batch_y = data_y[batch_idx : len_data, :]

        loss = self.model.UpdateModel(batch_x, batch_y)
        return loss.numpy().mean()

    # 알고리즘 성능 테스트
    def test_model(self, data_x, data_y):
        loss = self.model.loss(data_x, data_y)
        return loss.numpy().mean()

    # 모델 저장
    def save_model(self):
        self.model.save(save_path + "/model/model.h5")

    # 텐서보드에 손실 함수값 및 정확도 저장
    def Make_Summary(self):
        Train_Summary = tf.summary.create_file_writer(logdir=save_path+"/train")
        Val_Summary = tf.summary.create_file_writer(logdir=save_path+"/val")
        Train_metric = tf.keras.metrics.Mean("Train_loss", dtype=tf.float32)
        Val_metric = tf.keras.metrics.Mean("Val_loss", dtype=tf.float32)
        return Train_Summary, Val_Summary, Train_metric, Val_metric

    def Write_Summray(self, train_loss, val_loss, batch):
        self.Train_metric(train_loss)
        self.Val_metric(val_loss)

        with self.Train_Summary.as_default():
            tf.summary.scalar("loss", self.Train_metric.result(), step=batch)
        with self.Val_Summary.as_default():
            tf.summary.scalar("loss", self.Val_metric.result(), step=batch)

if __name__ == '__main__':
    ann = ANN()
    data_train = np.zeros([x_train.shape[0], data_size + 1])
    data_train[:, :data_size] = x_train
    data_train[:, data_size:] = y_train

    # 학습 수행 
    for epoch in range(num_epoch):
        
        train_loss_list = []
        val_loss_list = []

        # 데이터를 섞은 후 입력과 실제값 분리
        np.random.shuffle(data_train)
        train_x = data_train[:, :data_size]
        train_y = data_train[:, data_size:]

        # 학습 수행, 손실 함수 값 계산 및 텐서보드에 값 저장
        for batch_idx in range(0, x_train.shape[0], batch_size):
            train_loss = ann.train_model(train_x, train_y, batch_idx)
            val_loss = ann.test_model(x_valid, y_valid)
            
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            
        # 학습 진행 상황 출력 
        print("Epoch: {} / Train loss: {:.5f} / Val loss: {:.5f} "
                .format(epoch+1, np.mean(train_loss_list), np.mean(val_loss_list)))
        ann.Write_Summray(np.mean(train_loss_list), np.mean(val_loss_list), epoch)

    # 테스트 수행 
    test_loss = ann.test_model(x_test, y_test)
    print('----------------------------------')
    print('Test Loss: {:.5f}'.format(test_loss))
    # 모델 저장

    ann.save_model()
    print("Model is saved in {}".format(save_path + "/model/model"))

