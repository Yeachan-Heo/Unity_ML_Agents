# 라이브러리 불러오기
import tensorflow as tf
import numpy as np
import datetime
import os

# 파라미터 설정하기
algorithm = 'CNN'

img_size   = 28
data_size  = img_size**2

num_label = 10

batch_size = 256
num_epoch  = 50


learning_rate = 0.00025

date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")

save_path = "../saved_models/" + date_time + "_" + algorithm
load_path = f"../saved_models/20190312-11-12-35_CNN/model/model"

load_model = False

# MNIST 데이터셋 불러오기 ((x_train, y_train), (x_test, y_test))
mnist = tf.keras.datasets.mnist.load_data(path='mnist.npz')

x_train = mnist[0][0]
y_train = mnist[0][1]
x_test  = mnist[1][0]
y_test  = mnist[1][1]

x_test  = np.reshape(x_test, [-1, img_size, img_size, 1])

# 실제값을 onehot으로 변환
y_train_onehot = np.zeros([y_train.shape[0], num_label])
y_test_onehot  = np.zeros([y_test.shape[0], num_label])

for i in range(y_train.shape[0]):
    y_train_onehot[i, y_train[i]] = 1

for i in range(y_test.shape[0]):
    y_test_onehot[i, y_test[i]] = 1

# Validation Set 생성
x_train, x_val = x_train[:len(x_train)*9//10], x_train[len(x_train)*9//10:]
y_train_onehot, y_val_onehot = y_train_onehot[:len(y_train)*9//10], y_train_onehot[len(y_train)*9//10:]

x_val = np.reshape(x_val, [-1, img_size, img_size, 1])

# 네트워크 구조 정의, 손실 함수 정의 및 학습 수행 
class Model(tf.keras.models.Model):
    def __init__(self):

        # 입력 및 실제값 
        self.x_input  = tf.keras.layers.Input(shape = [img_size, img_size, 1])
        self.x_normalize = (self.x_input - (255.0/2)) / (255.0/2)

        # 네트워크 (Conv-> 2, 은닉층 -> 1)
        self.conv1 = tf.keras.layers.Conv2D(filters=32, activation=tf.nn.relu,
                                      kernel_size=[3,3], strides=[2,2], padding="same")(self.x_normalize)
        self.conv2 = tf.keras.layers.Conv2D(filters=64, activation=tf.nn.relu,
                                      kernel_size=[3,3], strides=[2,2], padding="same")(self.conv1)

        self.flat = tf.keras.layers.Flatten()(self.conv2)

        self.fc1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(self.flat)
        self.out = tf.keras.layers.Dense(num_label)(self.fc1)

        super().__init__(inputs=self.x_input, outputs=self.out)
        # 옵티마이
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    @tf.function
    def loss(self, prediction, y_target):
        return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_target, prediction))

    @tf.function
    def UpdateModel(self, x_input, y_target):
        with tf.GradientTape() as tape:
            prediction = self(x_input)
            loss = self.loss(prediction, y_target)
        grad = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.trainable_weights))
        return loss, prediction

# CNN 학습을 위한 다양한 함수들 
class CNN():
    def __init__(self):
        
        self.model = Model()

        (self.Train_Summary, self.Val_Summary, self.Train_metric_loss,
        self.Val_metric_loss, self.Train_metric_acc, self.Val_metric_acc) = self.Make_Summary()

        # 모델 불러오기
        if (load_model and os.path.exists(load_path)):
            self.model.load_weights(load_path)

    # 모델 학습
    def train_model(self, data_x, data_y, batch_idx):
        len_data = data_x.shape[0]

        if batch_idx + batch_size < len_data:
            batch_x = data_x[batch_idx : batch_idx + batch_size, :, :, :]
            batch_y = data_y[batch_idx : batch_idx + batch_size, :]
        else:
            batch_x = data_x[batch_idx : len_data, :, :, :]
            batch_y = data_y[batch_idx : len_data, :]

        batch_x = batch_x.astype(np.float32)
        batch_y = batch_y.astype(np.float32)

        loss, output = self.model.UpdateModel(batch_x, batch_y)

        accuracy = self.get_accuracy(output, batch_y)

        return loss, accuracy


    # 알고리즘 성능 테스트
    def test_model(self, data_x, data_y):
        data_x, data_y = data_x.astype(np.float32), data_y.astype(np.float32)
        output = self.model(data_x)

        loss = self.model.loss(output, data_y)
        
        accuracy = self.get_accuracy(output, data_y)

        return loss, accuracy

    # 정확도 계산
    def get_accuracy(self, pred, label):
        num_correct = 0.0
        for i in range(label.shape[0]):
            if np.argmax(label[i,:]) == np.argmax(pred[i,:]):
                num_correct += 1.0

        accuracy = num_correct / label.shape[0]

        return accuracy

    # 모델 저장
    def save_model(self):
        self.model.save_weights(save_path + "/model/model")

    # 텐서보드에 손실 함수값 및 정확도 저장
    def Make_Summary(self):
        Train_Summary = tf.summary.create_file_writer(logdir=save_path + "/train")
        Val_Summary = tf.summary.create_file_writer(logdir=save_path + "/val")
        Train_metric_loss = tf.keras.metrics.Mean("Train_loss", dtype=tf.float32)
        Val_metric_loss = tf.keras.metrics.Mean("Val_loss", dtype=tf.float32)
        Train_metric_acc = tf.keras.metrics.Mean("Train_acc")
        Val_metric_acc = tf.keras.metrics.Mean("Val_acc")
        return Train_Summary, Val_Summary, \
               Train_metric_loss, Val_metric_loss, \
               Train_metric_acc, Val_metric_acc

    def Write_Summray(self, accuracy, loss, accuracy_val, loss_val, batch):
        self.Train_metric_loss(loss)
        self.Train_metric_acc(accuracy)
        self.Val_metric_loss(loss_val)
        self.Val_metric_acc(accuracy_val)
        with self.Train_Summary.as_default():
            tf.summary.scalar("loss", self.Train_metric_loss.result(), batch)
            tf.summary.scalar("accuracy", self.Train_metric_acc.result(), batch)
        with self.Val_Summary.as_default():
            tf.summary.scalar("loss", self.Val_metric_loss.result(), batch)
            tf.summary.scalar("accuracy", self.Val_metric_acc.result(), batch)


if __name__ == '__main__':

    cnn = CNN()

    data_train = np.zeros([len(x_train), data_size + num_label])
    data_train[:, :data_size] = np.reshape(x_train, [-1, data_size])
    data_train[:, data_size:] = y_train_onehot 


    batch_num = 0

    loss_list = []
    acc_list  = []
    loss_val_list = []
    acc_val_list = []

    # 학습 수행 
    for epoch in range(num_epoch):

        # 데이터를 섞은 후 입력과 실제값 분리
        np.random.shuffle(data_train)

        train_x = data_train[:, :data_size]
        train_x = np.reshape(train_x, [-1, img_size, img_size, 1])

        train_y = data_train[:, data_size:]

        # 학습 수행, 손실 함수 값 계산 및 텐서보드에 값 저장
        for batch_idx in range(0, x_train.shape[0], batch_size):
            loss, accuracy = cnn.train_model(train_x, train_y, batch_idx)
            loss_val, accuracy_val = cnn.test_model(x_val, y_val_onehot)

            loss_list.append(loss)
            acc_list.append(accuracy)
            loss_val_list.append(loss_val)
            acc_val_list.append(accuracy_val)

            cnn.Write_Summray(accuracy, loss, accuracy_val, loss_val, batch_num)

            batch_num += 1

        # 학습 진행 상황 출력 
        print("Epoch: {} / Loss: {:.5f} / Val Loss: {:.5f} / Acc: {:.5f} / Val Acc: {:.5f}"
              .format(epoch+1, np.mean(loss_list), np.mean(loss_val_list), np.mean(acc_list), 
                     np.mean(acc_val_list)))

        loss_list = []
        acc_list  = []
        loss_val_list = []
        acc_val_list = []

    # 테스트 수행 
    loss_test, accuracy_test = cnn.test_model(x_test, y_test_onehot)
    print('----------------------------------')
    print('Test Accuracy: {:.3f}'.format(accuracy_test))
    print('Test Loss: {:.5f}'.format(loss_test))

    # 모델 저장
    cnn.save_model()
    print("Model is saved in {}".format(save_path + "/model/model"))
