import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
import random
from sklearn.neighbors import NearestNeighbors
from smote import *

np.set_printoptions(threshold=np.inf)

# 加载自制数据集（不平衡）
x_train_savepath = 'D:/学习习习习习/class4/class4/MNIST_FC/mnist_image_label/x_train_imbalance.npy'
y_train_savepath = 'D:/学习习习习习/class4/class4/MNIST_FC/mnist_image_label/y_train_imbalance.npy'
x_test_savepath = 'D:/学习习习习习/class4/class4/MNIST_FC/mnist_image_label/mnist_x_test.npy'
y_test_savepath = 'D:/学习习习习习/class4/class4/MNIST_FC/mnist_image_label/mnist_y_test.npy'
x_train = np.load(x_train_savepath)
y_train = np.load(y_train_savepath)

x_train_list = list(x_train.reshape((len(x_train),784)))
y_train_list = list(y_train)

x_test = np.load(x_test_savepath)
x_test = np.reshape(x_test, (len(x_test), 28, 28))
y_test = np.load(y_test_savepath)


#smote


x_0, y_0, x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4, \
    x_5, y_5, x_6, y_6, x_7, y_7, x_8, y_8, x_9, y_9= [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

#不同标签值的数据放入相应的列表
for i in range(len(x_train_list)):
    if int(y_train_list[i]) == 0:
        x_0.append(x_train_list[i])
        y_0.append(y_train_list[i])
    elif int(y_train_list[i]) == 1:
        x_1.append(x_train_list[i])
        y_1.append(y_train_list[i])
    elif int(y_train_list[i]) == 2:
        x_2.append(x_train_list[i])
        y_2.append(y_train_list[i])
    elif int(y_train_list[i]) == 3:
        x_3.append(x_train_list[i])
        y_3.append(y_train_list[i])
    elif int(y_train_list[i]) == 4:
        x_4.append(x_train_list[i])
        y_4.append(y_train_list[i])
    elif int(y_train_list[i]) == 5:
        x_5.append(x_train_list[i])
        y_5.append(y_train_list[i])
    elif int(y_train_list[i]) == 6:
        x_6.append(x_train_list[i])
        y_6.append(y_train_list[i])
    elif int(y_train_list[i]) == 7:
        x_7.append(x_train_list[i])
        y_7.append(y_train_list[i])
    elif int(y_train_list[i]) == 8:
        x_8.append(x_train_list[i])
        y_8.append(y_train_list[i])
    elif int(y_train_list[i]) == 9:
        x_9.append(x_train_list[i])
        y_9.append(y_train_list[i])

x_data_list = [x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9]
y_data_list = [y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9]
i = 0
for x in x_data_list:

    if len(x) <=5000:
        #print(np.array(x).shape)
        s =Smote(np.array(x), int(6000/len(x))*100)
        if i==0:
            print('原样本点', np.array(x[:1]).reshape((1,28,28)))

        x_new = s.over_sampling()
        #print(len(x_new))
        x_new = list(x_new)
        if i == 0:
            print('新样本点', np.array(x_new[:1]).reshape((1,28,28)))
        x_data_list[i] = x_new
        y_new_num = len(x_new)-len(x)
        y_data_list[i] = y_data_list[i] + [i] * y_new_num
    i=i+1
x_train_smote = []
y_train_smote = []
for x in x_data_list:
    x_train_smote = x + x_train_smote
    #print(len(x))
for y in y_data_list:
    y_train_smote = y + y_train_smote

x_train = np.reshape(x_train_smote, (len(x_train_smote), 28, 28))
y_train = np.array(y_train_smote)

#加载数据集
#mnist = tf.keras.datasets.mnist
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train, x_test = x_train/255.0, x_test/255.0
y_train, y_test = y_train.astype(np.int64), y_test.astype(np.int64)
print(y_train.shape, x_train.shape)
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

class Baseline(Model):
    def __init__(self):
        super(Baseline, self).__init__()
        self.c1 = Conv2D(filters=6, kernel_size=(5, 5), padding='same')  # 卷积层
        self.b1 = BatchNormalization()  # BN层
        self.a1 = Activation('relu')  # 激活层
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        self.d1 = Dropout(0.2)  # dropout层

        self.flatten = Flatten()
        self.f1 = Dense(128, activation='relu')
        self.d2 = Dropout(0.2)
        self.f2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d2(x)
        y = self.f2(x)
        return y
model = Baseline()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)
#model.summary()
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

