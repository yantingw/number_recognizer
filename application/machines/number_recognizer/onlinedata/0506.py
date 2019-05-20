# -*- coding: utf-8 -*-
"""
Created on Mon May  6 11:04:19 2019

@author: yanting
"""


from __future__ import print_function
import keras,random
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import os,re
# 定義梯度下降批量
batch_size = 64
# 定義分類數量
num_classes = 10
# 定義訓練週期
epochs = 10

# 定義圖像寬、高
img_rows, img_cols = 28, 28
input_shape =(1, img_rows, img_cols)

DATA_DIR = os.getcwd()
y = np.load(os.path.join(DATA_DIR,"y_test.npy"))
file_data = np.empty((y.size,28,28))
img_get = list()
for filename in os.listdir(DATA_DIR):

    if(re.match(r'img',filename,)==None):
        continue
    print ("Loading: %s" % filename)
    index =int( filename.split("img")[1].split(".npy")[0])
    loadFile = np.load(filename,'r')
    #loadFile[loadFile > 0] = 1# 二質化
    loadFile = loadFile.reshape(28,28)
    file_data[index] = loadFile
    img_get.append(index)

x_test = np.array( file_data).reshape(y.size,28,28,1)
y_test = keras.utils.to_categorical(y, num_classes)
#np.histogram(y,bins=['0','1','2','3','4','5','6','7','8','9']) #make sure the data detail

#%% change the number of data
#y[-1] =2
#np.save("y_test.npy",y)
#y_new =y[:114]
#np.save("y_test.npy",y_new)
#%% data expand
import cv2 , random
modified_data =[]
modified_y =[]
for i in range (y.size):
    img = x_test[i]
    rows, cols, channel = img.shape
    
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), random.randint(-35,35), random.uniform(0.7,1.1))
    dst = cv2.warpAffine(img, M, (cols, rows))
    
   # curr_img   = np.reshape(dst, (28, 28)) # 28 by 28 matrix 
    #curr_label =y[i,]
    #plt.matshow(curr_img, cmap=plt.get_cmap('gray'))
    #plt.title("" + str(i + 1) + "th Training Data " 
     #         + "Label is " + str(curr_label))
    modified_data.append(dst)
    modified_y.append(y[i])

#%%combine the whole data
y_try = np.append(y,np.array(modified_y))
modified_data = np.array(modified_data).reshape(y.size,28,28,1)
x_try = np.append(x_test,modified_data,axis = 0)
splite_position=int( x_try.shape[0]*0.1)
x_test = x_try[:splite_position,]
x_train = x_try[splite_position:,]
y_try = keras.utils.to_categorical(y_try, num_classes)
y_test = y_try[:splite_position,]
y_train = y_try[splite_position:,]
#%%
y_try = np.append(y,np.array(modified_y))
modified_data = np.array(modified_data).reshape(y.size,28,28,1)
x_try = np.append(x_test,modified_data,axis = 0)
splite_position=int( x_try.shape[0]*0.1)
x_test = x_try
y_try = keras.utils.to_categorical(y_try, num_classes)
y_test = y_try
#%%print try
import random
#for i in [ random.randint(0,y.size)]:
for i in [-1]:
    curr_img   = np.reshape(x_test[i, :], (28, 28)) # 28 by 28 matrix 
    #curr_label = np.argmax(y_test[i,] ) # Label
    curr_label =y[i]
    plt.matshow(curr_img, cmap=plt.get_cmap('gray'))
    plt.title("" + str(i) + "th Training Data " 
              + "Label is " + str(curr_label))
    

#%%test the current model

model = keras.models.load_model(os.path.join(DATA_DIR,"my_model0506.h5"))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#print('test before save: ', model.predict(x_test[0:2]))

#%%train data
(xt, yt), (ax, ay) = mnist.load_data()
xt = np.append(xt,ax,axis=0)
yt = np.append(yt,ay,axis=0)

if K.image_data_format() == 'channels_first':
    xt = x_train.reshape(xt.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else: # channels_last: 色彩通道(R/G/B)資料(深度)放在第4維度，第2、3維度放置寬與高
    xt = xt.reshape(xt.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# 轉換色彩 0~255 資料為 0~1
xt = xt.astype('float64')
xt /= 255

yt = keras.utils.to_categorical(yt, num_classes)

x_train = np.row_stack((x_train,xt))
y_train = np.row_stack((y_train,yt))


#%%model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(BatchNormalization)
# 建立卷積層，filter=64,即 output size, Kernal Size: 3x3, activation function 採用 relu
model.add(Conv2D(64, (3, 3), activation='relu'))
# 建立池化層，池化大小=2x2，取最大值
model.add(MaxPooling2D(pool_size=(2, 2)))
# Dropout層隨機斷開輸入神經元，用於防止過度擬合，斷開比例:0.25
#model.add(Dropout(0.25))
model.add(BatchNormalization)
# 建立卷積層，filter=64,即 output size, Kernal Size: 3x3, activation function 採用 relu
model.add(Conv2D(128, (3, 3), activation='relu'))
# 建立池化層，池化大小=2x2，取最大值
model.add(MaxPooling2D(pool_size=(2, 2)))
# Flatten層把多維的輸入一維化，常用在從卷積層到全連接層的過渡。
model.add(Flatten())
# 全連接層: 128個output
model.add(Dense(256, activation='relu'))
# Dropout層隨機斷開輸入神經元，用於防止過度擬合，斷開比例:0.5
model.add(Dropout(0.25))
# 使用 softmax activation function，將結果分類
model.add(Dense(num_classes, activation='softmax'))

# 編譯: 選擇損失函數、優化方法及成效衡量方式
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# 進行訓練, 訓練過程會存在 train_history 變數中
train_history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

#%%
# 顯示損失函數、訓練成果(分數)
model =keras.models.load_model('my_model0506.h5')
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#print('test before save: ', model.predict(x_test[0:2]))
model.save('my_model0506_2.h5')   # HDF5 file, you have to pip3 install h5py if don't have it
    
    