# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:28:46 2019

@author: yanting
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:53:27 2019

@author: yanting
"""


#from __future__ import print_function
import keras
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
batch_size = 32
# 定義分類數量
num_classes = 10
# 定義訓練週期
epochs = 10
duplicate_data = 10
# 定義圖像寬、高
img_rows, img_cols = 28, 28
input_shape =(1, img_rows, img_cols)

DATA_DIR = os.getcwd()
y = np.load(os.path.join(DATA_DIR,"y_test.npy"))
file_data = np.empty((y.size,28,28))
for filename in os.listdir(DATA_DIR):

    if(re.match(r'img',filename,)==None):
        continue
    print ("Loading: %s" % filename)
    index =int( filename.split("img")[1].split(".npy")[0])
    loadFile = np.load(filename,'r')
    #loadFile[loadFile > 0] = 1# 二質化
    loadFile = loadFile.reshape(28,28)
    file_data[index] = loadFile
x_test = np.array( file_data).reshape(y.size,28,28,1)
#np.histogram(y,bins=['0','1','2','3','4','5','6','7','8','9']) #make sure the data detail
#%%increase the image made by the net
modified_data =[]
modified_y =[]
import cv2,random

for i in range (y.size):
    img = x_test[i]   
    time = 0
    rows, cols, channel = img.shape
       # curr_img   = np.reshape(dst, (28, 28)) # 28 by 28 matrix 
    #curr_label =y[i,]
    #plt.matshow(curr_img, cmap=plt.get_cmap('gray'))
    #plt.title("" + str(i + 1) + "th Training Data " 
     #         + "Label is " + str(curr_label))
    while(time <duplicate_data): 
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), random.randint(-30,30), random.uniform(0.7,1.1))
        M = np.float32([[1,0,random.randint(-8,8)],[0,1,random.randint(-8,8)]])
        dst = cv2.warpAffine(img, M, (cols, rows))
    
        modified_data.append(dst)
        modified_y.append(y[i])
        time+=1
        
y_try = np.append(y,np.array(modified_y))
modified_data = np.array(modified_data).reshape(y.size*duplicate_data,28,28,1)
x_try = np.append(x_test,modified_data,axis = 0)
splite_position=int( x_try.shape[0]*0.1)
x_test = x_try
y_try = y_try

#%%print model result
model =keras.models.load_model('pre_model0513.h5')
 # Get rid of the classification layer
model.summary()
#model.save('pre_model0513.h5')

idx = random.randint(0,x_try.shape[0])
img = x_try[idx].reshape(1,28,28,1)
pre = model.predict(img)
print(pre)
import random
#for i in [ random.randint(0,y.size)]:
curr_img   = np.reshape(img, (28, 28)) # 28 by 28 matrix 
#curr_label = np.argmax(y_test[i,] ) # Label
curr_label =y_try[idx]
plt.matshow(curr_img, cmap=plt.get_cmap('gray'))
plt.title("" + str(i) + "th Training Data " 
          + "Label is " + str(curr_label))
x_test  = list()
y_test  = list()

for (idx,img) in zip(y_try, x_try): 
    pre = model.predict(img.reshape(1,28,28,1))
    x_test.append(pre)
    y_test.append(idx)

x_test = np.array(x_test).reshape(len(x_test),pre.size)
#y_test = keras.utils.to_categorical(y_test, num_classes)
y_test = np.array(y_test)
#%%
from sklearn import svm
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x_test, y_test, test_size = 0.20)

# kernel = 'rbf'
clf_rbf = svm.SVC(kernel='rbf')
clf_rbf.fit(xtrain,ytrain)
score_rbf = clf_rbf.score(xtest,ytest)
print("The score of rbf is : %f"%score_rbf)

# kernel = 'linear'
clf_linear = svm.SVC(kernel='linear')
clf_linear.fit(xtrain,ytrain)
score_linear = clf_linear.score(xtest,ytest)
print("The score of linear is : %f"%score_linear)

# kernel = 'poly'
clf_poly = svm.SVC(kernel='poly')
clf_poly.fit(xtrain,ytrain)
score_poly = clf_poly.score(xtest,ytest)
print("The score of poly is : %f"%score_poly)
#%%
from sklearn.externals import joblib
joblib.dump(clf_linear, 'clf_linear.pkl')

#读取Model
clf3 = joblib.load('clf_linear.pkl')

#测试读取后的Model
print(clf3.predict(pre))


#%%
