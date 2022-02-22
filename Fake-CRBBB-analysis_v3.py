#!/usr/bin/env python
# coding: utf-8

# In[1]:

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import numpy as np


import os

import time

def norm(x):
    return (x-min(x))/(max(x)-min(x))
X_CRBBB_fake_encode=np.loadtxt('GAN_DATA_3000(C2CGANV2_2Lead.py)super.csv')
X_CRBBB_fake_encode=X_CRBBB_fake_encode.reshape(10000,200,12)

for i in range(10000):
    
        for x in range(12):
            X_CRBBB_fake_encode[i,:,x]=norm(savgol_filter(X_CRBBB_fake_encode[i,:,x], 7 , 3))
       

X_CRBBB_fake_encode_lstm=np.loadtxt('GAN_DATA_3000_lstm(C2CGANV2_2Lead.py).csv')
X_CRBBB_fake_encode_lstm=X_CRBBB_fake_encode_lstm.reshape(10000,200,12)

for i in range(10000):
    
        for x in range(12):
            X_CRBBB_fake_encode_lstm[i,:,x]=norm(savgol_filter(X_CRBBB_fake_encode_lstm[i,:,x], 7 , 3))



X_CRBBB_real=np.loadtxt('X_CRBBB.csv')
n_samples=X_CRBBB_real.shape[0]
X_CRBBB_real=X_CRBBB_real.reshape(n_samples,200,12)


def norm(x):
    return (x-min(x))/(max(x)-min(x))

for i in range(len(X_CRBBB_real)):
    for j in range(12):
        x=X_CRBBB_real[i,:,j]
        X_CRBBB_real[i,:,j]=norm(x)

X_CRBBB_not=np.loadtxt('X_CRBBB_not.csv')
X_CRBBB_not=X_CRBBB_not.reshape(10000,200,12)

for i in range(len(X_CRBBB_not)):
    for j in range(12):
        x=X_CRBBB_not[i,:,j]
        X_CRBBB_not[i,:,j]=norm(x)

X_CRBBB_fake=np.loadtxt('FAKE_CRBBB_2000.csv')
X_CRBBB_fake=X_CRBBB_fake.reshape(10000,200,12)

      
X_CRBBB_fake_DCGAN=np.loadtxt('GAN_DATA_OLD.csv')
X_CRBBB_fake_DCGAN=X_CRBBB_fake_DCGAN.reshape(10000,200,12)
for i in range(10000):
    
    for x in range(12):
        X_CRBBB_fake_DCGAN[i,:,x]=norm(savgol_filter(X_CRBBB_fake_DCGAN[i,:,x], 7 , 3))




X_all=np.concatenate((X_CRBBB_real, X_CRBBB_not[:n_samples]), axis=0)

r,f=np.ones(n_samples),np.zeros(n_samples)
Y_all=np.concatenate((r,f), axis=0)

'''
X_norm=np.loadtxt('X_norm_12lead.csv').reshape(4000,200,12)
for i in range(4000):
    for j in range(12):
        x=X_norm[i,:,j]
        X_norm[i,:,j]=norm(x)
'''        



        

X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all, test_size=0.25, shuffle=True)


e =75




def define_indp_discriminator():
    model = tf.keras.Sequential()

    model.add(layers.Convolution1D(filters=96, kernel_size=11, strides=4, input_shape=(200,12)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.01))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2, padding='same'))

    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))

    model.add(layers.Convolution1D(filters=256, kernel_size=5, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.01))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2, padding='same'))

    model.add(layers.Convolution1D(filters=384, padding='same', kernel_size=3))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.01))
    
    model.add(layers.Convolution1D(filters=384, kernel_size=3))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.01))
    
    model.add(layers.Convolution1D(filters=256, kernel_size=3))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.01))


    model.add(layers.Convolution1D(filters=128, kernel_size=2, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.01))
    
    model.add(layers.MaxPooling1D(pool_size=2, strides=2, padding='same'))
    model.add(layers.GlobalAveragePooling1D())
    
    model.add(layers.Dense(128))
    model.add(layers.LeakyReLU(alpha=0.01))
    model.add(layers.Dropout(0.1))

    model.add(layers.Dense(128))
    model.add(layers.LeakyReLU(alpha=0.01))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(2, activation='sigmoid'))
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4), metrics=['accuracy'])

    return model



indp_discriminator_model = define_indp_discriminator()
indp_discriminator_model.summary()

t = 0
accuracy_1 = []
accuracy_2 = []
accuracy_3 = []
accuracy_4 = []
accuracy_5 = []
val = []
f12, f13, f14, f15  = [] , [], [], []
auc_score2, auc_score3, auc_score4, auc_score5 = [] , [], [], []
while t < 6000:
  t +=100  
  counter=0
  fake_list=[]
  fake_list_encode_ = []
  fake_list_encode = []
  fake_list_DC = []
  fake_list_DC_ = []
  fake_list_lstm = []
  fake_list_lstm_ = []
  
  for i in range(t):
    fake_list.append(i)
    fake_list_encode.append(i)
    fake_list_DC.append(i)
    fake_list_lstm.append(i)
  
  n_fakes= t
  n_encode_fakes = t
  n_DC_fakes = t
  n_lstm = t
  X_CRBBB_fake_validated_encode = X_CRBBB_fake_encode[fake_list_encode]
  X_CRBBB_fake_validated_DC = X_CRBBB_fake_DCGAN[fake_list_DC]
  X_CRBBB_fake_validated = X_CRBBB_fake[fake_list]
  X_CRBBB_fake_validated_lstm =X_CRBBB_fake_encode_lstm[fake_list_lstm]
  
  n_total_encode =  n_samples+n_encode_fakes
  n_total = n_samples+n_fakes
  n_total_DC = n_samples + n_DC_fakes
  n_total_lstm = n_samples + n_lstm
  print('#########################################', n_total, ' - Total Values used in this epoch #############################################')

  X=np.concatenate((X_CRBBB_real, X_CRBBB_fake_validated, X_CRBBB_not[:n_total]), axis=0)
  
  r,f=np.ones(n_total),np.zeros(n_total)
  Y=np.concatenate((r,f), axis=0)
  
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, shuffle=True)
  
  indp_discriminator_model_2 = define_indp_discriminator()
  y_pred = []
  m2=indp_discriminator_model_2.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=e)
  y_pred_ = indp_discriminator_model_2.predict(X_test)
  for i in range(len(y_pred_)):
        if y_pred_[i][1] < y_pred_[i][0]:
            y_pred.append(0)
        else:
            y_pred.append(1)
  f12.append(f1_score(Y_test, y_pred))
  y_prob = y_pred_[:, 1]
  auc_score2.append(roc_auc_score(Y_test, y_prob))
  
  
  
  
  X_encode=np.concatenate((X_CRBBB_real, X_CRBBB_fake_validated_encode, X_CRBBB_not[:n_total_encode]), axis=0)
  
  r_encode,f_encode=np.ones(n_total_encode),np.zeros(n_total_encode)
  Y_encode=np.concatenate((r_encode,f_encode), axis=0)
  
  X_train, X_test, Y_train, Y_test = train_test_split(X_encode, Y_encode, test_size=0.25, shuffle=True)
  
  indp_discriminator_model_3 = define_indp_discriminator()
  y_pred = []
  m3=indp_discriminator_model_3.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=e)
  y_pred_ = indp_discriminator_model_3.predict(X_test)
  for i in range(len(y_pred_)):
        if y_pred_[i][1] < y_pred_[i][0]:
            y_pred.append(0)
        else:
            y_pred.append(1)
  f13.append(f1_score(Y_test, y_pred))
  y_prob = y_pred_[:, 1]
  auc_score3.append(roc_auc_score(Y_test, y_prob))
    
  
  X_DC=np.concatenate((X_CRBBB_real, X_CRBBB_fake_validated_DC, X_CRBBB_not[:n_total_DC]), axis=0)
  
  r_DC,f_DC=np.ones(n_total_DC),np.zeros(n_total_DC)
  Y_DC=np.concatenate((r_DC,f_DC), axis=0)
  
  X_train, X_test, Y_train, Y_test = train_test_split(X_DC, Y_DC, test_size=0.25, shuffle=True)
  
  indp_discriminator_model_4 = define_indp_discriminator()
  y_pred = []
  m4=indp_discriminator_model_4.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=75)
  y_pred_ = indp_discriminator_model_4.predict(X_test)
  for i in range(len(y_pred_)):
        if y_pred_[i][1] < y_pred_[i][0]:
            y_pred.append(0)
        else:
            y_pred.append(1)
  f14.append(f1_score(Y_test, y_pred))
  y_prob = y_pred_[:, 1]
  auc_score4.append(roc_auc_score(Y_test, y_prob))
    
  
  X_lstm=np.concatenate((X_CRBBB_real, X_CRBBB_fake_validated_lstm , X_CRBBB_not[:n_total_lstm]), axis=0)
  
  r_lstm,f_lstm=np.ones(n_total_lstm),np.zeros(n_total_lstm)
  Y_lstm=np.concatenate((r_lstm,f_lstm), axis=0)
  
  X_train, X_test, Y_train, Y_test = train_test_split(X_lstm, Y_lstm, test_size=0.25, shuffle=True)
  
  indp_discriminator_model_5 = define_indp_discriminator()
  y_pred = []
  m5=indp_discriminator_model_5.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=e)
  y_pred_ = indp_discriminator_model_5.predict(X_test)
  for i in range(len(y_pred_)):
        if y_pred_[i][1] < y_pred_[i][0]:
            y_pred.append(0)
        else:
            y_pred.append(1)
  f15.append(f1_score(Y_test, y_pred))
  y_prob = y_pred_[:, 1]
  auc_score5.append(roc_auc_score(Y_test, y_prob))
    
  

  accuracy_2.append(m2.history['val_accuracy'][-1])
  accuracy_3.append(m3.history['val_accuracy'][-1])
  accuracy_4.append(m4.history['val_accuracy'][-1])
  accuracy_5.append(m5.history['val_accuracy'][-1])
  val.append(t)
  

plt.figure()

plt.plot(val, accuracy_2, color ='blue', label = 'Real + LSTM')
plt.plot(val, accuracy_3, color ='green', label = 'Real + encoder - decoder')
plt.plot(val, accuracy_4, color ='black', label = 'Real + DCGAN')
plt.plot(val, accuracy_5, color ='red', label = 'Real + lstm encoder - decoder')
plt.legend(loc='lower right')
plt.grid()
plt.xlabel('Number of samples')
plt.ylabel('Accuracy')
plt.title('Binary classification problem')
plt.savefig('accuracy_val')
plt.close()


np.savetxt('Accuracy_on_val.csv', accuracy_1)
np.savetxt('Accuracy_LSTM_val.csv', accuracy_2)
np.savetxt('Accuracy_encoder_val.csv', accuracy_3)
np.savetxt('Accuracy_DCGAN_val.csv', accuracy_4)
np.savetxt('Accuracy_encoder_valm.csv', accuracy_5)


np.savetxt('f1_lstm.csv', f12)
np.savetxt('f1_encoder.csv', f13)
np.savetxt('f1_dcgan.csv', f14)
np.savetxt('f1_encoder_lstm.csv', f15)

np.savetxt('auc_lstm.csv', auc_score2)
np.savetxt('auc_encoder.csv', auc_score3)
np.savetxt('auc_dcgan.csv', auc_score4)
np.savetxt('auc_encoder_lstm.csv', auc_score5)

plt.plot(val, f12, color ='blue', label = 'Real + LSTM')
plt.plot(val, f13, color ='green', label = 'Real + encoder - decoder')
plt.plot(val, f14, color ='black', label = 'Real + DCGAN')
plt.plot(val, f15, color ='red', label = 'Real + lstm encoder - decoder')
plt.legend(loc='lower right')
plt.grid()
plt.xlabel('Number of samples')
plt.ylabel('Accuracy')
plt.title('Binary classification problem')
plt.savefig('f1')
plt.close()

plt.plot(val, auc_score2, color ='blue', label = 'Real + LSTM')
plt.plot(val, auc_score3, color ='green', label = 'Real + encoder - decoder')
plt.plot(val, auc_score4, color ='black', label = 'Real + DCGAN')
plt.plot(val, auc_score5, color ='red', label = 'Real + lstm encoder - decoder')
plt.legend(loc='lower right')
plt.grid()
plt.xlabel('Number of samples')
plt.ylabel('Accuracy')
plt.title('Binary classification problem')
plt.savefig('auc_score')
plt.close()

'''




X3 = X=np.concatenate((X_CRBBB_real[:n_fakes], X_CRBBB_fake_validated), axis=0)
r,f=np.ones(n_fakes),np.zeros(n_fakes)
Y3=np.concatenate((r,f), axis=0)

X_train, X_test, Y_train, Y_test = train_test_split(X3, Y3, test_size=0.25, shuffle=True)

indp_discriminator_model_4 = define_indp_discriminator()

m4=indp_discriminator_model_4.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=75, verbose = '1')

'''









