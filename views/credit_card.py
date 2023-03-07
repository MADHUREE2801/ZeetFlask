import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score,f1_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.layers import Conv1D,MaxPool1D
from tensorflow.keras.optimizers import Adam


import pickle                                            #converts data in python file into serialized form

credit_card_data=pd.read_csv('C:/Users/Dell/Documents/Major Project frontend/login_sys/views/creditcard.csv')

#Dropping missing values
credit_card_data=credit_card_data.dropna(how='any')

#Distributing data into legit and fraud
legit= credit_card_data[credit_card_data.Class==0]
fraud=credit_card_data[credit_card_data.Class==1] 

#Under-sampling
legit_sample=legit.sample(n=492)
#Conactenating
new_dataset=pd.concat([legit_sample,fraud],axis=0)

#Splitting the data into features and target
X=new_dataset.drop(columns='Class',axis=1)
Y=new_dataset['Class']

#Split the data into training data and testing data
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

#Random Forest
#rf=RandomForestClassifier()
#rf.fit(X_train.values,Y_train.values)
#Y_train_pred3=rf.predict(X_train.values)
#Y_test_pred3=rf.predict(X_test.values)

#we need to reshape the data from 2D to 3D before passing it as input to CNN
X_train=X_train.to_numpy()
X_test=X_test.to_numpy()
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)

#Build CNN

#By increasing no of epochs and adding pooling layer we are achieving good training accuracy and less validation loss(overcoming overfitting problem )

epochs=40
model=Sequential()

#First Layer
model.add(Conv1D(32,2,activation='relu',input_shape=X_train[0].shape))
model.add(BatchNormalization())
model.add(MaxPool1D(2))
model.add(Dropout(0.2))     #20% of neurons will be dropped after 1st layer

#Second Layer
model.add(Conv1D(64,2,activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool1D(2))

model.add(Dropout(0.5))     #50% of neurons will be dropped after 2nd layer

#Flattening Layer   ----Converting multidimensional data into a vector
model.add(Flatten())

#Dense Layer
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy'])

#Train the model
history=model.fit(X_train,Y_train,epochs=epochs,validation_data=(X_test,Y_test),verbose=1)

Y_train_pred=model.predict(X_train).round()
CNN_train_accuracy=accuracy_score(Y_train,Y_train_pred)
CNN_train_precision=precision_score(Y_train,Y_train_pred)
CNN_train_recall=recall_score(Y_train,Y_train_pred)
CNN_train_f1=f1_score(Y_train,Y_train_pred)



#writing into pickle file-->Dumping ML model into pickle file
pickle.dump(model,open('model.pkl','wb'))

#Loading the pickle file-->reading the pickle file
model=pickle.load(open('model.pkl','rb'))