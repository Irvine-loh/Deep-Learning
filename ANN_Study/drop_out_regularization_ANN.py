import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

df=pd.read_csv('C:/Users/User/Desktop/TensorFlow/sonar_dataset.csv',header=None)

# print(df.head())
# print(df.shape)
# print(df.isnull().sum())
# print(df[60].value_counts())
x=df.drop([60],axis=1)
y=df[60]

y=pd.get_dummies(y,drop_first=True) # R= 1 and M = 0
# print(y.sample(5))
# print (y.value_counts())
X_train,x_test,Y_train,y_test=train_test_split(x,y, test_size=0.25 ,random_state=1)
print(f'X_train {X_train.shape}, x_test : {x_test.shape}')

### First Training model before dropout applies ###
model=keras.Sequential([
    keras.layers.Dense(60, input_dim=60 , activation='relu'),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
value=model.fit(X_train,Y_train,epochs=100,batch_size=8)


test_acc=model.evaluate(x_test,y_test)
# print('test model accuracy :',test_acc)

y_pred=model.predict(x_test).reshape(-1)
print(y_pred)

y_pred=np.round(y_pred)
print('predicted test data :',y_pred)
print('Observed test data :',y_test)

print(classification_report(y_test,y_pred))

### 2nd training model include dropout ###
model_1=keras.Sequential([
    keras.layers.Dense(60,input_dim=60,activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid'),
])

model_1.compile(loss='binary_crossentropy',optimizer='adam',metrics='accuracy')
model_1.fit(X_train,Y_train,epochs=100,batch_size=8)

test_acc_2=model_1.evaluate(x_test,y_test)
y_pred_2=model_1.predict(x_test)
y_pred_2=np.round(y_pred_2)
print('2nd model predict result : ',y_pred_2)
print('2nd model observed result : ',y_test)
print(classification_report(y_test,y_pred_2))


