import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten 

def myModel():

    model = Sequential()

    model.add(Flatten())
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(16, activation = 'sigmoid'))
    model.add(Dense(10, activation= "softmax"))

    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

    return model 

