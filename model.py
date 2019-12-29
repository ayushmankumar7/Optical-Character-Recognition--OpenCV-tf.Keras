import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten,Conv2D, MaxPooling2D, Dropout, Input 
import data_preprocess

_,_,_,_,inpx = data_preprocess.data_preprocessed()
def myModel():
    global inpx
    inpx = Input(shape=inpx) 
    layer1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(inpx) 
    layer2 = Conv2D(64, (3, 3), activation='relu')(layer1) 
    layer3 = MaxPooling2D(pool_size=(3, 3))(layer2) 
    layer4 = Dropout(0.5)(layer3) 
    layer5 = Flatten()(layer4) 
    layer6 = Dense(250, activation='sigmoid')(layer5) 
    layer7 = Dense(10, activation='softmax')(layer6) 
    model = Model([inpx], layer7) 
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

    return model 

