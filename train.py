import tensorflow as tf 
import model
import data_preprocess

def fit_model():

    X_train, y_train, X_test, y_test = data_preprocess.data_preprocessed()

    mo = model.myModel()

    history = mo.fit(X_train, y_train, epochs = 10)
    mo.save('mnist.h5')
    return history

