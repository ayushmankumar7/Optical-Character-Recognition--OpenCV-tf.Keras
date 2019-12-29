import tensorflow as tf 
import tensorflow.keras.backend as k

def data_preprocessed():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    img_rows, img_cols=28, 28
  
    if k.image_data_format() == 'channels_first': 
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols) 
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols) 
        inpx = (1, img_rows, img_cols) 
        
    else: 
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1) 
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1) 
        inpx = (img_rows, img_cols, 1) 
    
    x_train = x_train.astype('float32') 
    x_test = x_test.astype('float32') 
    x_train, x_test = x_train/255.0, x_test/255.0 #normalization  Helps us achive faster training 

    return (x_train, y_train, x_test, y_test, inpx)



