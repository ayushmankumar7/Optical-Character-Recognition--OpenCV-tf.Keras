import cv2 
import numpy as np 
from tensorflow.keras.models import load_model
def show():
        
    img = cv2.imread("test.jpeg")
    grey = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    model  = load_model('mnist.h5')
    print(model.summary())
    preprocessed_digits = []

    for c in contours:
        x,y,w,h = cv2.boundingRect(c)


        img = cv2.rectangle(img, (x,y), (x+w, y+h), color = (0,255,0), thickness = 2)
        digit = thresh[y:y+h, x:x+w]

        resized_digit = cv2.resize(digit, (18,18))

        padded_digit = np.pad(resized_digit, ((5,5), (5,5)), "constant", constant_values = 0)

        preprocessed_digits.append(padded_digit)


    for digit in preprocessed_digits:
        prediction = model.predict(digit.reshape(1,28,28,1))

        print(np.argmax(prediction))


    cv2.imshow('img', img)
    cv2.imshow('thresh', thresh)
    cv2.waitKey(0)

    cv2.destroyAllWindows()