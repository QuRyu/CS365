# CS365 Spring 2019 
# Project 5
# Qingbo Liu, Iris Lian 

import cv2 as cv 
import numpy as np
import keras 
from keras.models import load_model
from keras.datasets import mnist 
from keras import backend as K 

# input image dimensions
img_rows, img_cols = 28, 28

threshold = 0.5 

def img_proc(img):
  grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  height, width = grayscale.shape 
  height_center, width_center = height//2, width//2 
  cropped = grayscale[height_center-50:height_center+50, width_center-50:width_center+50]
  resized = cv.resize(cropped, (28, 28))
  return cv.bitwise_not(resized)

def main():
  cap = cv.VideoCapture(0)
  cv.namedWindow("Video", 1)

  model = load_model("../data/model.h5")

  while True: 
    ret, frame = cap.read()
    
    processed = img_proc(frame)
    cv.imshow("Video", frame)

    if K.image_data_format() == 'channels_first':
      data = processed.reshape(1, 1, img_rows, img_cols)
    else:
      data = processed.reshape(1, img_rows, img_cols, 1)

    predicted = model.predict(data)

    for i, x in np.ndenumerate(predicted[0]):
      if x > threshold: 
        number = i
        print("predicted number ", number, ", confidence value", x)

    key = cv.waitKey(50)
    if key & 0xFF == ord('q'):
      print(key)
      break

  cap.release()


if __name__ == "__main__":
  main()


