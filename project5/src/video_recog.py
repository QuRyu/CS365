# CS365 Spring 2019 
# Project 5
# Qingbo Liu, Iris Lian 
import operator 

import cv2 as cv 
import numpy as np
import keras 
from keras.models import load_model
from keras.datasets import mnist 
from keras import backend as K 

# input image dimensions
img_rows, img_cols = 28, 28

threshold = 0.5 

bounding_boxes = [30, 50, 70, 90, 110]

def img_proc(img):
  grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  height, width = grayscale.shape 
  height_center, width_center = height//2, width//2 

  frames = np.arange(len(bounding_boxes)*img_rows*img_cols).reshape(len(bounding_boxes), img_rows, img_cols)
  for i in range(len(bounding_boxes)):
      size = bounding_boxes[i]
      cropped = grayscale[height_center-size:height_center+size, width_center-size:width_center+size]
      resized = cv.resize(cropped, (28, 28))

      frames[i] = cv.bitwise_not(resized)
  return frames

def vote(predictions):
    numbers = [] 
    for pred in predictions:
        for i, x in np.ndenumerate(pred):
            if x > threshold:
                numbers.append(i[0])
                break 

    count = {}
    for n in numbers:
        if n in count:
            count[n] += 1 
        else:
            count[n] = 1 

    count = sorted(count.items(), key=lambda kv:kv[1])
    num, vote = count[len(count)-1]
     
    return num 

def main():
  cap = cv.VideoCapture(0)
  cv.namedWindow("Video", 1)

  model = load_model("../data/model.h5")

  while True: 
    ret, frame = cap.read()
    
    processed = img_proc(frame)
    cv.imshow("Video", frame)

    if K.image_data_format() == 'channels_first':
      data = processed.reshape(processed.shape[0], 1, img_rows, img_cols)
    else:
      data = processed.reshape(processed.shape[0], img_rows, img_cols, 1)

    predicted = model.predict(data)
    num = vote(predicted)
    print(num)

    key = cv.waitKey(50)
    if key & 0xFF == ord('q'):
      print(key)
      break

  cap.release()


if __name__ == "__main__":
  main()


