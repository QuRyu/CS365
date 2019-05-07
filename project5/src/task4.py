# CS365 Spring 2019 
# Qingbo Liu, Iris Lian 

import cv2 as cv 

import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import models 
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist 
from tensorflow.keras import backend as K 

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# set of dropout rates to evaluate, the pair (0.2, 0.4) has the best performance  
dropout_rates = [(0.1, 0.2), (0.2, 0.4), (0.25, 0.5), (0.3, 0.6), (0.35, 0.7), (0.4, 0.8)]


# on training data, (2,2) was the best 
# but (4, 4) performs better on test data 
kernel_sizes = [(2,2), (3,3), (4,4), (5,5)]

# 1024 was the best, but 512 was the worst 
dense_nodes = [64, 128, 256, 512, 1024]

batch_sizes = [32, 64, 128, 256, 512, 1024]

def main():
  np.random.seed(42)

  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
  else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255
  print('x_train shape:', x_train.shape)
  print(x_train.shape[0], 'train samples')
  print(x_test.shape[0], 'test samples')
  
  # convert class vectors to binary class matrices
  y_train = keras.utils.to_categorical(y_train, num_classes)
  y_test = keras.utils.to_categorical(y_test, num_classes)

  # arrays holding accuracy and losses 
  test_scores = np.zeros((len(batch_sizes), 2))
  train_scores = np.zeros((len(batch_sizes), 2))

  for i in range(len(batch_sizes)):
    model = models.Sequential()
    model.add(Conv2D(32, kernel_size=(4,4), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(4,4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train, 
                  batch_size=batch_sizes[i], 
                  epochs=12, 
                  verbose=1, 
                  validation_data=(x_test, y_test))
    train_scores[i] = model.evaluate(x_train, y_train, verbose=1)
    test_scores[i] = model.evaluate(x_test, y_test, verbose=1)

  train_file = open("../data/batch_sizes_train_scores", "wb")
  test_file = open("../data/batch_sizes_test_scores", "wb")
  np.save(train_file, train_scores)
  np.save(test_file, test_scores)

  # model.save("../data/model.h5")
    
if __name__ == "__main__":
    main()


