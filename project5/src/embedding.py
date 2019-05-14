import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv 
import keras 
from keras import models
from keras.models import load_model 
from keras.datasets import mnist 
from keras import backend as K 
import csv 

img_rows, img_cols = 28, 28

def read_csv():
    data = np.loadtxt("../data/greek_ours_data.csv", delimiter=' ', skiprows=1, dtype=np.uint8)
    label = np.loadtxt("../data/greek_ours_label.csv", delimiter=' ', skiprows=1, dtype=np.int)

    if K.image_data_format() == 'channels_first':
      data = data.reshape(data.shape[0], 1, img_rows, img_cols)
    else:
      data = data.reshape(data.shape[0], img_rows, img_cols, 1)
    return data, label

def ssd(x, xs):
    result = np.ndarray(xs.shape[0])
    for i in range(len(xs)): 
        result[i] = np.sum(np.square(xs[i] - x))

    return result 

def main():
    model = load_model("../data/model.h5")
    model = models.Model(inputs=model.input, outputs=model.get_layer(index=5).output)
    model.compile(loss=keras.losses.categorical_crossentropy,
                         optimizer=keras.optimizers.Adam(),
                         metrics=['accuracy'])

    data, label = read_csv()

    output = model.predict(data)
    # ssd1 = ssd(output[0], output)
    # ssd2 = ssd(output[7], output)
    # ssd3 = ssd(output[8], output)
    ssd1 = ssd(output[0], output) #gamma
    ssd2 = ssd(output[3], output) #alpha
    ssd3 = ssd(output[6], output) #beta
    print("{}\n\n{}\n\n{}".format(ssd1, ssd2, ssd3))

if __name__ == "__main__":
    main()

# if K.image_data_format() == 'channels_first':
  # x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
# else:
  # x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

