# CS365 Spring 2019 
# Project 5
# Qingbo Liu, Iris Lian 

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv 
import keras 
from keras import models
from keras.models import load_model
from keras.layers import Dense, Reshape
from keras.datasets import mnist 
from keras import backend as K 
import csv 
from sklearn.neighbors import KNeighborsClassifier

img_rows, img_cols = 28, 28

def read_csv():
    data = np.loadtxt("../data/greek_data.csv", delimiter=' ', skiprows=1, dtype=np.uint8)
    label = np.loadtxt("../data/greek_label.csv", delimiter=' ', skiprows=1, dtype=np.int)

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

def build_knn(model, output_size):
    # Flatten feature vector
    flat_dim_size = np.prod(model.output_shape[1:])
    x = Reshape(target_shape=(flat_dim_size,),
                name='features_flat')(model.output)
    
    # Dot product between feature vector and reference vectors
    x = Dense(units=output_size,
              activation='linear',
              name='knn',
              use_bias=False)(x)   
                
    classifier = models.Model(inputs=[model.input], outputs=x)
    return classifier

def main():
    model = load_model("../data/model.h5")
    # index = 6 when on lab's computer otherwise index = 5
    model = models.Model(inputs=model.input, outputs=model.get_layer(index=6).output)
    model.compile(loss=keras.losses.categorical_crossentropy,
                         optimizer=keras.optimizers.Adam(),
                         metrics=['accuracy'])

    data, label = read_csv()

    output = model.predict(data)
    ssd1 = ssd(output[0], output)
    ssd2 = ssd(output[7], output)
    ssd3 = ssd(output[8], output)
    print("{}\n\n{}\n\n{}".format(ssd1, ssd2, ssd3))

    # model = load_model("../data/model.h5")
    # joined_model = build_knn(model, 27)
    # joined_model.summary()

    # # temp_weights = joined_model.get_weights()
    # # temp_weights[-1] = data/np.linalg.norm(data, axis=0)
    # # joined_model.set_weights(temp_weights)

    # output = joined_model.predict(data)
    # print(output.shape)
    # # linfnorm = np.linalg.norm(output, axis=1, ord=np.inf)
    # # output = output.astype(np.float) / linfnorm[:,None]
    # print(output)

if __name__ == "__main__":
    main()

