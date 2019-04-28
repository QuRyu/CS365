import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv 
import keras 
from keras import models
from keras.models import load_model 
from keras.datasets import mnist 
from keras import backend as K 

img_rows, img_cols = 28, 28

model = load_model("../data/model.h5")

(x_train, y_train), _ = mnist.load_data()

if K.image_data_format() == 'channels_first':
  x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
else:
  x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

img = x_train

first_layer = model.get_layer(index=1)
weights = np.array(first_layer.get_weights())[0]
weights = weights.reshape(32, 1, 3, 3)
maxVal = weights.max()
minVal = weights.min()
absMax = max(abs(minVal), abs(maxVal))
weights = (weights / absMax)*255
# weights = (weights - np.min(weights))/np.ptp(weights) # normalize weights 

# plot the filters
fig, axes = plt.subplots(nrows=8, ncols=4)
for i in range(8):
  for j in range(4):
    filt = weights[i*4+j][0]
    print(filt)
    axes[i][j].imshow(filt)
plt.show()

# plot the 32 filtered images
fig, axes = plt.subplots(nrows=8, ncols=4)
for i in range(8):
  for j in range(4):
    filtered = cv.filter2D(img[0], -1, weights[i*4+j][0])
    axes[i][j].imshow(filtered)
plt.show()


first_layer_model = models.Model(inputs=model.input, outputs=first_layer.output)
first_layer_model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adam(),
                          metrics=['accuracy'])

model_output = first_layer_model.evaluate(img)

# print(model_output.shape)

# fig, axes = plt.subplots(nrows=8, ncols=4)
# for i in range(8):
#   for j in range(4):
#     filtered = cv.filter2D(img, -1, weights[i*4+j][0])
#     axes[i][j].imshow(model_output)


# plt.show()

