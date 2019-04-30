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

first_layer = model.get_layer(index=0)
weights = np.array(first_layer.get_weights())[0]
weights = weights.reshape(32, 1, 3, 3)
maxVal = weights.max()
minVal = weights.min()
absMax = max(abs(minVal), abs(maxVal))
weights = (weights / absMax)*255
# weights = (weights - np.min(weights))/np.ptp(weights) # normalize weights 

# plot the filters
fig0, axes0 = plt.subplots(nrows=8, ncols=4)
for i in range(8):
  for j in range(4):
    filt = weights[i*4+j][0]
    axes0[i][j].imshow(filt)

# plot the 32 filtered images
fig1, axes1 = plt.subplots(nrows=8, ncols=4)
for i in range(8):
  for j in range(4):
    filtered = cv.filter2D(img[2], -1, weights[i*4+j][0])
    axes1[i][j].imshow(filtered)


first_layer_model = models.Model(inputs=model.input, outputs=first_layer.output)
first_layer_model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adam(),
                          metrics=['accuracy'])

model_output = first_layer_model.predict(img)

output2 = model_output[2]
fig2, axes2 = plt.subplots(nrows=8, ncols=4)
for i in range(8):
  for j in range(4):
      axes2[i][j].imshow(output2[:, :, i*4+j])


second_layer = model.get_layer(index=1)
two_layers_model = models.Model(inputs=model.input, outputs=second_layer.output)
two_layers_model.compile(loss=keras.losses.categorical_crossentropy,
                         optimizer=keras.optimizers.Adam(),
                         metrics=['accuracy'])
model_output2 = two_layers_model.predict(img)

output3 = model_output2[2]
fig3, axes3 = plt.subplots(nrows=8, ncols=4)
for i in range(8):
  for j in range(4):
      axes3[i][j].imshow(output3[:, :, i*4+j])


third_layer = model.get_layer(index=2) 
three_layers_model = models.Model(inputs=model.input, outputs=third_layer.output)
three_layers_model.compile(loss=keras.losses.categorical_crossentropy,
                         optimizer=keras.optimizers.Adam(),
                         metrics=['accuracy'])
model_output3 = three_layers_model.predict(img) 

output4 = model_output3[2] 
fig4, axes4 = plt.subplots(nrows=8, ncols=4)
for i in range(8):
  for j in range(4):
      axes4[i][j].imshow(output4[:, :, i*4+j])



plt.show()

