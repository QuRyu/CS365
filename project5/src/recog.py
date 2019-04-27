import numpy as np
import keras 
from keras.models import load_model
from keras.datasets import mnist 
from keras import backend as K 

# input image dimensions
img_rows, img_cols = 28, 28

def main():
  model = load_model("../data/model.h5")

  _, (x_test, y_test) = mnist.load_data()

  if K.image_data_format() == 'channels_first':
    print("channels first")
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
  else:
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

  predictions = model.predict(x_test[:10])
  for i in range(10):
      print("predicted {}, actual {}".format(predictions[i], y_test[i]))

  

if __name__ == "__main__":
  main()
