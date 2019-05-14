
import numpy as np
import cv2 as cv 
from keras.models import load_model 
from keras import backend as K 

NUM_IMG = 10 
img_rows, img_cols = 28, 28
img_fp = "../data/numbers/"

def main():
    # read in images and resize them 
    images = np.zeros((NUM_IMG,28,28))
    for i in range(NUM_IMG): 
      img_name = img_fp + str(i) + ".jpeg" 
      img = cv.imread(img_name)
      img_binary = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
      _, img_thresholded = cv.threshold(img_binary, 250, 255, cv.THRESH_BINARY_INV)
      resized = cv.resize(img_thresholded, (28, 28))
      images[i] = resized

    # for i in range(NUM_IMG): 
      # cv.imshow(str(i), images[i])
    # cv.waitKey(0)

    model = load_model("../data/model.h5")

    
    if K.image_data_format() == 'channels_first':
      images = images.reshape(images.shape[0], 1, img_rows, img_cols)
    else:
      images = images.reshape(images.shape[0], img_rows, img_cols, 1)

    result = model.predict(images)
    for i in range(NUM_IMG):
      print("predicted {}, actual {}".format(result[i], i))
 


if __name__ == "__main__":
    main()
