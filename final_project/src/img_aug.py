import os 

import cv2 as cv
from cv2 import imread, imshow, waitKey
import numpy as np
from keras import backend as K 

img_rows = img_cols = 64

def augment(img):
    print(img.shape)

    # flip
    flipped_img = np.fliplr(img)
    imshow("flipped", flipped_img)

    # shifting right
    shifted_right = np.copy(img)
    for j in range(img_rows):
        for i in range(img_cols):
            if (i < img_cols-20):
                shifted_right[j][i] = img[j][i+20]
    imshow("shift_right", shifted_right)

    # noise
    noised = np.copy(img)
    noise = np.random.randint(5, size = (64, 64, 3))
    for i in range(64):
        for j in range(64):
            for k in range(3):
                if (noised[i][j][k] != 255):
                    noised[i][j][k] += noise[i][j][k]
    imshow("noise", noised)

    return [flipped_img, shifted_right, noised]


if __name__ == "__main__":
    img_name = "7.jpeg" 
    img = cv.imread(img_name)
    resized = cv.resize(img, (64, 64))
    
    augmented = augment(resized)

    waitKey(0)


