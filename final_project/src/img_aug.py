import cv2 as cv
from cv2 import imread, imshow, waitKey
import numpy as np

img_rows = img_cols = 64

def augment(img):
    # flip
    flipped_img = np.fliplr(img)

    # shifting right
    shifted_right = np.copy(img)
    for j in range(img_cols):
        for i in range(img_rows):
            if (i < img_rows-20):
                shifted_right[j][i] = img[j][i+20]

    # noise
    noised = np.copy(img)
    noise = np.random.randint(5, size = (64, 64, 3))
    for i in range(64):
        for j in range(64):
            for k in range(3):
                if (noised[i][j][k] != 255):
                    noised[i][j][k] += noise[i][j][k]

    # rotate
    M = cv.getRotationMatrix2D((img_cols/2,img_rows/2), 90, 1.0)
    rotated = cv.warpAffine(img, M, (img_cols,img_rows))

    return [flipped_img, shifted_right, noised, rotated]


if __name__ == "__main__":
    img_name = "7.jpeg" 
    img = cv.imread(img_name)
    resized = cv.resize(img, (64, 64))
    
    augmented = augment(resized)

    waitKey(0)


