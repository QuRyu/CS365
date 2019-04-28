import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist 

def plotFirstTwo():
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  fig, (first, second) = plt.subplots(nrows=2, ncols=1)

  first.imshow(x_train[0], cmap='gray')
  second.imshow(x_train[1], cmap='gray')

  plt.show()

def main():
  train_file = open("../data/train_scores", "rb") 
  test_file = open("../data/test_scores", "rb")
  train_scores = np.load(train_file) 
  test_scores = np.load(test_file)

  train_loss, train_acc = train_scores.T
  test_loss, test_acc = test_scores.T 

  x_axis = np.arange(1, 13)

  fig, (loss, acc) = plt.subplots(nrows=2, ncols=1)

  loss.set_title("loss")
  loss.plot(x_axis, train_loss, label="train")
  loss.plot(x_axis, test_loss, label="test")
  loss.legend()

  acc.set_title("accuracy")
  acc.set_xlabel("epochs")
  acc.plot(x_axis, train_acc, label="train")
  acc.plot(x_axis, test_acc, label="test")
  acc.legend()

  plt.show()


if __name__ == "__main__":
  # plotFirstTwo()
  main()