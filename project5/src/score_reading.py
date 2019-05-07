import numpy as np 
import sys 

def main(args):
  print(args) 
  train_fp = args[1]
  test_fp = args[2]

  train_scores = np.load(open(train_fp, "rb"))
  test_scores = np.load(open(test_fp, "rb"))

  train_loss, train_acc = train_scores.T
  test_loss, test_acc = test_scores.T 

  for i in range(train_scores.shape[0]):
      print("train loss {}, train acc {}, test loss{}, test acc {}".format(train_loss[i], train_acc[i], test_loss[i], test_acc[i]))

if __name__ == "__main__":
    main(sys.argv)
