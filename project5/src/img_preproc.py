# read in image file and convert them to csv file 

import os 
import csv 
import cv2 as cv

def main():
    path = "../data/greek/"
    files = [] 
    labels = {}

    for r, d, f in os.walk(path):
        for file in f:
            if '.png' in file: 
                files.append((os.path.join(r, file), file))

    with open("../data/greek_data.csv", "w", newline='') as dataF:
     with open("../data/greek_label.csv", "w") as labelF: 
      with open("../data/greek_label_map.csv", "w") as mapF:
        dataW = csv.writer(dataF, delimiter=' ')
        labelW = csv.writer(labelF, delimiter=' ')
        mapW = csv.writer(mapF, delimiter=' ')

        dataW.writerow(range(0, 784))
        labelW.writerow(["category"])
        mapW.writerow(["value", "category"])

        counter = 0

        for r, f in files:
            img = cv.imread(r)
            resized = cv.resize(img, (28, 28))
            img_gray = cv.cvtColor(resized, cv.COLOR_RGB2GRAY)
            img_gray = img_gray.flatten()

            dataW.writerow(img_gray)
            label = f.split("_")[0]

            if label not in labels:
                labels[label] = counter
                counter = counter + 1 

            labelW.writerow([labels[label]])

        for (k, v) in labels.items(): 
            mapW.writerow([v, k])
            


if __name__ == "__main__":
    main()
