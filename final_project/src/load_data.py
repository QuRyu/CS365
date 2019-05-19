import os 

from cv2 import imread, imshow, waitKey
import numpy as np
from keras import backend as K 

from img_aug import augment

img_rows = img_cols = 64

# load labels for each class 
def load_labels(path):
    words_path = os.path.join(path, 'words.txt')
    with open(words_path, 'r') as f: 
        words = dict(line.split('\t') for line in f)
        for nid, word in words.items():
            words[nid] = [w.strip() for w in word]

    return words 


# load wnids for each class
def load_wnids(path):
    wnids_path = os.path.join(path, 'wnids.txt')
    with open(wnids_path, 'r') as f: 
        wnids = [x.strip() for x in f]


    wnids_to_label = {wnid: i for i, wnid in enumerate(wnids)} # convert wnids to integer labels 
    return wnids, wnids_to_label


def load_train_data(path, augmentation, wnids, wnids_to_label, dtype):
    print("loading training data")

    x_train = []
    y_train = []

    for i, wnid in enumerate(wnids):

        # the boxes file gives the name of each image 
        boxes_file = os.path.join(path, 'train', wnid, '%s_boxes.txt' % wnid)
        with open(boxes_file, 'r') as f:
            filenames = [x.split('\t')[0] for x in f]
        num_images =  len(filenames) * 5 if augmentation else len(filenames)

        if K.image_data_format() == "channels_first": 
            x_train_block = np.zeros((num_images, 3, img_rows, img_cols), dtype=dtype)
        else:
            x_train_block = np.zeros((num_images, img_rows, img_cols, 3), dtype=dtype)
        y_train_block = wnids_to_label[wnid] * np.ones(num_images, dtype=np.int32)

        for j, img_file in enumerate(filenames):
            img_file = os.path.join(path, 'train', wnid, 'images', img_file)
            img = imread(img_file)

            if augmentation:
                augmented = augment(img)

                idx = j*5

                x_train_block[idx] = img
                for i in range(len(augmented)):
                    x_train_block[idx+i+1] = augmented[i]
            else:
                x_train_block[j] = img 


            
        x_train.append(x_train_block)
        y_train.append(y_train_block)


    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    return x_train, y_train


def load_validation_data(path, wnids, wnids_to_label, dtype):
    print("load validation data")
    with open(os.path.join(path, 'val', 'val_annotations.txt'), 'r') as f:
        img_files = []
        val_wnids = [] 

        for line in f: 
            if line.split()[1] in wnids: # only use images in training data 
                img_file, wnid = line.split('\t')[:2]
                img_files.append(img_file)
                val_wnids.append(wnid)

        num_images = len(img_files)
        
        if K.image_data_format() == "channels_first": 
            x_val = np.zeros((num_images, 3, img_rows, img_cols), dtype=dtype)
        else:
            x_val = np.zeros((num_images, img_rows, img_cols, 3), dtype=dtype)
        y_val = np.array([wnids_to_label[wnid] for wnid in val_wnids])

        for i, img_file in enumerate(img_files):
            img_file = os.path.join(path, 'val', 'images', img_file)
            img = imread(img_file)

            x_val[i] = img
        
        return x_val, y_val


def load_tiny_imagenet(path, augmentation=False, dtype=np.uint8):
    wnids, wnids_to_label = load_wnids(path)
    wnids_to_words = load_labels(path)

    x_train, y_train = load_train_data(path, augmentation, wnids, wnids_to_label, dtype)
    x_val, y_val = load_validation_data(path, wnids, wnids_to_label, dtype)

    return x_train, y_train, x_val, y_val, wnids, wnids_to_label, wnids_to_words


if __name__ == "__main__":
    x_train, y_val, x_val, y_val, wnids, wnids_to_words, wnids_to_words = \
            load_tiny_imagenet("/home/quryu/Downloads/tiny-imagenet-200")
    
    # print("are they equal? ", np.array_equal(first_img, x_train[0][0]))
    # imshow("first image", first_img)
    # imshow("block read", x_train[0][0])
    print(x_train.shape)




