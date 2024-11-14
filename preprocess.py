import cv2
import numpy as np
import json
import glob
import pickle

target_size = (128, 128)

dataset = []

print("Creating preprocessed dataset ....")

for dir in ["dataset/train", "dataset/valid", "dataset/test"]:

    mask_dir = f"{dir}_mask"

    annot = json.load(open(glob.glob(f"{dir}/*.json")[0]))

    X = [cv2.resize(cv2.imread(dir + "/" + image['file_name']), target_size) for image in annot['images']]
    y = [cv2.resize(cv2.imread(mask_dir + "/mask_" + image['file_name'], cv2.IMREAD_GRAYSCALE), target_size) for image in annot['images']]

    X = np.array(X)
    y = np.expand_dims(np.array(y), axis=-1)

    X = X.astype('float32') / 255.0
    y = y.astype('float32') / 255.0

    y = (y > 0.5).astype(np.float32)

    dataset.append(X)
    dataset.append(y)

pickle.dump(dataset, open("dataset.pkl", "wb"))

print("Done.")