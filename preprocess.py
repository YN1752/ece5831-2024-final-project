import cv2
import numpy as np
import json
import glob

def load_data():

    target_size = (128, 128)

    dataset = []

    print("Loading preprocessed dataset ....")

    # Iterate through dataset folders
    for dir in ["dataset/train", "dataset/valid", "dataset/test"]:

        # Specify mask directory to read
        mask_dir = f"{dir}_mask"

        # Read json file
        annot = json.load(open(glob.glob(f"{dir}/*.json")[0]))

        # Read images
        X = [cv2.resize(cv2.imread(dir + "/" + image['file_name']), target_size) for image in annot['images']]
        y = [cv2.resize(cv2.imread(mask_dir + "/mask_" + image['file_name'], cv2.IMREAD_GRAYSCALE), target_size) for image in annot['images']]

        # Convert to numpy array
        X = np.array(X)
        y = np.expand_dims(np.array(y), axis=-1)

        # Normalize pixels
        X = X.astype('float32') / 255.0
        y = y.astype('float32') / 255.0

        # Remove noise
        y = (y > 0.5).astype(np.float32)

        # Append to dataset list
        dataset.append(X)
        dataset.append(y)

    print("Done")

    return dataset