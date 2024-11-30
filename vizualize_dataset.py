import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys


def display_image():

    # Read image filename
    img_name = annot["images"][int(idx)]["file_name"]

    # Load image
    sample_img = cv2.imread(f"{dir}/{img_name}")

    # Get Bounding box coordinates
    bbox = annot["annotations"][int(idx)]["bbox"]
    bbox = np.array(bbox, dtype=np.int32)
    x, y, w, h =  bbox

    # Draw bounding box
    cv2.rectangle(sample_img, (x, y), (x+w, y+h), (0,255,0), 3)

    return sample_img


if __name__ == "__main__":

    # Input the dataset folder for prediction
    dir = f"dataset/{sys.argv[1]}"

    # Read json file
    annot = json.load(open(glob.glob(f"{dir}/*.json")[0]))

    # Randomly display 9 images with bounding box
    if len(sys.argv) == 2:
        num_images = 9
        indices = np.random.randint(0, len(annot["images"]), size = num_images)
        for i, idx in enumerate(indices):
            plt.subplot(3, 3, i+1)
            plt.axis("off")
            plt.tight_layout()
            plt.imshow(display_image())

    # Display single image
    else:
        idx = sys.argv[2]
        plt.imshow(display_image())

    plt.show()
            