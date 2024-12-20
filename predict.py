from keras.models import load_model
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys

# Load trained model
saved = load_model("U_Net.keras")

def pred_image(idx):

    # Read image filename
    img_name = annot["images"][int(idx)]["file_name"]

    # Load image
    inp_img = cv2.imread(f"{dir}/{img_name}")

    # Resize image to model input dimensions
    sample_img = cv2.resize(inp_img,(128,128))

    # Convert image to numpy array
    sample_img = np.array(sample_img)

    # Normalize pixels
    sample_img = sample_img.astype('float32') / 255.0

    # Predict the tumor regions
    pred_mask = saved.predict(sample_img.reshape(1,128,128,3))[0]

    # Remove background noise
    pred_mask[pred_mask < 0.3] = 0

    # Resize back to original dimensions
    pred_mask = cv2.resize(pred_mask, (640,640))

    return inp_img, pred_mask


if __name__ == "__main__":

    # Input the dataset folder for prediction
    dir = f"dataset/{sys.argv[1]}"

    # Read json file
    annot = json.load(open(glob.glob(f"{dir}/*.json")[0]))

    # Display true images with bounding boxes and predicted segmentation area
    if len(sys.argv) == 2:
        num_images = 9
        indices = np.random.randint(0, len(annot["images"]), size = num_images)
        for i, idx in zip(range(0,18,2),indices):

            plt.subplot(3, 6, i+1)
            plt.axis("off")

            true_img, pred_img = pred_image(idx)

            copy_img = true_img.copy()

            # Read bounding box data
            bbox = annot["annotations"][int(idx)]["bbox"]
            bbox = np.array(bbox, dtype=np.int32)
            x, y, w, h =  bbox

            # Draw bounding box
            cv2.rectangle(true_img, (x, y), (x+w, y+h), (0,255,0), 3)

            # Show true image with bounding box
            plt.imshow(true_img)

            # Show predicted segmentation
            plt.subplot(3, 6, i+2)
            plt.axis("off")
            plt.tight_layout()
            plt.imshow(copy_img)
            plt.imshow(pred_img, cmap="jet", alpha=0.5)

    # Display single image and prediction
    else:

        # Read index
        idx = sys.argv[2]
        plt.subplot(1, 2, 1)
        plt.axis("off")

        true_img, pred_img = pred_image(idx)

        copy_img = true_img.copy()
        
        # Read bounding box data
        bbox = annot["annotations"][int(idx)]["bbox"]
        bbox = np.array(bbox, dtype=np.int32)
        x, y, w, h =  bbox

        # Draw bounding box
        cv2.rectangle(true_img, (x, y), (x+w, y+h), (0,255,0), 3)

        # Show true image with bounding box
        plt.imshow(true_img)

        # Show predicted segmentation
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.tight_layout()
        plt.imshow(copy_img)
        plt.imshow(pred_img, cmap="jet", alpha=0.5)

    plt.show()
            