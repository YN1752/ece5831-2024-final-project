import glob
import json
import cv2
import os
import numpy as np

# Iterate through datset folders
for dir in ["dataset/train", "dataset/test", "dataset/valid"]:

    # Name of the mask directory
    mask_dir = f"{dir}_mask"

    # Check if directory already exists
    if os.path.exists(mask_dir) == False:

        # Make mask directory
        os.mkdir(mask_dir)

        print(f"{mask_dir} directory created. Loading masks....")

        # Read json file
        annot = json.load(open(glob.glob(f"{dir}/*.json")[0]))

        # Iterate through images
        for idx in range(len(annot["images"])):
            
            # Read images
            img_name = annot["images"][idx]["file_name"]
            img = cv2.imread(f"{dir}/{img_name}")

            # Create masks by first marking every pixel to black
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

            # Get bounding box coordinates
            bbox = annot["annotations"][idx]["bbox"]
            bbox = np.array(bbox, dtype=np.int32)
            x, y, w, h =  bbox

            # Mark bounding box region to white
            cv2.rectangle(mask, (x, y), (x+w, y+h), (255,255,255), -1)

            # Save masks into mask directory
            cv2.imwrite(f"{mask_dir}/mask_{img_name}", mask)

        print("Done")
    
    # If dataset already exists, print message
    else:
        print(f"{mask_dir} already exists")

