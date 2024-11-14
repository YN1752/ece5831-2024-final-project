import glob
import json
import cv2
import os
import numpy as np

for dir in ["dataset/train", "dataset/test", "dataset/valid"]:

    mask_dir = f"{dir}_mask"

    if os.path.exists(mask_dir) == False:

        os.mkdir(mask_dir)

        print(f"{mask_dir} directory created. Loading masks....")

        annot = json.load(open(glob.glob(f"{dir}/*.json")[0]))

        for idx in range(len(annot["images"])):
    
            img_name = annot["images"][idx]["file_name"]
            img = cv2.imread(f"{dir}/{img_name}")

            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

            bbox = annot["annotations"][idx]["bbox"]
            bbox = np.array(bbox, dtype=np.int32)
            x, y, w, h =  bbox
            cv2.rectangle(mask, (x, y), (x+w, y+h), (255,255,255), -1)

            cv2.imwrite(f"{mask_dir}/mask_{img_name}", mask)

        print("Done")

    else:
        print(f"{mask_dir} already exists")

