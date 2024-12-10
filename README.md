In this project I trained a U-Net architecture model for detecting tumor regions in the MRI scan images of the brain using Semantic Segmentation. I borrowed the dataset from Kaggle which contains various MRI scan images and data of the tumor region. Mask images were created which is basically creating ground truth images using the segmentation data. Preprocessing techniques were applied before training the model. Final result is a colormap image with detected tumor regions labeled with a different color.

`doucumentation.ipynp` : Used for testing functions before implementing them in the python scripts<br>
`vizualize_dataset.py` : Used for displaying the images in the dataset with bounding boxes showing the tumor region<br>
`create_masks.py` : Creat mask images which would act as target attribute for training the model<br>
`preprocess.py` : For prerocessing and loading the images for training<br>
`train.py` : For training the model<br>
`predict.py` : To display the predictions<br>

Dataset: https://www.kaggle.com/datasets/pkdarabi/brain-tumor-image-dataset-semantic-segmentation?select=train<br>
Presentation: https://docs.google.com/presentation/d/1NWWjN8O9xHF__m79rCFwqI_byLDv7sYF/edit?usp=sharing&ouid=106661166248148569804&rtpof=true&sd=true<br>
Report: https://drive.google.com/file/d/17aovJlcu1gaIkN1KiB72GDWeAGqFn4hM/view?usp=sharing<br>
Demo: https://youtu.be/-3EobRN6PD8<br>

The trained model was too large to upload in GitHub. So I uploaded the model in Google Drive
