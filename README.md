In this project I trained a U-Net architecture model for detecting tumor regions in the MRI scan images of the brain using Semantic Segmentation. I borrowed the dataset from Kaggle which contains various MRI scan images and data of the tumor region. Mask images were created which is basically creating ground truth images using the segmentation data. Preprocessing techniques were applied before training the model. Final result is a colormap image with detected tumor regions labeled with a different color.

The project contains following scripts and notebooks:

`doucumentation.ipynp`  : Used for testing functions before implementing them in the python scripts<br>
`vizualize_dataset.py`  : Used for displaying the images in the dataset with bounding boxes showing the tumor region<br>
`create_masks.py`       : Create mask images which would act as target attribute for training the model<br>
`preprocess.py`         : For prerocessing and loading the images for training<br>
`train.py`              : For training the model<br>
`predict.py`            : To display the predictions<br>
`final-project.ipynb`   : To show how to run the scripts

Dataset: https://drive.google.com/file/d/1EeuEtN2WGJZM1ToK_9QXx1fU3LtBdJnE/view?usp=sharing<br>
Presentation Slides: https://docs.google.com/presentation/d/1cjPYJYh3qi72pRlro83hyDv0amBk68Ma/edit?usp=drive_link&ouid=106661166248148569804&rtpof=true&sd=true<br>
Presentation Video: https://youtu.be/FpXvROZYz1o <br>
Report: https://drive.google.com/file/d/17aovJlcu1gaIkN1KiB72GDWeAGqFn4hM/view?usp=sharing<br>
Demo Video: https://youtu.be/qmkUdwrKwlc<br>
Model: https://drive.google.com/file/d/1BrLLVyB-YN--a3QQ8rSwcL8iabc8CQoa/view?usp=drive_link

The trained model was too large to upload in GitHub. So I uploaded the model in Google Drive
Drive Link: https://drive.google.com/drive/folders/1EknkFe_QwiIOnaABdWnzuD_wZdcsKlkG?usp=drive_link
