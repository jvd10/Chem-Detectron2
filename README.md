# Detectron2 Chemical Graph Image Feature Extractor
This repo is modified based on an AI competition top-performing model to exctract image features from chemical structure diagrams.
It utilizes FAIR's Detectron2-FastRCNN model to train on the image recognition task. It can be used to generate SMILES strings but it does not perform well on complicated chemical structure. It is better used as a visual backbone to extract image features to feed into a transformer-based model to generate the final SMILES string predictions.

## DACON Smiles competition Code - PBRH

Main approach uses a RCNN type model to directly predict individual atom and bond locations, and constructs molecules from those.


## Installation & Setup

This is setup on Ubuntu 18.04 + torch 1.7.1 + cuda 11.0

1. Use the following to install all dependencies:
```bash
bash setup.sh
```
2. Create Folders:
    - Create a images/ folder inside data/ folder
    - Create train/ and val/ folders inside images/ folder

## Trained Model
There is a trained model that is stored under the trained_models/ folder ready for use. It is trained with the following hyperparameters/parameters:
- 500,000 images of sizes 300 * 300
- batch size 12
- learning rate 0.005
- iterations 50,000
- ROIs per image 1024

## How to Use the Repo for various tasks
#### Train Chem-Detectron2 
- **PubChem_SMILES.ipynb**: Use this notebook to generate a list of SMILES (sampled from PubChem database that is updated daily).
- **main.py** : Run this file to generate image data from SMILES to train Chem-Detectron2 and run training and validation. (hyperparameters and filepaths can be changed in this file)
To run, simply do:
```python
python main.py
```

#### Generate Image Features from Chem-Detectron2 (as inputs to Chem-OSCAR model)
- **generate_images.py**: Run this file to generate images from SMILES (if the images do not exist already). (filepaths can be changed in this file)
- **detectron2_extract.py**: Run this file to generate image features from images. (hyperparameters and filepaths can be changed in this file)
To run, simply do:
```python
python generate_images.py
python detectron2_extract.py
```

## Comments on Usage from the original Repo
#### Usage
- **main.py** : Main entry to code, constructs trainer object to train model, and predicts smiles for images in test folder.
- **trainer.py** : Trainer class, preprocesses data, and trains model.
- **labels_generation.py** : Functions to preprocess data and generate annotations for RCNN.
- **inference.py** : Constructs mol object from predicted atom and bond bounding boxes .
- **utils.py** : Other general helper functions.

#### How to train
Set train_new_model to True in main.py and run main.py.

#### How to predict
Set test_data_path in main.py to the directory containing test images and run main.py.
Really important to set the parameter **'input_format'** to **"RGB" or "GBR"** depending
on the particular set of images!

#### Data
**No external data was used for any training or prediction**, all was calculated directly
from RDKit. Even the training images are generated in case they are not present in 
the folder **"./data/images/train"**.

#### Pretrained model
A pretrain model is available to obtain predictions similar to the competition submission.
**"./trained_models/final_submission.pth"**.

#### Final remarks
Due to random initialization difference between different operating systems, compute environment and GPU types, 
results can slightly differ between machines.  
