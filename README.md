# DACON Smiles competition Code - PBRH

Main approach uses a RCNN type model to directly predict individual atom and bond locations, and constructs molecules from those.


## Installation & Setup

This is setup on Ubuntu 18.04 + torch 1.7.1 + cuda 11.0

1. Use the following to install all dependencies:
```bash
bash setup.sh
```
2. Create Folders:
    - Create a images/ folder inside data/ folder
    - Create train/ and dev/ folders inside images/ folder

## Usage

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
