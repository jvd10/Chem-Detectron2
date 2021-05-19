import os

import multiprocessing
import trainer
import sys
import cv2
from tqdm import tqdm
import torch
import glob

if __name__ == '__main__':
    # run flags
    train_new_model = False                    # NOTE: if false, previous model is restored
    #test_data_path = '/data/images/test'
    test_data_path = '/oscar_images/val'
    model_name = 'Resnet101_FPN_3x_' + test_data_path.split("/")[-1]
    checkpoint_name = 'model_final.pth'
    predictions_path = './predictions'
    device = 'cuda'
    base_path = '/ocean/projects/tra190016p/ylix/dacon'
    # general parameters
    # './data/train.csv'
    params = {'base_path':            base_path,     # NOTE: base path of the environment.
              'min_points_threshold': 500,     # NOTE: Minimum number of instances of an atom to be considered as a label. Atoms with less than this value are considered "other".
              'n_sample_hard':        500000, 
              'n_sample_per_label':   50000,   # NOTE: applies to both train and validation sets. Originally 20000
              'overwrite':            False,   # NOTE: determines if we overwrite existing data.
              'input_format':         "RGB",   # NOTE: Important to set depending on data format!
              'n_jobs':               multiprocessing.cpu_count() - 1,
              'train_path':            '/data/pubchem_smiles_2000000.csv',
              'val_size':            4000,
              'saved_model_path':    '/trained_models',
              'num_gpu':             1
    }
    batch_size = 70
    # train parameters
    train_params = {'images_per_batch':         batch_size,
                    'learning_rate':            0.005,
                    'maximum_iterations':       50000,
                    'checkpoint_save_interval': 10000,
                    'ROI_batch_per_image':      1024,
                    'evaluation_interval':      4000,
                    'num_workers':              8}

    # construct trainer object
    tr = trainer.Trainer(params=params)
    
    print("Pre-processing finished.")

    # train model
    if train_new_model:
     tr.train(train_params=train_params)

    # # restore trained model
    tr.load_model(checkpoint_name, device=device)

    # # predict test images per batch in parallel
    results = tr.predict_batch(images_path=test_data_path, batch_size=batch_size)

