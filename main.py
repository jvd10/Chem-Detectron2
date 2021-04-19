import multiprocessing
import trainer

if __name__ == '__main__':
    # run flags
    train_new_model = False                    # NOTE: if false, previous model is restored
    test_data_path = '/data/images/test'
    model_name = 'Resnet101_FPN_3x_' + test_data_path.split("/")[-1]
    checkpoint_name = 'model_final.pth'
    predictions_path = './predictions'
    device = 'cuda'

    # general parameters
    # './data/train.csv'
    params = {'base_path':            '/content/LG_SMILES_1st',     # NOTE: base path of the environment.
              'min_points_threshold': 1,     # NOTE: Minimum number of instances of an atom to be considered as a label. Atoms with less than this value are considered "other".
              'n_sample_hard':        2000000, 
              'n_sample_per_label':   50000,   # NOTE: applies to both train and validation sets.
              'overwrite':            False,   # NOTE: determines if we overwrite existing data.
              'input_format':         "RGB",   # NOTE: Important to set depending on data format!
              'n_jobs':               multiprocessing.cpu_count() - 1,
              'train_path':            '/data/pubchem_smiles_2000000.csv'}
    # train parameters
    train_params = {'images_per_batch':         6,
                    'learning_rate':            0.005,
                    'maximum_iterations':       80000,
                    'checkpoint_save_interval': 500,
                    'ROI_batch_per_image':      256,
                    'evaluation_interval':      2000,
                    'num_workers':              8}

    # construct trainer object
    model = trainer.Trainer(params=params)
    
    print("Pre-processing finished.")

#     # train model
#     if train_new_model:
#         model.train(train_params=train_params)

#     # restore trained model
#     model.load_model(checkpoint_name, device=device)

#     # predict test images per batch in parallel
#     results = model.predict_batch(images_path=test_data_path)

#     # saves predictions to
#     results.to_csv(f'{predictions_path}/predictions_{model_name}.csv', index=False)
