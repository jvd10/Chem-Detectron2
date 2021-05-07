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
    base_path = '/ocean/projects/tra190016p/ylix/dacon_old/dacon'
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

    # # saves predictions to
    #results.to_csv(f'{predictions_path}/predictions_{model_name}.csv', index=False)
    print("We finished.")

    # # DACON2OSCAR
    # #images = ImageList.from_tensors(...)  # preprocessed input tensor
    # predictor = tr.predictor
    #     # with torch.no_grad():
    #     #     # Apply pre-processing to image.
    #     #     if self.input_format == "RGB":
    #     #         # whether the model expects BGR inputs or RGB
    #     #         images_batch = [img[:, :, ::-1] for img in images_batch]
    #     #     height, width = images_batch[0].shape[:2]
    #     #     inputs = []
    #     #     for image in images_batch:
    #     #         image = self.aug.get_transform(image).apply_image(image)
    #     #         image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    #     #         inputs.append({"image": image, "height": height, "width": width})
    #     #     predictions = self.model(inputs)
    #     #     return predictions
    # images = None
    # images_paths = glob.glob(base_path + f'{test_data_path}/*.png')
    # for i in tqdm(range(0, len(images_paths), batch_size)):
    #             # input format for the model list[{"image"}: ...,], image: Tensor, image in (C, H, W) format.
    #     images = [cv2.resize(cv2.imread(path), (300,300))[:, :, ::-1] for path in images_paths[i:i + batch_size]]
    # if predictor.input_format == "RGB":
    #     images = [img[:, :, ::-1] for img in images]
    # for i, _ in enumerate(images):
    #     images[i] = images[i].astype("float32").transpose(2, 0, 1)
    # # for a in images:
    # #     print(a.shape)
    # images = torch.tensor(images)
    # print(images.shape)
    # predictor.model.eval()
    # with torch.no_grad():
    #     images = images.to(device)
    #     features = predictor.model.backbone(images)
    #     proposals, _ = predictor.model.proposal_generator(images, features)
    #     instances, _ = predictor.model.roi_heads(images, features, proposals)
    #     mask_features = [features[f] for f in predictor.model.roi_heads.in_features]
    #     mask_features = predictor.model.roi_heads.mask_pooler(mask_features, [x.pred_boxes for x in instances])
    #     print(mask_features.shape)
    #sys.exit()
