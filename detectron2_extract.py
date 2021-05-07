import os
import io
import pickle
import detectron2
import json
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
import multiprocessing
# import some common libraries
import numpy as np
import cv2
import torch
from detectron2.model_zoo import model_zoo
from detectron2.modeling.poolers import ROIPooler
# Show the image in ipynb
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference_single_image
import pandas as pd
import glob
from tqdm import tqdm

def process_feature_extraction(predictor, img, pred_thresh=0.5):#step 3
    '''
    #predictor.model.roi_heads.box_predictor.test_topk_per_image = 1000
    #predictor.model.roi_heads.box_predictor.test_nms_thresh = 0.99
    #predictor.model.roi_heads.box_predictor.test_score_thresh = 0.0
    #pred_boxes = [x.pred_boxes for x in instances]#can use prediction boxes
    '''
    torch.cuda.empty_cache()
    with torch.no_grad():#https://detectron2.readthedocs.io/_modules/detectron2/modeling/roi_heads/roi_heads.html : _forward_box()
        features = predictor.model.backbone(img.tensor)#have to unsqueeze
        proposals, _ = predictor.model.proposal_generator(img, features, None)
        results, _ = predictor.model.roi_heads(img, features, proposals, None)
        #instances = predictor.model.roi_heads._forward_box(features, proposals)
        #get proposed boxes + rois + features + predictions

        proposal_boxes = [x.proposal_boxes for x in proposals]
        proposal_rois = predictor.model.roi_heads.box_pooler([features[f] for f in predictor.model.roi_heads.in_features], proposal_boxes)
        box_features = predictor.model.roi_heads.box_head(proposal_rois)
        predictions = predictor.model.roi_heads.box_predictor(box_features)#found here: https://detectron2.readthedocs.io/_modules/detectron2/modeling/roi_heads/roi_heads.html
        #WE CAN USE THE PREDICTION CLS TO FIND TOP SCOREING PROPOSAL BOXES!
        #pred_instances, losses = predictor.model.roi_heads.box_predictor.inference(predictions, proposals)#something to do with: NMS threshold for prediction results. found: https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/roi_heads/fast_rcnn.py#L460
        pred_df = pd.DataFrame(predictions[0].softmax(-1).tolist())
        pred_classes = pred_df.iloc[:,:-1].apply(np.argmax, axis=1)#get predicted classes
        keep = pred_df[pred_df.iloc[:,:-1].apply(lambda x: (x > pred_thresh)).values].index.tolist()#list of instances we should keep
        #start subsetting
        try:
            keep = keep[:NUM_OBJECTS]
            box_features = box_features[keep]
        except:
            return None
        # proposal_boxes = proposals[0].proposal_boxes[keep]
        # pred_classes = pred_classes[keep]
        # try:
        #     probs = pred_df.iloc[keep, :].apply(lambda x: x[np.argmax(x)], axis=1).tolist()
        # except:
        #     return None

    #['bbox', 'num_boxes', 'objects', 'image_width', 'image_height', 'cls_prob', 'image_id', 'features']
    #img.image_sizes[0]#h, w
    return box_features.to('cpu').detach().numpy()
    # result = {
    #     'bbox': proposal_boxes.tensor.to('cpu').numpy(),
    #     'num_boxes' : len(proposal_boxes),
    #     'objects' : pred_classes.to_numpy,
    #     'cls_prob': np.asarray(probs),#needs to turn into vector!!!!!!!!!!
    #     'features': box_features.to('cpu').detach().numpy()
    # }
    # return result

if __name__ == '__main__':
    mode = 'train'
    batch_size = 80
    images_path = f'/oscar_images/{mode}'
    features_file_path = f'/oscar_data/{mode}_img_feats.pt'
    NUM_OBJECTS = 50
    base_path = '/ocean/projects/tra190016p/ylix/dacon_old/dacon'
    saved_model_path = '/trained_models'
    num_gpu = 1
    model_name = "model_final.pth"
    NMS_THRESH = 0.6
    SCORE_THRESH = 0.4
    params = {'base_path':            base_path,     # NOTE: base path of the environment.
              'min_points_threshold': 500,     # NOTE: Minimum number of instances of an atom to be considered as a label. Atoms with less than this value are considered "other".
              'n_sample_hard':        500000, 
              'n_sample_per_label':   50000,   # NOTE: applies to both train and validation sets. Originally 20000
              'overwrite':            False,   # NOTE: determines if we overwrite existing data.
              'input_format':         "RGB",   # NOTE: Important to set depending on data format!
              'n_jobs':               multiprocessing.cpu_count() - 1,
              'train_path':            '/data/pubchem_smiles_2000000.csv',
              'val_size':            4000,
              'saved_model_path':    saved_model_path,
              'num_gpu':             num_gpu
    }
    train_params = {'images_per_batch':         batch_size,
                    'learning_rate':            0.005,
                    'maximum_iterations':       50000,
                    'checkpoint_save_interval': 10000,
                    'ROI_batch_per_image':      1024,
                    'evaluation_interval':      4000,
                    'num_workers':              8}
                    
    unique_labels = json.load(open(base_path + f'/data/labels.json', 'r'))
    unique_labels['other'] = 0
    labels = list(unique_labels.keys())

    for mode in ["train", "val"]:
        dataset_name = f"smilesdetect_{mode}"
        if dataset_name in DatasetCatalog.list():
            DatasetCatalog.remove(dataset_name)
            
        DatasetCatalog.register(dataset_name, lambda mode=mode: pickle.load(open(base_path + f'/data/annotations_{mode}.pkl', 'rb')))
        MetadataCatalog.get(dataset_name).set(thing_classes=labels)
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    # Passing the Train and Validation sets
    cfg.DATASETS.TRAIN = ("smilesdetect_train",)
    cfg.DATASETS.TEST = ("smilesdetect_val",)
    cfg.OUTPUT_DIR = base_path + saved_model_path
    cfg.INPUT.FORMAT = params['input_format']
    # Number of data loading threads

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(unique_labels)
    cfg.NUM_GPUS = num_gpu
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.SOLVER.IMS_PER_BATCH = train_params['images_per_batch']
    cfg.SOLVER.BASE_LR = train_params['learning_rate']
    cfg.SOLVER.MAX_ITER = train_params['maximum_iterations']
    cfg.SOLVER.CHECKPOINT_PERIOD = train_params['checkpoint_save_interval']
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = train_params['ROI_batch_per_image']
    cfg.TEST.EVAL_PERIOD = train_params['evaluation_interval']
    cfg.DATALOADER.NUM_WORKERS = train_params['num_workers']
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_name)
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = NMS_THRESH
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH
    predictor = DefaultPredictor(cfg)
    print("Model loaded.")
    features_dict = {}

    images_paths = glob.glob(base_path + f'{images_path}/*.png')
    # extract_features(predictor, images_paths, features_dict)
    # torch.save(features_dict, features_file_path)

    # result = extract_features(predictor, [base_path + "/data/images/test/rem.png"], features_dict)
    # print(result["features"])
    print(f"Extracting features for {mode} dataset.")

    for image_name in tqdm(images_paths):
        img = cv2.resize(cv2.imread(image_name), (300,300))[:, :, ::-1]
        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        image_example = [{"image": img, "height": img.shape[0], "width": img.shape[1]}]
        imageList = predictor.model.preprocess_image(image_example)
        #returns features and infos
        features = process_feature_extraction(predictor, imageList)
        if features is not None:
            features_dict[os.path.basename(image_name)] = features

    #assert features_dict.keys() == len(images_paths)
    print(f"There are {len(features_dict.keys())} pictures generated")
    torch.save(features_dict, base_path + features_file_path)
