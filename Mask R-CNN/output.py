# import some common libraries
import os
import random
import numpy as np
import cv2
import glob
from utils import utils
from fvcore.common.file_io import PathManager

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.projects import point_rend

# Dataset registration
train_img_dir = './post_img/'
train_mask_dir = './post_msk/'
utils_ = utils(train_img_dir, train_mask_dir)
dataset_train_dicts, dataset_val_dicts = utils_.register()

cfg = get_cfg()
# Add PointRend-specific config
point_rend.add_pointrend_config(cfg)

# Load a config from file
cfg.merge_from_file("detectron2/projects/PointRend/configs/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes.yaml")
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(utils_.class_ids)
cfg.MODEL.POINT_HEAD.NUM_CLASSES = len(utils_.class_ids)

cfg.OUTPUT_DIR="./outputs/whole_dataset/"
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
#cfg.INPUT.COLOR_AUG_SSD = False


predictor = DefaultPredictor(cfg)
dataset_dicts = glob.glob('./pre_img/*')

for d in dataset_dicts: # shuffle the validation dataset

    im = cv2.imread(d)

    # Produce predicted image
    outputs = predictor(im) 
    outputs = outputs["sem_seg"].argmax(dim=0)
    out_mask = outputs.to("cpu").numpy()

    cv2.imwrite(d.replace('/pre_img/', '/output_img/'), out_mask)
    np.savetxt(d.replace('/pre_img/', '/output_csv/').replace('.png', '.csv'), out_mask)
