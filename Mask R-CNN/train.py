# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os
import cv2
import random
from utils import *
from fvcore.common.file_io import PathManager

# import some common detectron2 utilities
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.projects import point_rend
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog

######### 

# Dataset registration
train_img_dir = './post_img/'
train_mask_dir = './post_msk/'
utils_ = utils(train_img_dir, train_mask_dir)
dataset_train_dicts, dataset_val_dicts = utils_.register()

"""
# Visualize original img
for d in random.sample(dataset_post_train_dicts, 5):
    img = cv2.imread(d["file_name"])
    mask = cv2.imread(d["sem_seg_file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get('xview_post_train'),
                            scale=1, instance_mode=ColorMode.SEGMENTATION)
    out = visualizer.draw_dataset_dict(d)
    
    utils_.visualize(
        original_image = out.get_image()[:, :, ::-1],
        mask = mask
    )
"""

# Training
cfg = get_cfg()
point_rend.add_pointrend_config(cfg)
# Load a config from file
cfg.merge_from_file("detectron2/projects/PointRend/configs/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes.yaml")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = "./outputs/flood_2_with_xview/model_0154999.pth"
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(utils_.class_ids)
cfg.MODEL.POINT_HEAD.NUM_CLASSES = len(utils_.class_ids)

cfg.DATASETS.TRAIN = ("flood_train",)
cfg.DATASETS.TEST = ("flood_val",)


cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR
cfg.SOLVER.MAX_ITER = 100000 # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []


cfg.INPUT.MIN_SIZE_TRAIN = (1024,)
cfg.INPUT.MIN_SIZE_TEST = (1024,)
cfg.INPUT.MAX_SIZE_TRAIN = 1024
cfg.INPUT.MAX_SIZE_TEST = 1024
cfg.INPUT.CROP.ENABLED = False

cfg.DATALOADER.NUM_WORKERS = 5
cfg.OUTPUT_DIR="outputs/whole_dataset/"


# Revoke training
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()