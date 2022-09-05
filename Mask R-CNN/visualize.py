# import some common libraries
import os
import random
import numpy as np
import cv2
from utils import utils
from fvcore.common.file_io import PathManager
import pandas as pd

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.projects import point_rend
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog


# Dataset registration
train_img_dir = './data/img_1/'
train_mask_dir = './data/mask_1/'
utils_ = utils(train_img_dir, train_mask_dir)
dataset_train_dicts, dataset_val_dicts = utils_.register()

cfg = get_cfg()
# Add PointRend-specific config
point_rend.add_pointrend_config(cfg)

# Load a config from file
cfg.merge_from_file("detectron2/projects/PointRend/configs/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes.yaml")
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(utils_.class_ids)
cfg.MODEL.POINT_HEAD.NUM_CLASSES = len(utils_.class_ids)

cfg.OUTPUT_DIR="outputs/flood_3_with_vhmixup/"
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0029999.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
#cfg.INPUT.COLOR_AUG_SSD = False


predictor = DefaultPredictor(cfg)
dataset_dicts = dataset_val_dicts # load dataset from xview_post_val
output_list = []
name_list = []

for d in dataset_dicts: # shuffle the validation dataset

    # Load img
    im = cv2.imread(d["file_name"])
    gt_mask = cv2.imread(d["sem_seg_file_name"])
    name_list.append(d["file_name"].replace("./data/img_1/", ""))

    #im = cv2.imread("./data/img_1/Area2_post_cropped_58.png")
    #gt_mask = cv2.imread("./data/mask_1/Area2_post_cropped_58.png")

    # Produce predicted image
    outputs = predictor(im) 
    outputs = outputs["sem_seg"].argmax(dim=0)
    out_mask = outputs.to("cpu").numpy()
    out_ = out_mask.astype(int)
    output_list.append(out_)

    
    """
    # Visualize gt & predicted img
    visualizer = Visualizer(im[:, :, ::-1],
                            metadata=MetadataCatalog.get('flood_val'),
                            scale=1,
                            instance_mode=ColorMode.SEGMENTATION
    )
    vis_mask = visualizer.draw_sem_seg(outputs.to("cpu"), area_threshold=None, alpha=0.8)
    
    utils_.visualize(
        image = im,
        predicted_mask = vis_mask.get_image()[:, :, ::-1]
    )
    """
output_list = np.asarray(output_list)
name_list = np.asarray(name_list)
out_reshaped = output_list.reshape()
np.savetxt("../FEMA_Claims/CV_name.txt", name_list, fmt='%s')