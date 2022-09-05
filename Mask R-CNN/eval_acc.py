# import some common libraries
import os
from utils import utils

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.projects import point_rend
from detectron2.evaluation import SemSegEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

######
train_img_dir = './data/img/'
train_mask_dir = './data/mask/'
utils_ = utils(train_img_dir, train_mask_dir)
dataset_train_dicts, dataset_val_dicts = utils_.register()

# Revoke Evaluation for mAP and ACC
cfg = get_cfg()

# Add PointRend-specific config
point_rend.add_pointrend_config(cfg)

# Load a config from file
cfg.merge_from_file("detectron2/projects/PointRend/configs/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes.yaml")
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(utils_.class_ids)
cfg.MODEL.POINT_HEAD.NUM_CLASSES = len(utils_.class_ids)
cfg.OUTPUT_DIR="./outputs/flood_1/"
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
#cfg.INPUT.COLOR_AUG_SSD = False

predictor = DefaultPredictor(cfg)
evaluator = SemSegEvaluator('flood_val')
val_loader = build_detection_test_loader(cfg, 'flood_val')
print(inference_on_dataset(predictor.model, val_loader, evaluator))