# import some common libraries
import os
import random
import numpy as np
import cv2
from utils import utils
from fvcore.common.file_io import PathManager

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.projects import point_rend
from sklearn.model_selection import train_test_split


# Revoke Evaluation for F1-Score
# Initialize counters
TP = 0
TN = 0
FN = 0
FP = 0

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

cfg.OUTPUT_DIR="outputs/whole_dataset/"
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
#cfg.INPUT.COLOR_AUG_SSD = False


predictor = DefaultPredictor(cfg)
#dataset_dicts = dataset_val_dicts # load dataset from xview_post_val

dir_name = './pre_img/'

all_files = []
for d in sorted(os.listdir(dir_name)):
    all_files.append(dir_name+d)

train_dicts, dataset_dicts = train_test_split(all_files, test_size=0.2)
cnt_eval = 1
class0, class1, class2, class3 = 0, 0, 0, 0
print('****************************************')
print('Start evaluating F1 Score')
print('Total Dataset Length (Samples): ', len(dataset_dicts))

for d in dataset_dicts: # shuffle the validation dataset

    #print('F1 Score Evaluation on Epoch: %d' % cnt_eval)
    #cnt_eval += 1

    # Load img
    """
    im = cv2.imread(d["file_name"])
    gt_mask = cv2.imread(d["sem_seg_file_name"])
    gt_mask_flat = np.bincount(gt_mask.flatten())
    gt_mask_class_len = len(gt_mask_flat)
    """
    im = cv2.imread(d)
    gt_mask = cv2.imread(d.replace('/pre_img/', '/post_msk/').replace('_pre_', '_post_'))
    gt_mask_flat = np.bincount(gt_mask.flatten())
    gt_mask_class_len = len(gt_mask_flat)
    if (gt_mask_class_len > 0):
        class0 += gt_mask_flat[0]
    if (gt_mask_class_len > 1):    
        class1 += gt_mask_flat[1]
    if (gt_mask_class_len > 2):    
        class2 += gt_mask_flat[2]
    if (gt_mask_class_len > 3):    
        class3 += gt_mask_flat[3]

    # Produce predicted image
    outputs = predictor(im) 
    outputs = outputs["sem_seg"].argmax(dim=0)
    out_mask = outputs.to("cpu").numpy()

    """
    # Visualize gt & predicted img
    visualizer = Visualizer(im[:, :, ::-1],
                            metadata=MetadataCatalog.get('xview_post_val'),
                            scale=1,
                            instance_mode=ColorMode.SEGMENTATION
    )
    vis_mask = visualizer.draw_sem_seg(outputs.to("cpu"), area_threshold=None, alpha=0.8)
    
    utils_.visualize(
        image = im,
        predicted_mask = vis_mask.get_image()[:, :, ::-1]
    )
    """
    
    # Calculate TP, TN, FP, and FN
    for i in range(gt_mask.shape[0]):
        for j in range(gt_mask.shape[1]):
            k = gt_mask[i][j][0]
            # buildings (postives in ground truth)
            if (k > 1):  
                if (k == out_mask[i][j]): TP += 1
                else: FN += 1
            # background (negatives in ground truth)
            elif (k == 0):
                if (k == out_mask[i][j]): TN += 1
                else: FP += 1

print('Backgroung Count: ', class0)
print('Flood/Water Count: ', class1)
print('Non-Flooded Building Count: ', class2)
print('Flooded Building Count: ', class3)

print('TP: ', TP)
print('TN: ', TN)
print('FP: ', FP)
print('FN: ', FN)
Precision = ((TP)/(TP+FP))
Recall = ((TP)/(TP+FN))

f_score = 2*(Recall * Precision) / (Recall + Precision)
print('F1 Score on Validation Dataset: %.8f' % f_score)
print('****************************************')
