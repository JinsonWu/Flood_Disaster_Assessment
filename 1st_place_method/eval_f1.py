import os
import cv2
import numpy as np
import glob
from sklearn.model_selection import train_test_split

dir_name = './res50_2_csv/'

all_files = []
for d in sorted(os.listdir(dir_name)):
    all_files.append(dir_name+d)

train_dicts, dataset_dicts = train_test_split(all_files, test_size=0.2)

cnt_eval = 1
class0, class1, class2, class3 = 0, 0, 0, 0
TP, TN, FP, FN = 0, 0, 0, 0
print('****************************************')
print('Start evaluating F1 Score')
print('Total Dataset Length (Samples): ', len(dataset_dicts))

for d in dataset_dicts: # shuffle the validation dataset

    #print('F1 Score Evaluation on Epoch: %d' % cnt_eval)
    #cnt_eval += 1

    # Load img
    out_mask = np.genfromtxt(d)
    gt_mask = cv2.imread(d.replace(dir_name, './post_msk/').replace('_pre_', '_post_').replace('.csv', '.png'))
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
