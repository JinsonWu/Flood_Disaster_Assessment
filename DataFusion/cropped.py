# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import os
import numpy as np
import rasterio as rio
from rasterio import features
from rasterio.mask import mask
import json

#pre  "BoundingBox(left=244089.6, bottom=3285341.2, right=252879.6, top=3297959.2)"
#post "BoundingBox(left=244092.4, bottom=3285340.8, right=252882.4, top=3297958.8)"

"""
img_pre = rio.open('./Area2_pre/Area2_4-3-2017_Ortho_ColorBalance.tif')
img_post = rio.open('./Area2_post/Area2_9-2-2017_Ortho_ColorBalance.tif')
#print(img.bounds)

with img_pre as src, img_post as shaped:
    src_affine = src.meta.get('transform')
    band = src.read(1)
    band[np.where(band!=src.nodata)] = 1
    geo_ = []
    for geometry, raster_value in features.shapes(band, transform=src_affine):
        if (raster_value == 1):
            geo_.append(geometry)
    
    out_img, out_transform = mask(dataset=shaped, shapes=geo_, crop=True)
    with rio.open('Area2_post_reshaped.tif', 'w', driver='GTiff', height=out_img.shape[1],
                  width=out_img.shape[2], count=src.count, dtype=out_img.dtype, transform=out_transform) as dst:
        dst.write(out_img)
    


"""

#
img = cv2.imread('Area2_black_masked.png')
height, width, channels = img.shape
pth = './Area2_fp/'
if (os.path.isdir(pth) == False):
    os.mkdir(pth)
os.chdir(pth)
x = 1024
y = 1024
cnt = 0
boundary_list = []  # left, right, top, bottom

for i in range(int(height/x)+1):
    for j in range(int(width/x)+1):
        if (i == (int(height/x)) and j == (int(width/x))):
            img_ = img[height-x:height, width-x:width]
            boundary_list.append([width-x, width, height-x, height])
        elif (i == (int(height/x))):
            img_ = img[height-x:height, j*x:(j+1)*x]
            boundary_list.append([j*x, (j+1)*x, height-x, height])
        elif (j == (int(width/x))):
            img_ = img[i*x:(i+1)*x, width-x:width]
            boundary_list.append([width-x, width, i*x, (i+1)*x])
        else: 
            img_ = img[i*x:(i+1)*x, j*x:(j+1)*x]
            boundary_list.append([j*x, (j+1)*x, i*x, (i+1)*x])
        cnt += 1
        #print(j, i)
        #cv2.imwrite('Area2_fp_'+str(cnt)+'.png', img_)
boundary_list = np.asarray(boundary_list)
print(boundary_list.shape)
np.savetxt("./boundary_list.txt", boundary_list, delimiter=',')
print("Done!")
