import cv2
import os
import numpy as np
import timeit

t0 = timeit.default_timer()
color_list = [[0, 0, 0], [255, 255, 0], [255, 0, 255], [255, 0, 0]]
dir_name = './res50cls_cce_2_tuned/'
out_dir = './res50_2_csv/'

os.makedirs(out_dir, exist_ok=True)

for f in sorted(os.listdir(dir_name)):
    img = cv2.imread(dir_name+f, cv2.IMREAD_COLOR)
    img_ = np.zeros((1024, 1024))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            cls = 0
            dis_ = (color_list[0][0] - img[i][j][0])**2 + (color_list[0][1] - img[i][j][1])**2 + (color_list[0][2] - img[i][j][2])**2
            
            for k in range(1, len(color_list)):
                b = (color_list[k][0] - img[i][j][0])**2
                g = (color_list[k][1] - img[i][j][1])**2
                r = (color_list[k][2] - img[i][j][2])**2
                if (b+g+r < dis_):
                    cls = k
                    dis_ = b+g+r
                    img_[i][j] = cls
    np.savetxt(out_dir+f.replace('.png', '.csv'), img_)
    
elapsed = timeit.default_timer() - t0
print('Time: {:.3f} min'.format(elapsed / 60))