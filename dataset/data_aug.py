import os
import numpy as np
import matplotlib.pyplot as plt
import nnabla as nn
from nnabla.ext_utils import get_extension_context
from nnabla.utils.image_utils import imread, imresize
from nnabla.ext_utils import get_extension_context
import random

from MixedDataLearning import *

img_flood = [2, 26, 57, 58, 59, 60, 69, 70, 71, 72,]
img_pth = './post_img/'
n = 1

while n < 51:
    print("Processing Img No. ", n)
    img1, img2 = random.choice(img_flood), random.choice(os.listdir(img_pth))
    input_img1 = img_pth+'Area2_post_cropped_'+str(img1)+'.png'
    input_img2 = img_pth+img2

    image1 = imread(input_img1, channel_first=True)[:3]
    image2 = imread(input_img2, channel_first=True)[:3]
    scale = float(image1.shape[1]) / image2.shape[1]
    image2 = imresize(image2, size=(int(image2.shape[2]*scale), int(image2.shape[1]*scale)), channel_first=True)

    larger_shape = [max(image1.shape[i], image2.shape[i]) for i in range(3)]
    pad_length_1 = [larger_shape[i] - image1.shape[i] for i in range(3)]
    pad_length_2 = [larger_shape[i] - image2.shape[i] for i in range(3)]

    image1 = np.pad(image1, (
                    (0, 0),
                    (pad_length_1[1] // 2, pad_length_1[1] // 2 + pad_length_1[1] % 2),
                    (pad_length_1[2] // 2, pad_length_1[2] // 2 + pad_length_1[2] % 2)),
                    mode="reflect")

    image2 = np.pad(image2, (
                    (0, 0),
                    (pad_length_2[1] // 2, pad_length_2[1] // 2 + pad_length_2[1] % 2),
                    (pad_length_2[2] // 2, pad_length_2[2] // 2 + pad_length_2[2] % 2)),
                    mode="reflect")

    #@title Choose data augmentation config.

    #@markdown Choose which data augmentation is used.
    mixtype = "vhmmixup"  #@param ['mixup', 'cutmix', 'vhmmixup']
    #@markdown choose alpha value. (default: 0.5)
    alpha = 1.04  #@param {type: "slider", min: 0.0, max: 2.0, step: 0.01}

    inshape = (2,) + image1.shape
    if mixtype == "mixup":
        mdl = MixupLearning(2, alpha=alpha)
    elif mixtype == "cutmix":
        mdl = CutmixLearning(inshape, alpha=alpha, cutmix_prob=1.0)
    else:
        # "vhmixup" is used.
        mdl = VHMixupLearning(inshape, alpha=alpha)

    image_train = nn.Variable(inshape)
    label_train = nn.Variable((2, 1))
    mix_image, mix_label = mdl.mix_data(image_train, F.one_hot(label_train, (2, )))
    image_train.d[0] = image1 / 255.
    image_train.d[1] = image2 / 255.

    mdl.set_mix_ratio()
    mix_image.forward()
    fname = input_img1.replace('.png', '_'+str(n)+'.png')
    plt.imsave(fname, mix_image.d[1].transpose(1,2,0))
    n += 1