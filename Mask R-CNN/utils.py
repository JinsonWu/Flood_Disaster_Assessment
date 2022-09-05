# import some common libraries
import numpy as np
import os
import cv2
import sys
import json
import glob
import timeit
import random
import pandas as pd
import matplotlib.pyplot as plt
import requests
from PIL import UnidentifiedImageError
from PIL import Image
from skimage import io
from multiprocessing import Pool
from shapely.wkt import loads
from shapely.geometry import mapping, Polygon
from fvcore.common.file_io import PathManager
from skimage.morphology import square, dilation, watershed, erosion

# import some common detectron2 utilities
from detectron2.data import MetadataCatalog, DatasetCatalog

# Environ setting
np.random.seed(1)
random.seed(1)
sys.setrecursionlimit(10000)

######### 
class utils():

    def __init__(self, train_img_dir, train_mask_dir) -> None:
        super(utils, self).__init__()
        # dmg classes
        self.damage_dict = {
                        "Background": 0,
                        "Water/Flood": 1,
                        "Non-Flooded Building": 2,
                        "Flooded Building": 3,
        }
        # class ids
        self.class_ids = [
                        "Background",
                        "Water/Flood",
                        "Non-Flooded Building",
                        "Flooded Building",
        ]

        # fetch pth from each data classes
        img_pth = [os.path.join(img_id) for img_id in sorted(glob.glob(train_img_dir+'*.png'))]
        mask_pth = [os.path.join(img_id) for img_id in sorted(glob.glob(train_mask_dir+'*.png'))]
        

        # create dataframe for pre dataset
        pth_df = pd.DataFrame()
        pth_df['img_pth'] = img_pth
        pth_df['mask_pth'] = mask_pth

        # Shuffling dataset
        pth_df = pth_df.sample(frac=1).reset_index(drop=True)

        # Perform 80/20 split for train / val for pre, post dataset
        self.val_df = pth_df.sample(frac=0.2, random_state=42)
        self.train_df = pth_df.drop(self.val_df.index)

    def func(self, train_or_val):
        results = []
        if(train_or_val == 'train'):
            image_paths = self.train_df['img_pth'].tolist()
            mask_paths = self.train_df['mask_pth'].tolist()
        else:
            image_paths = self.val_df['img_pth'].tolist()
            mask_paths = self.val_df['mask_pth'].tolist()

        counter = 0
        for image_file, gt_file in zip(image_paths, mask_paths):

            # PIL Image is efficient as it doesn't read image to retrieve size
            im = Image.open(image_file)
            width, height = im.size
            record = {}
            record["file_name"] = image_file
            record["image_id"] = counter
            record["sem_seg_file_name"] = gt_file
            record["height"] = height
            record["width"] = width
            counter = counter + 1
            results.append(record)
            

            assert len(results),  f"No images found in {image_file}!"
            assert PathManager.isfile(results[0]["sem_seg_file_name"]
            ), "Passed"  # noqa

        return results

    def visualize(self, **images):
        """
        Plot images in one row
        """
        n_images = len(images)
        plt.figure(figsize=(20,8))
        for idx, (name, image) in enumerate(images.items()):
            plt.subplot(1, n_images, idx + 1)
            plt.xticks([]); 
            plt.yticks([])
            # get title from the parameter names
            plt.title(name.replace('_',' ').title(), fontsize=20)
            plt.imshow(image)
        plt.show()

    def register(self):

        # Register pre & post dataset

        for d in ['train', 'val']:
            DatasetCatalog.register('flood_' + d, lambda d=d: self.func(str(d)))
            MetadataCatalog.get('flood_' + d).set(stuff_color=[], 
                                                    stuff_classes=self.class_ids, 
                                                    evaluator_type=['sem_seg'],
                                                    stuff_dataset_id_to_contiguous_id=self.damage_dict,
                                                    ignore_label=[],
                                                    )

        dataset_train_dicts = self.func('train')
        dataset_val_dicts = self.func('val')

        return dataset_train_dicts, dataset_val_dicts


class create_mask():
    def __init__(self, masks_dir, train_dir) -> None:
        super(create_mask, self).__init__()
        self.damage_dict = {
                        "Background": 0,
                        "Water/Flood": 1,
                        "Non-Flooded Building": 2,
                        "Flooded Building": 3,
                    }
        self.masks_dir = masks_dir
        self.train_dir = train_dir

    def mask_for_polygon(poly, im_size=(1024, 1024)):
        img_mask = np.zeros(im_size, np.uint8)
        int_coords = lambda x: np.array(x).round().astype(np.int32)
        exteriors = [int_coords(poly.exterior.coords)]
        interiors = [int_coords(pi.coords) for pi in poly.interiors]
        cv2.fillPoly(img_mask, exteriors, 1)
        cv2.fillPoly(img_mask, interiors, 0)
        return img_mask

    def process_image(self, json_file):
        js1 = json.load(open(json_file))
        js2 = json.load(open(json_file.replace('_pre_disaster', '_post_disaster')))

        msk = np.zeros((1024, 1024), dtype='uint8')
        msk_damage = np.zeros((1024, 1024), dtype='uint8')

        for feat in js1['features']['xy']:
            poly = loads(feat['wkt'])
            _msk = self.mask_for_polygon(poly)
            msk[_msk > 0] = 255

        for feat in js2['features']['xy']:
            poly = loads(feat['wkt'])
            subtype = feat['properties']['subtype']
            _msk = self.mask_for_polygon(poly)
            msk_damage[_msk > 0] = self.damage_dict[subtype]

        cv2.imwrite(json_file.replace('/labels/', '/masks/').replace('_pre_disaster.json', '_pre_disaster.png'),
                    msk, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        cv2.imwrite(json_file.replace('/labels/', '/masks/').replace('_pre_disaster.json', '_post_disaster.png'),
                    msk_damage, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    def create_mask(self):
        t0 = timeit.default_timer()

        all_files = []
        for d in self.train_dirs:
            os.makedirs(os.path.join(d, self.masks_dir), exist_ok=True)
            for f in sorted(os.listdir(os.path.join(d, 'images'))):
                if '_pre_disaster.png' in f:
                    all_files.append(os.path.join(d, 'labels', f.replace('_pre_disaster.png', '_pre_disaster.json')))


        with Pool() as pool:
            _ = pool.map(self.process_image, all_files)

        elapsed = timeit.default_timer() - t0
        print('Time: {:.3f} min'.format(elapsed / 60))

"""
class create_mask_flood():
    def __init__(self, masks_dir, train_dir) -> None:
        super(create_mask_flood, self).__init__()
        self.damage_dict = {
                        "Background": 0,
                        "Water/Flood": 1,
                        "Non-Flooded Building": 2,
                        "Flooded Building": 3,
                    }
        self.masks_dir = masks_dir
        self.train_dir = train_dir

    def open_json(self, path):
        with open(path) as file:
            return json.load(file)

    def open_img(self, url):
        try:
            return Image.open(requests.get(url, stream=True).raw)
        except UnidentifiedImageError:
            return None

    def mask_for_polygon(poly, im_size=(1024, 1024)):
        img_mask = np.zeros(im_size, np.uint8)
        int_coords = lambda x: np.array(x).round().astype(np.int32)
        exteriors = [int_coords(poly.exterior.coords)]
        interiors = [int_coords(pi.coords) for pi in poly.interiors]
        cv2.fillPoly(img_mask, exteriors, 1)
        cv2.fillPoly(img_mask, interiors, 0)
        return img_mask

    def process_image(self, json_file):
        js = self.open_json(json_file)
        print(json_file)
        msk = np.zeros((1024, 1024), dtype='uint8')

        for feat in js['Label']['objects']:
            img = self.open_img(feat['instanceURI'])
            print(feat)
            poly = feat['polygon']
            dmg = feat['title']
            _msk = self.mask_for_polygon(poly)
            msk[_msk > 0] = self.damage_dict[dmg]

        cv2.imwrite(json_file.replace('/json/', '/mask/').replace('.json', '.png'),
                    msk, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    def create_mask(self):
        t0 = timeit.default_timer()

        all_files = []
        os.makedirs(self.masks_dir, exist_ok=True)
        for f in sorted(os.listdir(self.train_dir)):
            all_files.append(self.train_dir + f)

        with Pool() as pool:
            _ = pool.map(self.process_image, all_files)

        elapsed = timeit.default_timer() - t0
        print('Time: {:.3f} min'.format(elapsed / 60))
"""