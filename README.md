# Flood Disaster Assessment
Utilized computer vision algorithms to predict flooded/non-flooded buildings through satellite images. Attempted to figure out the relation between accompanying payments and flood extent. Purposed to compose an end-to-end map by following data fusion and feature extraction with DeepCovidNet based on DeepFN.

## Data Preprocessing
This section includes pre-works of computer vision models and preprocessing methods dealing with the original satellite image. Primary distribution is to wrap up large-scale imagery into small pieces (1024x1024) for following annotation work and computer vision training.

### Satellite Image Correction
Due to discrepancies in geographic coordinates (latitude and longitude), pre-disaster and post-disaster images were ranged with pretty different regions. The biggest difference was about 100 pixel-wide between pre and post images. Therefore, the first priority was to adjust the coordinates to make them indicate the identical area. I utilized rasterio to capture overlapping areas to crop, which made the two large images have the same sections.

### Split and Annotation
After the correction, the images should be split into small pieces (1024x1024 in this project) to be annotated. There existed several constraints that could not completely clip images with totally unique content in each sample. Some of them contain overlapping objects in the edges, but they
do not affect the following labeling and training process.

## Computer Vision Implementation
It introduces all the devotion toward computer vision models we implemented. Primary work referred to Detectron2* and 1st place method of xview2 competition**. We also performed a data augmentation strategy to address the imbalanced class distribution. The first several parts will demonstrate the input formation and data augmentation, followed by the training procedure of computer vision models and their evaluations.

*https://github.com/facebookresearch/detectron2  
**https://github.com/vdurnov/xview2_1st_place_solution

### Mask Creation
According to the labeling classes, we had four groups of tags on our images, respectively background, water/flood, non-flooded building, and flooded building. To provide ground truth for subsequent training, they should be converted to arrays with exactly the same to the cropped images we produced previously. They were composed of 1024x1024x3 pixels per image, and might be processed to be two-dimensional arrays if required. This conversion was based on the JSON file we downloaded from LabelBox. It could directly create masks without preliminary processing.

### Data Augmentation
Due to imbalanced class distribution (0.125% initially for flooded building), a data augmentation strategy was necessary to address this problem. I referred to a paper concerning horizontal comparison with data augmentation methods[1][2]. “VHMixUP” was the one with the lowest error rate in their experiment. Therefore, I adopted this mechanism to increase the flooded building class in our dataset. It in return gave us an pixel-wisely increase of 2% (2.22% eventually).

[1] Cecilia Summers and Michael J. Dinneen. Improved Mixed-Example Data Augmentation. 2018. DOI:
10.48550/ARXIV.1805.11272. URL: https://arxiv.org/abs/1805.11272  
[2] Shorten, C., Khoshgoftaar, T.M. A survey on Image Data Augmentation for Deep Learning. J Big Data 6, 60
(2019). https://doi.org/10.1186/s40537-019-0197-0

### Model Adoption and Modifications
I ultimately chose five models to perform the computer vision training for the sake of ablation study, Mask R-CNN, ResNet34_Double_Unet, Squeeze & Excitation ResNext34x4D, Dual Path Network Double, and Squeeze & Excitation Network_154. They all pretrained with ImageNet in the beginning. I further pretrained them with xBD dataset with at least 50 epochs to strengthen their robustness toward damaged buildings. Detailed modifications with each model will be demonstrated in the following section.

1. Mask R-CNN

2. ResNet34_Double_Unet

3. Squeeze & Excitation ResNext32x4D

4. Dual Path Network_Double

5. Squeeze & Excitation Network_154

### Evaluation and Afterwards Adjustment  
Below is the F1-Score comparison between models*. Squeeze & Excitation ResNext32x4D obtained the best performance. Since the output from it required further tasks to determine classes in each image (it produced rgb images, which indicated the assigned colors, not classes), I designed an application to define what’s the represented class in each pixel. However, it seemed to have overfitting that impacted the subsequent pixel segmentation prediction.
| Models | F1-Score |
| --- | --- |
| Mask R-CNN | 0.89655344 |
| ResNet34_Double_Net | 0.92077386 |
| Squeeze & Excitation ResNext32x4D | 0.96468256 |
| Dual Path Network_Double | 0.89946563 |
| Squeeze & Excitation Network_154 | 0.94684531 |

### Revised Training & Performance
To comply with the referenced methods, I retrained the best model from 1st place method (SE ResNext50) and Mask R-CNN. The input format was modified to be a siamese-like structure. Models underwent pretraining with xBD dataset, followed by training on our own dataset accompanying the data augmentation mentioned above. Parameters were identical with previous training. The F1-Score was shown in the following table, it was calculated without the class of water/flood due to unknown problems in training.

| Model | F1-Score |
| --- | --- |
| Mask R-CNN | 0.77499838 |
| SE ResNext50x4D Seed 0 | 0.85413800 |
| SE ResNext50x4D Seed 1 | 0.86095552 |
| SE ResNext50x4D Seed 2 | 0.86635134 |


## Feature Integration
It demonstrates how I integrated the computer vision output, Microsoft footprints (https://github.com/microsoft/USBuildingFootprints), and claim data. It also consists of plenty of methods attempting to align the polygons and points in different data, then results will be exhibited. Some directions of alignment optimization will be introduced in the last part. 


### Microsoft Texas Footprint Building Polygon Selection
Microsoft Texas footprint data was composed of millions of building polygons. According to the latitude and longitude provided by the initial satellite image, we could acquire the polygon coordinates that were inside our target regions. There were two major coordinate systems, which were “crs: 3857” and “crs: 4326“. 3857 would be a better one for us because it was created by meter so that we could better convert it to pixels.

### Footprint Alignment with Satellite Image
Distortion was unavoidable because discrepancies would be accumulated while reflecting absolute geographic coordinates on two-dimensional planes. The biggest gap was about five-building wide in the edge of imagery. I attempted direct shifting, scaling (zoom-in & out), and projection from the center of images. Unfortunately, none of these could address this problem. Eventually, we chose to align the polygons in a small sample scale, which was 1024x1024, to assure accurate locations of buildings. Besides, some polygons would be lost if they were on the boundaries of small pieces. I adopted a
maze-like strategy to keep all information on the image based on the relative location of a nine-image maze.

### Adjustment to Align Footprint Data and Computer Vision Output
With the assistance of maze-like strategy and alignment correction, we could firstly apply direct shifting to have preliminary aligned images. However, tiles required strict rules to be placed in desired regions. Relative boundaries, conversion between polygons and pixels, and threshold that determines classes were performed to improve alignment quality.

### Feature Integration with Claim Data and Computer Vision Output
The claim data was composed of data points. In case that directly converted polygons would cause data corruption, I decided to merge polygons, data points, and computer vision output with a method that did not change their intrinsic information. This accordingly took a plethora of time to complete integration. Additionally, we found that some building footprints were duplicated. It was resolved by correcting all the features’ uniqueness.

### Data Fusion Optimizations
Recently, we realized that the claim data was not totally aligned with the footprints. This led to severe problems that lost precious information and put risk on our following end-to-end map training. Perhaps, another data collection is expected to recover this data defect. In addition, alignment with footprint itself has much of a gap to be improved. Direct shifting has displacement to some extent even if we align it with a small sample size. One possible way is to locate buildings through computer vision outputs. Taking the first and last building on the image as anchors to range polygons enables better alignment.

*Dataset and some details of the project are under the constraint of NDA with Urban Resilience Lab @ Texas A&M University. Implementation will be updated once the paper is successfully published*

## Contact Info
Author: Chun-Sheng Wu, MS student in Computer Engineering @ Texas A&M University  
Email: jinsonwu@tamu.edu  
LinkedIn: https://www.linkedin.com/in/chunshengwu/
