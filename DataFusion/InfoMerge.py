import json
import geopandas as gpd
import numpy as np
import os
import cv2
import time
import glob
from shapely.geometry import Polygon

def open_geojson(fname):
    return gpd.read_file(fname)

def open_json(fname):
    with open(fname) as f:
        fload = json.load(f)
    return fload

def coor_value(li):
    return np.max(list(zip(*li))[0]), np.min(list(zip(*li))[0]), np.max(list(zip(*li))[1]), np.min(list(zip(*li))[1])

def specify_region(fname):
    df_tx = open_geojson(fname)
    df_tx = df_tx.to_crs(3857)

    #df_tx_1 = df_tx[df_tx['release'] == 1]
    df_tx_2 = df_tx[df_tx['release'] != 1]

    #df_tx_json = df_tx_1.to_json()
    df_tx_json = df_tx_2.to_json()
    parsed = json.loads(df_tx_json)

    #outjson = 'release1.json'
    outjson = 'release2_3857.json'
    with open(outjson, 'w', encoding='utf-8') as fp:
        json.dump(parsed, fp)

def base_coor(fname):
# set up the targeted region with geojson file
    shp_file = open_json(fname)

    list_shp = []
    for feat in shp_file['features']:
        list_shp.append(feat['geometry']['coordinates'])
    shp_arr = np.asarray(list_shp)
    shp_arr = np.reshape(shp_arr, (shp_arr.shape[2], shp_arr.shape[3]))
    return shp_arr

def img_with_cv_output(fname):
    dataset_dicts = glob.glob(fname)
    li = []
    for fn in sorted(dataset_dicts):
        ptr1 = fn.find('cropped_') + 8
        ptr2 = fn.find('.csv')
        num = int(fn[ptr1:ptr2])
        li.append(num-1)
    return li

def select_tile(n):
    if (n == 1):
        return 0, 1024, 0, 1024
    elif (n == 2):
        return 1024, 2048, 0, 1024
    elif (n == 3):
        return 2048, 3072, 0, 1024
    elif (n == 4):
        return 0, 1024, 1024, 2048
    elif (n == 5):
        return 1024, 2048, 1024, 2048
    elif (n == 6):
        return 2048, 3072, 1024, 2048
    elif (n == 7):
        return 0, 1024, 2048, 3072
    elif (n == 8):
        return 1024, 2048, 2048, 3072
    elif (n == 9):
        return 2048, 3072, 2048, 3072
    else: 
        print("Wrong Tile Number!! Tile Number: ", n)
        return 0, 0, 0, 0

def select_bound(num):
    edge = 14
    if (num == 0):
        left_top_ = bound_cond[num]
        right_bottom_ = bound_cond[num+2*(edge+1)]  # [num+26]
        cutting_tile = 1
    elif (num == edge-1):
        left_top_ = bound_cond[num-2]
        right_bottom_ = bound_cond[num+2*edge]
        cutting_tile = 3
    elif (num == len(bound_cond)-edge):
        left_top_ = bound_cond[num-2*edge]
        right_bottom_ = bound_cond[num+2]
        cutting_tile = 7
    elif (num == len(bound_cond)-1):
        left_top_ = bound_cond[num-2*(edge+1)]
        right_bottom_ = bound_cond[num]
        cutting_tile = 9
    elif (num // edge == 0):
        left_top_ = bound_cond[num-1]
        right_bottom_ = bound_cond[num+2*edge+1]
        cutting_tile = 2
    elif (num // edge == (len(bound_cond) / edge) - 1):
        left_top_ = bound_cond[num-2*edge-1]
        right_bottom_ = bound_cond[num+1]
        cutting_tile = 8
    elif (num % edge == 0):
        left_top_ = bound_cond[num-edge]
        right_bottom_ = bound_cond[num+edge+2]
        cutting_tile = 4
    elif (num % edge == edge - 1):
        left_top_ = bound_cond[num-edge-2]
        right_bottom_ = bound_cond[num+edge]
        cutting_tile = 6
    else:
        left_top_ = bound_cond[num-edge-1]
        right_bottom_ = bound_cond[num+edge+1]
        cutting_tile = 5
    return left_top_, right_bottom_, cutting_tile

def footprint_list(parsed):
    list_release2 = []
    for feat in parsed['features']:
        list_release2.append(feat['geometry']['coordinates'])

    release2_arr = np.asarray(list_release2, dtype=object).flatten()

    fp_list = []
    for i in range(release2_arr.shape[0]):
        fp_list.append(np.asarray(release2_arr[i]))
    return fp_list

def footprint_inrange(fp_list, left_bound, right_bound, top_bound, bottom_bound):
    fp_list_selected = []
    for i in range(len(fp_list)):
        right_, left_, top_, bottom_ = coor_value(fp_list[i])
        if (left_ > left_bound and right_ < right_bound and top_ < top_bound and bottom_ > bottom_bound):
            fp_list_selected.append(fp_list[i])
    return fp_list_selected

def claim_(fname, claim_fname):
    claim_df = open_geojson(fname)
    claim_df = claim_df.to_crs(3857)
    claim_df.to_file(claim_fname, driver='GeoJSON')
    return open_json(claim_fname)

def merge_(ori_img, left_bound, right_bound, top_bound, bottom_bound, claim_json, fp_list_selected, cv_output, cutting_tile):
    #base_img = ori_img[int(left_top_[2]):int(right_bottom_[3]), int(left_top_[0]):int(right_bottom_[1])]
    base_img = np.zeros((int(right_bottom_[1])-int(left_top_[0]), int(right_bottom_[3])-int(left_top_[2])))

    long_ = right_bound - left_bound
    lat_ = top_bound - bottom_bound
    #center_x = left_bound + long_/2
    #center_y = bottom_bound + lat_/2
    shift_x = 1#.01
    shift_y = 1#.01
    x_ = base_img.shape[1]
    x_ratio = x_/long_
    y_ = base_img.shape[0]
    y_ratio = y_/lat_
    #alpha = 0.5
    thres_ = 0.5
    list_poly = []
    flooded_no_claim = 0
    unflooded_no_claim = 0

    # Need to project the coordinates from the center to normalize the curved distance caused by geographic records
    for i in range(len(fp_list_selected)):
        claim_cnt = 0
        claim_xmax, claim_xmin, claim_ymax, claim_ymin = coor_value(fp_list_selected[i])
        while claim_cnt < len(claim_json['features']):
            if (claim_json['features'][claim_cnt]['geometry']['coordinates'][0] <= claim_xmax and
                claim_json['features'][claim_cnt]['geometry']['coordinates'][0] >= claim_xmin and
                claim_json['features'][claim_cnt]['geometry']['coordinates'][1] <= claim_ymax and
                claim_json['features'][claim_cnt]['geometry']['coordinates'][1] >= claim_ymin):
                break
            else:
                claim_cnt += 1

        poly = []
        for coor in fp_list_selected[i]:
            x = (coor[0] - left_bound) - 28
            y = (top_bound - coor[1]) + 95
            """
            if coor[0] < center_x:
                x = (center_x - ((center_x - coor[0] - long_/300)*shift_x)) - left_bound
            else:
                x = (center_x + ((coor[0] - center_x - long_/300)*shift_x)) - left_bound
            if coor[0] < center_x:
                y = top_bound - (center_y - ((center_y - coor[1] - lat_/300)*shift_y))
            else:
                y = top_bound - (center_y + ((coor[1] - center_y - lat_/300)*shift_y))
            """

            pair = [x*x_ratio, y*y_ratio]
            poly.append(pair)
        
        polygon = Polygon(poly)
        int_coords = lambda x: np.array(x).round().astype(np.int32)
        exterior = [int_coords(polygon.exterior.coords)]
        ex_xmax, ex_xmin, ex_ymax, ex_ymin = coor_value(np.reshape(np.asarray(exterior), (np.shape(exterior)[1], np.shape(exterior)[2])))
        x_min_bound, x_max_bound, y_min_bound, y_max_bound = select_tile(cutting_tile)
        if (((ex_xmax > x_min_bound and ex_xmax < x_max_bound) and (ex_ymax > y_min_bound and ex_ymax < y_max_bound)) or
            ((ex_xmax > x_min_bound and ex_xmax < x_max_bound) and (ex_ymin > y_min_bound and ex_ymin < y_max_bound)) or
            ((ex_xmin > x_min_bound and ex_xmin < x_max_bound) and (ex_ymax > y_min_bound and ex_ymax < y_max_bound)) or
            ((ex_xmin > x_min_bound and ex_xmin < x_max_bound) and (ex_ymin > y_min_bound and ex_ymin < y_max_bound))):
            if (ex_xmin < x_min_bound):
                ex_xmin = x_min_bound
            if (ex_xmax > x_max_bound):
                ex_xmax = x_max_bound
            if (ex_ymin < y_min_bound):
                ex_ymin = y_min_bound
            if (ex_ymax > y_max_bound):
                ex_ymax = y_max_bound

            sel = base_img[ex_ymin:ex_ymax, ex_xmin:ex_xmax]
            cnt_flood = 0
            for j in range(sel.shape[0]):
                for k in range(sel.shape[1]):
                    # 1. -> Water/Flood, 3. -> Flooded Building
                    if (cv_output[ex_xmin-x_min_bound+k][ex_ymin-y_min_bound+j] == 3. or
                        cv_output[ex_xmin-x_min_bound+k][ex_ymin-y_min_bound+j] == 1.):
                        cnt_flood += 1

            if ((cnt_flood / sel.flatten().shape[0]) > thres_):
                if (claim_cnt != len(claim_json['features'])):
                    list_poly.append({"geometry": Polygon(fp_list_selected[i]), "Flooded/Unflooded": True, 
                                    "t_dmg_bldg": claim_json['features'][claim_cnt]['properties']['t_dmg_bldg'], 
                                    "pay_bldg": claim_json['features'][claim_cnt]['properties']['pay_bldg']})
                else:
                    list_poly.append({"geometry": Polygon(fp_list_selected[i]), "Flooded/Unflooded": True, 
                                    "t_dmg_bldg": 0.0, "pay_bldg": 0.0})
                    flooded_no_claim += 1
            else:
                if (claim_cnt != len(claim_json['features'])):
                    list_poly.append({"geometry": Polygon(fp_list_selected[i]), "Flooded/Unflooded": False, 
                                    "t_dmg_bldg": claim_json['features'][claim_cnt]['properties']['t_dmg_bldg'], 
                                    "pay_bldg": claim_json['features'][claim_cnt]['properties']['pay_bldg']})
                else:
                    list_poly.append({"geometry": Polygon(fp_list_selected[i]), "Flooded/Unflooded": False, 
                                    "t_dmg_bldg": 0.0, "pay_bldg": 0.0})
                    unflooded_no_claim += 1
        
        cv2.fillPoly(base_img, exterior, color=(255, 255, 0))
    base_img = base_img[x_min_bound:x_max_bound, y_min_bound:y_max_bound]
    #cv2.addWeighted(upper_mask, alpha, base_img, 1 - alpha, 0, base_img)
    #cv2.imwrite('./Area2_post_masked_projected_58/Area2_post_masked_58_1.png', base_img)
    return list_poly, unflooded_no_claim, flooded_no_claim

def building_info(df, unflooded_no_claim_all, flooded_no_claim_all):
    array_ = np.asarray(df.iloc[:, 1])
    print("#####################################################################")
    print("Total Building Percentage in this Region: ",
        len(df))
    print("Unflooded Building in this Region: ",
        np.bincount(array_.astype(int))[0])
    print("Flooded Building Percentage in this Region: ",
        np.bincount(array_.astype(int))[1])
    print("Unflooded Building Percentage in this Region: ",
        np.bincount(array_.astype(int))[0] / len(df))
    print("Flooded Building Percentage in this Region: ",
        np.bincount(array_.astype(int))[1] / len(df))
    print("Unflooded Building with Claims: ", np.bincount(array_.astype(int))[0] - unflooded_no_claim_all)
    print("Unflooded Building without Claims: ", unflooded_no_claim_all)
    print("Flooded Building with Claims: ", np.bincount(array_.astype(int))[1] - flooded_no_claim_all)
    print("Flooded Building without Claims: ", flooded_no_claim_all)
    print("#####################################################################")

#### main program

start = time.time()

#specify_region('Texas.geojson')

# record the buildings with aligned timeline (release=2)
print("Loading Building Info......")
parsed = open_json('release2_3857.json')
#
print("Loading Region Boundary Info......")
shp_arr = base_coor('../Harvey_AOI1_3857.geojson')
#
a = list(zip(*shp_arr))
#
print("Loading Footprint Info......")
fp_list = footprint_list(parsed)


# select the targeted polygon within the region
ori_img_filename = 'Area1_post_reshaped.tif'
ori_img = cv2.imread(ori_img_filename)

bound_cond = np.genfromtxt('/home/staff/j/jinsonwu/Flood_Disaster_Assessment/FEMA_Claims/bound_1.csv', delimiter=',')

print("Collecting Claim Info......")
claim_json = claim_("nifp_harvey.shp", "nifp_harvey.geojson")

print("Merging Features......")
cv_output_list = img_with_cv_output('/home/staff/j/jinsonwu/Flood_Disaster_Assessment/FEMA_Claims/best_csv/Area1*')
cutting_tile = -1   # record the tile which should be cut in the 9-tile maze
list_poly_all = []
unflooded_no_claim_all = 0
flooded_no_claim_all = 0
cnt = 1
print('Total Images Required to Merge: ', len(cv_output_list))
for num in cv_output_list:
    time_s = time.time()
    print('Merging Image No.', cnt)
    # load cv output
    cv_fname = '/home/staff/j/jinsonwu/Flood_Disaster_Assessment/FEMA_Claims/best_csv/Area1_post_cropped_' + str(num+1) + '.csv'
    cv_output = np.genfromtxt(cv_fname)#, delimiter=',')
    # create boundary
    # obtain all relative tile location in the maze
    left_top_, right_bottom_, cutting_tile = select_bound(num)
    
    #left_top_ = bound_cond[44] # 58-13-1 = 44
    #right_bottom_ = bound_cond[70] # 58+13-1 = 70
    left_bound = (left_top_[0] / ori_img.shape[1]) * (np.max(a[0]) - np.min(a[0])) + np.min(a[0])
    right_bound = (right_bottom_[1] / ori_img.shape[1]) * (np.max(a[0]) - np.min(a[0])) + np.min(a[0])
    top_bound = np.max(a[1]) - (left_top_[2] / ori_img.shape[0]) * (np.max(a[1]) - np.min(a[1]))
    bottom_bound = np.max(a[1]) - (right_bottom_[3] / ori_img.shape[0]) * (np.max(a[1]) - np.min(a[1]))

    #print("Filtering Footprint Data......")
    fp_list_selected = footprint_inrange(fp_list, left_bound, right_bound, top_bound, bottom_bound)
    
    list_poly, unflooded_no_claim, flooded_no_claim = merge_(ori_img, left_bound, right_bound, top_bound,
                                                            bottom_bound, claim_json, fp_list_selected, cv_output, cutting_tile)

    list_poly_all += list_poly
    unflooded_no_claim_all += unflooded_no_claim
    flooded_no_claim_all += flooded_no_claim_all
    cnt += 1
    print("Single Batch Execution Time: ", time.time()-time_s, "sec")
    
df_all = gpd.GeoDataFrame(list_poly_all)
df_all.fillna(0, inplace=True)
df_all.set_crs('epsg:3857')
df_all.to_file("df_Area1_post_cropped_all.geojson", driver='GeoJSON')
df_all.to_csv("df_Area1_post_cropped_all.csv")

building_info(df_all, unflooded_no_claim_all, flooded_no_claim_all)

print("Task Completed!! Total Execution Time: ", time.time()-start, "sec")