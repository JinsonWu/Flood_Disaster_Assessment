from PIL import Image, ImageColor, ImageDraw
from PIL import UnidentifiedImageError
import requests
import os
import json

def manual_classes():
    """
    Change your preferenced color-coding below. 
    If you want to use manual coloring, you also need to change the Label-Classes (Title)
    """
    manual_dict = {
        'Background': 0,
        'Flood/Water': 1,
        'Non-Flooded Building': 2,
        'Flooded Building': 3,
    }
    return manual_dict

def open_json(path):
    with open(path) as file:
        return json.load(file)

def open_img(url):
    try:
        return Image.open(requests.get(url, stream=True).raw)
    except UnidentifiedImageError:
        return None

def color_extractor(data, color_coding):
    """takes the given dictionary part and extracts all needed information. returns also colors for 3 different types"""

    if color_coding == 'auto':
        color = ImageColor.getcolor(data['color'], 'RGB')
    elif color_coding == 'manual':
        color = (manual_classes()[data['title']], manual_classes()[data['title']], manual_classes()[data['title']],255)
    elif color_coding == 'binar':
        color = (255,255,255)
    else:
        print('no valid color-code detected - continue with binarized Labels.')
        color = (255,255,255)
    return color


def img_color(img, color):
    """change color of label accordingly"""
    if color == (255,255,255,255):
        return img
    img = img.convert('RGBA')
    width, height = img.size
    for x in range(width):
        for y in range(height):
            if img.getpixel((x,y)) == (255,255,255,255):
                img.putpixel((x,y), color)
    return img


def img_draw_polygon(size, polygon, color):
    """draw polygons on image"""
    img = Image.new('RGBA', size, (0,0,0,0))
    img = img.convert('RGBA')
    draw = ImageDraw.Draw(img)
    # read points
    points = []
    for i in range(len(polygon)):
        points.append((int(polygon[i]['x']),int(polygon[i]['y'])))
    draw.polygon(points, fill = (color))
    return img


def progressBar(current, total, barLength = 20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))

    print('Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')


json_pth = './json_show/'
output_pth = './mask_show/'
type = 'auto'
type_m = 'manual'

for json_file in os.listdir(json_pth):
    # create image list for Labels
    img_list = []
    js = open_json(json_pth+json_file)

    # read original image
    original_img = open_img(js['Labeled Data'])

    try:
        width, height = original_img.size
    except Exception:
        print('Original image data not callable. Please provide image width and height.')

    for i in range(len(js['Label']['objects'])):
    #for i in range(3):
        # read path and open image
        img = open_img(js['Label']['objects'][i]['instanceURI'])

        # if path is not readable try to read polygon-data-points
        if not img is None:
            if (js['Label']['objects'][i]['title'] == 'Flood/Water' or js['Label']['objects'][i]['title'] == 'Background'):
                img = img_color(img, color_extractor(js['Label']['objects'][i], type_m))
                img_list.append(img)
            else:
                img = img_color(img, color_extractor(js['Label']['objects'][i], type))
                img_list.append(img)
        else:
            if (js['Label']['objects'][i]['title'] == 'Flood/Water' or js['Label']['objects'][i]['title'] == 'Background'):
                img = img_draw_polygon((width,height), js['Label']['objects'][i]['polygon'], color_extractor(js['Label']['objects'][i], type_m))
                img_list.append(img)
            else:
                try:
                    # img = img_draw_polygon(img, data[0]['Label']['objects'][i]['polygon'], data[0]['Label']['objects'][i]['title'])
                    img = img_draw_polygon((width,height), js['Label']['objects'][i]['polygon'], color_extractor(js['Label']['objects'][i], type))
                    #img.convert('RGB')
                    img_list.append(img)
                except Exception:
                    print('Note: There are no available polygon-data-points & web-data-information for Label #{}.'.format(i))

        # print current progress status
        #progressBar(i, len(js['Label']['objects']))

        img = img_list[0]
        for i in range(1, len(img_list)):
            img.paste(img_list[i], (0,0), mask=img_list[i])

        img.save(output_pth + json_file.replace('.json', '.png'))
