from PIL import Image
from PIL import UnidentifiedImageError
import requests
import json

def open_json(path):
    with open(path) as file:
        return json.load(file)

def open_img(url):
    try:
        return Image.open(requests.get(url, stream=True).raw)
    except UnidentifiedImageError:
        return None

json_pth = './all.json'
json_file = open_json(json_pth)
output_pth = './post_img/'

for img in json_file:
    orig_img = open_img(img['Labeled Data'])
    ptr1 = img['Labeled Data'].find('Area')
    ptr2 = img['Labeled Data'].find('.png') + 4
    orig_img.save(output_pth + img['Labeled Data'][ptr1:ptr2])