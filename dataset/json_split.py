import json
import os

json_pth = './all.json'
output_pth = './json_all/'

with open(json_pth, 'r') as f:
    json_file = json.load(f)

for img in json_file:
    ptr1 = img['Labeled Data'].find('Area')
    ptr2 = img['Labeled Data'].find('.png')
    filename = img['Labeled Data'][ptr1:ptr2]+'.json'
    with open(os.path.join(output_pth, filename), 'w') as f:
        json.dump(img, f)