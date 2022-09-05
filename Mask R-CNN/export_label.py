# install latest labelbox version (3.0 or above)
# !pip3 install labelbox[data]
from labelbox import Client, OntologyBuilder
from labelbox.data.annotation_types import Geometry
from PIL import Image
import numpy as np
import os
import json

# Enter your Labelbox API key here
LB_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbDM3c3ByanVxNnl1MDc5c2c4cTIzejU1Iiwib3JnYW5pemF0aW9uSWQiOiJjazVvNmR2bWIzendxMDgzNXdqcjZsOG1pIiwiYXBpS2V5SWQiOiJjbDQ2M2g0eXgxOGZtMDc3NGd3cm5nbmZoIiwic2VjcmV0IjoiMDgyNTMzMGVkZWQxODc0MGY0MjVlYzAyN2M1MTkzNjQiLCJpYXQiOjE2NTQ3MjMyMTAsImV4cCI6MjI4NTg3NTIxMH0.gkIj23QRUObSI_Rx2Xecj5JeoIUCS5bldeBwvEROF_w"
# Create Labelbox client
client = Client(api_key=LB_API_KEY)
# Get project by ID
project = client.get_project('cl37r9rvzoer307ckfijq29py')
# Export image and text data as an annotation generator:
labels = project.label_generator()
labels = labels.as_list()

