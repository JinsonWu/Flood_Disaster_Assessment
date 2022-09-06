# -*- coding: utf-8 -*-
"""
Created on Tue May 17 22:15:47 2022

@author: USER
"""
"""
from uuid import uuid4

API_KEY = "cl37r9rvzoer307ckfijq29py"
client = labelbox.Client(api_key=API_KEY)

dataset = client.create_dataset(name='Area1_post_cropped')
"""
import labelbox

# Enter your Labelbox API key here
LB_API_KEY = "cl37r9rvzoer307ckfijq29py"

# Create Labelbox client
lb = labelbox.Client(api_key=LB_API_KEY)

# Create a new dataset
dataset = lb.create_dataset(name="Area1_post_cropped")

# Create data payload
# External ID is recommended to identify your data_row via unique reference throughout Labelbox workflow.
my_data_rows = [
  {
    "row_data": "https://picsum.photos/200/300",
    "external_id": "uid_01"},
  {
    "row_data": "https://picsum.photos/200/400",
    "external_id": "uid_02"
  }
]

# Bulk add data rows to the dataset
task = dataset.create_data_rows(my_data_rows)

task.wait_till_done()
print(task.status)