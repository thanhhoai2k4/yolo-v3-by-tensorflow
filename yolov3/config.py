import numpy as np
import os


checkpoint_path = "model.h5"

image_width = 416
image_height = 416

batch_size = 16
epochs = 100
anchors = np.array([[[132, 172], [197, 252], [344, 428]],
                    [[52, 65], [68, 93], [96, 122]],
                    [[12, 17], [23, 31], [35, 48]]],
                   dtype=np.float32) / image_width

class_ids = ["Human face"]
num_class = len(class_ids)
class_mapping_decoder = dict(zip( range(len(class_ids)), class_ids ))
class_mapping_encoder = dict(zip(class_ids, range(len(class_ids))))

try:
    xml_list_length = os.listdir("data/annotations") # so luong huan luyen
    xml_list_val_length = os.listdir("val/annotations") # so luong val

    step_per_epoch = len(xml_list_length)//batch_size
    step_per_val = len(xml_list_val_length)//batch_size
except FileNotFoundError:
    pass