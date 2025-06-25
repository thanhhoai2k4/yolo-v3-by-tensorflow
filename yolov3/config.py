import numpy as np
import os

num_class = 3
checkpoint_path = "model.h5"

image_width = 416
image_height = 416

batch_size = 16
epochs = 100
anchors = np.array([[[40, 44], [59, 64], [103, 124]],
                    [[18, 25], [24, 25], [30, 33]],
                    [[7, 8], [11, 12], [15, 17]]],
                   dtype=np.float32) / 416

class_ids = ["mask_weared_incorrect", "without_mask","with_mask"]
class_mapping_decoder = dict(zip( range(len(class_ids)), class_ids ))
class_mapping_encoder = dict(zip(class_ids, range(len(class_ids))))


xml_list_length = os.listdir("data/annotations") # so luong huan luyen
xml_list_val_length = os.listdir("val/annotations") # so luong val

step_per_epoch = len(xml_list_length)//batch_size
step_per_val = len(xml_list_val_length)//batch_size