import numpy as np
import os


checkpoint_path = "model.h5"

image_width = 416
image_height = 416

batch_size = 16
epochs = 100
anchors = np.array([[[46, 80], [71, 110], [135, 138]],
                    [[22, 34], [23, 44], [33, 57]],
                    [[8, 14], [13, 23], [17, 31]]],
                   dtype=np.float32) / image_width

class_ids = ["with_mask", "without_mask","mask_weared_incorrect"]
num_class = len(class_ids)
class_mapping_decoder = dict(zip( range(len(class_ids)), class_ids ))
class_mapping_encoder = dict(zip(class_ids, range(len(class_ids))))

try:
    xml_list_length = os.listdir("data/annotations") # so luong huan luyen
    xml_list_val_length = os.listdir("val/annotations") # so luong val

    step_per_epoch = len(xml_list_length)//batch_size
    step_per_val = len(xml_list_val_length)//batch_size
except FileNotFoundError:
    print("Duong dan data hoac val chua ton tai. DU lieu chua duoc tao.")
    pass