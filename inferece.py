from yolov3.yolo_v3_model import *
import os
from yolov3.config import num_class, checkpoint_path
xxx = "val/images"
xml_list = os.listdir(xxx) # lay danh sach cac file xml
xml_list = [os.path.join(os.getcwd(),xxx,xml) for xml in xml_list]
model = tf.keras.models.load_model(checkpoint_path)
for xml in xml_list:
    inference(xml, num_class, model)