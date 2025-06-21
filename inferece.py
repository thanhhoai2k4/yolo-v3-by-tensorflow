from yolov3.yolo_v3_model import *
import os
xxx = "data/images"
xml_list = os.listdir(xxx) # lay danh sach cac file xml
xml_list = [os.path.join(os.getcwd(),xxx,xml) for xml in xml_list]
model = tf.keras.models.load_model("model1.h5")
for xml in xml_list:
    inference(xml, 3, model)