from yolov3.yolo_v3_model import create_yolo_v3
from yolov3.data_loader import datagenerator_cache
import tensorflow as tf
from yolov3.losses import getloss
import numpy as np
import os

xml_list = os.listdir("data/annotations") # lay danh sach cac file xml
batch_size = 1

anchors = np.array([[[116,90], [156,198], [373,326]],
                    [[30,61], [62,45], [59,119]],
                    [[10,13], [16,30], [33,23]]],
                   dtype=np.float32) / 416

epochs = 60

if  not os.path.exists("model.h5"):

    yolo_model = create_yolo_v3()
    optimizer = tf.keras.optimizers.Adam(0.001)
    yolo_model.compile(optimizer=optimizer, loss=[getloss(3, anchors[0]), getloss(3,anchors[1]), getloss(3,anchors[2])], run_eagerly=True)
else:
    print("load model tuyen huan luyen")
    yolo_model = tf.keras.models.load_model("model.h5")
    optimizer = tf.keras.optimizers.Adam(0.001)
    yolo_model.compile(optimizer=optimizer, loss=[getloss(3, anchors[0]), getloss(3,anchors[1]), getloss(3,anchors[2])], run_eagerly=True)


imgs , head13, head26, head52 = datagenerator_cache()
dataset = tf.data.Dataset.from_tensor_slices((imgs, (head13, head26, head52))).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

history = yolo_model.fit(dataset, epochs = epochs, steps_per_epoch=len(xml_list) // batch_size)
yolo_model.save("model.h5")