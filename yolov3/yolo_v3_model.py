# https://colab.research.google.com/drive/1j8jvtaH4MkWyBnLvGrWiWvqZ8Gz9qVOd?usp=sharing
import cv2
import tensorflow as tf
import numpy as np
from yolov3.config import anchors, class_ids, class_mapping_decoder, class_mapping_encoder, image_width, image_height, num_class


def conv_bn_leaky(x, filters, kernel_size, strides=1, activation=True, bn = True):
    # stride == 2 thi padding thu cong.
    if strides == 2:
        x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
        padding = "valid"
    else:
        padding = "same"

    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size = kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,             # theo Darknet, BatchNorm thay bias
        kernel_regularizer=tf.keras.regularizers.l2(0.0005),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        bias_initializer=tf.constant_initializer(0.)
        )(x)
    if bn:
        x = tf.keras.layers.BatchNormalization()(x)
    if activation:
        x = tf.keras.layers.LeakyReLU(0.1)(x)
    return x

def residual_block(x, filters):
    shortcut = x

    x =conv_bn_leaky(x, filters//2, 1,1)
    x = conv_bn_leaky(x, filters, 3, 1)
    x = tf.keras.layers.Add()([shortcut,x])
    return x

def darknet53(input_shape=(416,416,3)):
    inputs = tf.keras.layers.Input(input_shape)

    x = conv_bn_leaky(inputs, 32, 3, 1)
    x = conv_bn_leaky(x, 64, 3, 2)
    x = residual_block(x, 64)
    x = conv_bn_leaky(x, 128, 3, 2)

    for _ in range(2):
        x = residual_block(x, 128)
    x = conv_bn_leaky(x, 256, 3, 2)

    for _ in range(8):
        x = residual_block(x, 256)
    head52 = x

    x = conv_bn_leaky(x, 512, 3, 2)

    for _ in range(8):
        x = residual_block(x, 512)
    head26 = x

    x = conv_bn_leaky(x, 1024, 3, 2)

    for _ in range(4):
        x = residual_block(x, 1024)
    head13 = x


    model = tf.keras.Model(inputs, [head52, head26, head13])
    return model

def yoloBlock(x, filters):
    x = conv_bn_leaky(x, filters, 1, 1)
    x = conv_bn_leaky(x, filters*2, 3, 1)
    x = conv_bn_leaky(x, filters, 1, 1)
    x = conv_bn_leaky(x, filters*2, 3, 1)
    x = conv_bn_leaky(x, filters, 1, 1)

    return x

def create_yolo_v3():
    backbone = darknet53(input_shape=(image_width,image_height,3))
    inputs = backbone.inputs
    head52, head26, head13 = backbone.outputs

    x = yoloBlock(head13, 512)
    head13 = conv_bn_leaky(x, 1024, 3, 1)
    head13 = conv_bn_leaky(head13, (5+num_class)*3, 1, 1,False, False)

    x = conv_bn_leaky(x, 256,1,1)
    x = tf.keras.layers.UpSampling2D(size=(2, 2), data_format="channels_last", interpolation='nearest')(x)
    x = tf.keras.layers.Concatenate()([x,head26])


    x = yoloBlock(x, 256)
    head26 = conv_bn_leaky(x, 512, 3, 1)
    head26 = conv_bn_leaky(head26,(5+num_class)*3,1,1, False, False)


    x = conv_bn_leaky(x, 128,1,1)
    x = tf.keras.layers.UpSampling2D(size=(2, 2), data_format="channels_last", interpolation='nearest')(x)
    x = tf.keras.layers.Concatenate()([x,head52])

    x = yoloBlock(x, 128)
    head52 = conv_bn_leaky(x, 256, 3,1)
    head52 = conv_bn_leaky(head52,(5+num_class)*3,1,1, False, False)

    model = tf.keras.Model(inputs, [head13, head26, head52])
    return model

def decode_predictions(y_pred, anchors, grid_size, num_classes=3):
    """
    Chuyển đổi tensor đầu ra thô của model thành tọa độ bounding box thực tế.

    Args:
        y_pred (tf.Tensor): Tensor đầu ra từ một head của model.
                             Shape: (batch_size, grid_size, grid_size, num_anchors * (5 + num_classes))
        anchors (np.ndarray): Mảng chứa các anchor box cho head này.
                              Shape: (num_anchors, 2)
        grid_size (int): Kích thước của grid (ví dụ: 13, 26, hoặc 52).
        num_classes (int): Số lượng lớp đối tượng.

    Returns:
        tf.Tensor: Tensor chứa các thông tin đã giải mã.
                   Mỗi hàng có dạng [x_center, y_center, width, height, confidence, class_prob_1, class_prob_2, ...].
                   Shape: (batch_size, grid_size * grid_size * num_anchors, 5 + num_classes)
    """
    # Lấy các thông số từ shape của tensor
    batch_size = y_pred.shape[0]
    num_anchors = len(anchors)

    # Reshape đầu vào để dễ xử lý hơn
    # Từ (batch, grid, grid, 24) -> (batch, grid, grid, 3, 8)
    y_pred = tf.reshape(y_pred, (batch_size, grid_size, grid_size, num_anchors, 5 + num_classes))

    # Tách các thành phần từ tensor dự đoán
    # Các giá trị này đều là logits (đầu ra thô)
    tx_ty = y_pred[..., 0:2]  # tx, ty
    tw_th = y_pred[..., 2:4]  # tw, th
    confidence_logit = y_pred[..., 4:5]
    class_probs_logits = y_pred[..., 5:]

    # --- Bước 1: Tạo Grid Cell Offsets ---
    # Tạo một grid để biết vị trí của mỗi cell
    grid_y = tf.tile(tf.reshape(tf.range(grid_size, dtype=tf.float32), [-1, 1, 1, 1]), [1, grid_size, 1, 1])
    grid_x = tf.tile(tf.reshape(tf.range(grid_size, dtype=tf.float32), [1, -1, 1, 1]), [grid_size, 1, 1, 1])
    grid = tf.concat([grid_x, grid_y], axis=-1)  # Shape: (grid_size, grid_size, 1, 2)

    # --- Bước 2: Giải mã tọa độ và kích thước ---
    # Áp dụng sigmoid cho tx, ty và cộng với offset của grid cell
    # Sau đó chia cho grid_size để chuẩn hóa về khoảng [0, 1]
    box_xy = (tf.sigmoid(tx_ty) + grid) / tf.cast(grid_size, tf.float32)

    # Áp dụng hàm mũ cho tw, th và nhân với kích thước anchor
    # anchors có shape (3, 2) cần reshape để nhân với tw_th có shape (batch, grid, grid, 3, 2)
    box_wh = tf.exp(tw_th) * anchors.reshape(1, 1, 1, num_anchors, 2)

    # --- Bước 3: Giải mã điểm tin cậy và xác suất lớp ---
    confidence = tf.sigmoid(confidence_logit)
    class_probs = tf.sigmoid(class_probs_logits)

    # --- Bước 4: Ghép nối và reshape kết quả cuối cùng ---
    decoded_preds = tf.concat([box_xy, box_wh, confidence, class_probs], axis=-1)

    # Reshape thành một danh sách các dự đoán để dễ dàng xử lý sau này
    # Shape: (batch_size, total_boxes, 5 + num_classes)
    decoded_preds = tf.reshape(decoded_preds, (batch_size, -1, 5 + num_classes))

    return decoded_preds


def inference(path, num_classes, model):
    import os
    grid_size = [13, 26, 52]
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (416, 416))/255.0
    img_expanded = np.expand_dims(img, axis=0)
    kq = model.predict(img_expanded)

    all = []
    for i in range(3):
        ac = anchors[i].reshape(-1,2)
        decoded_preds = decode_predictions(kq[i], ac, grid_size[i], num_classes)[0]
        confidences = decoded_preds[..., 4]
        mask = confidences >= 0.9
        decoded_preds = decoded_preds[mask]
        scores = confidences[mask]

        for row in decoded_preds:
            all.append(row)

    # luc nay all dang la 1 ds vs 3 phan tu
    # can noi tat ca cac phan tu cua ds trong ds
    all = np.array(all)
    nms = tf.image.non_max_suppression(
        boxes=all[..., 0:4], scores=all[...,4], max_output_size=20, iou_threshold=0.5
    )
    for j in nms:
        x,y,w,h = all[j][:4] * 416
        c = np.round(all[j][4],2)
        p = all[j][5:]

        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        x_left = int(x - w/2)
        y_top = int(y - h/2)
        x_right = int(x + w/2)
        y_bottom = int(y + h/2)

        nameclass = class_mapping_decoder[np.argmax(p)]

        cv2.rectangle(img, (x_left,y_top), (x_right,y_bottom), (0,255,0), 2)
        cv2.putText(img, nameclass+" : " + str(c), (x_left, y_top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    try:
        filename = os.path.basename(path)
        ten_file, phan_mo_rong = os.path.splitext(filename)
        cv2.imwrite("RESULT/" + ten_file + ".png", (img*255.0).astype(np.uint8))
        print("save image access in "+ "RESULT/" + ten_file + ".png")
    except:
        pass

    cv2.imshow('anh', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
