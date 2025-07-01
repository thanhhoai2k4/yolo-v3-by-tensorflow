import tensorflow as tf

def compute_iou_for_yolo(boxes1, boxes2):
    x1_1 = boxes1[..., 0] - boxes1[..., 2] / 2
    y1_1 = boxes1[..., 1] - boxes1[..., 3] / 2  # y1 = y - h/2
    x2_1 = boxes1[..., 0] + boxes1[..., 2] / 2  # x2 = x + w/2
    y2_1 = boxes1[..., 1] + boxes1[..., 3] / 2  # y2 = y + h/2

    x1_2 = boxes2[..., 0] - boxes2[..., 2] / 2
    y1_2 = boxes2[..., 1] - boxes2[..., 3] / 2
    x2_2 = boxes2[..., 0] + boxes2[..., 2] / 2
    y2_2 = boxes2[..., 1] + boxes2[..., 3] / 2

    x1_inter = tf.maximum(x1_1, x1_2)  # Góc trên trái x
    y1_inter = tf.maximum(y1_1, y1_2)  # Góc trên trái y
    x2_inter = tf.minimum(x2_1, x2_2)  # Góc dưới phải x
    y2_inter = tf.minimum(y2_1, y2_2)  # Góc dưới phải y

    inter_width = tf.maximum(x2_inter - x1_inter, 0)  # Đảm bảo không âm
    inter_height = tf.maximum(y2_inter - y1_inter, 0)
    inter_area = inter_width * inter_height

    area1 = boxes1[..., 2] * boxes1[..., 3]  # w * h của boxes1
    area2 = boxes2[..., 2] * boxes2[..., 3]  # w * h của boxes2

    union_area = area1 + area2 - inter_area

    iou = inter_area / (union_area + tf.keras.backend.epsilon())  # Thêm epsilon để tránh chia cho 0

    return iou
# copy from pylesson: https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3/blob/master/yolov3/yolov3.py#L254
def bbox_giou(boxes1, boxes2):
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    # Calculate the iou value between the two bounding boxes
    iou = inter_area / union_area

    # Calculate the coordinates of the upper left corner and the lower right corner of the smallest closed convex surface
    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)

    # Calculate the area of the smallest closed convex surface C
    enclose_area = enclose[..., 0] * enclose[..., 1]

    # Calculate the GIoU value according to the GioU formula
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou


def t_xywh_to_xyxy(box,grid_size,anchors):

    tx_ty = box[..., 0:2]  # tx, ty
    tw_th = box[..., 2:4]  # tw, th
    grid_y = tf.tile(tf.reshape(tf.range(grid_size, dtype=tf.float32), [-1, 1, 1, 1]), [1, grid_size, 1, 1])
    grid_x = tf.tile(tf.reshape(tf.range(grid_size, dtype=tf.float32), [1, -1, 1, 1]), [grid_size, 1, 1, 1])
    grid = tf.concat([grid_x, grid_y], axis=-1)  # Shape: (grid_size, grid_size, 1, 2)

    tw_th = tf.clip_by_value(tw_th, -20, 20)
    box_xy = (tx_ty + grid) / tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(tw_th) * tf.reshape(anchors,(1,1,1,3,2))

    box_affter = tf.concat([box_xy, box_wh], axis=-1) # x_center, y_center, width, height [0-1]


    return box_affter


def getloss(num_class,anchors ,weight = [5.0, 2.0, 0.5, 1.0], IGNORE_THRESH=0.7):
    def loss_function(y_true, y_pred):

        batch = tf.shape(y_pred)[0]
        grid = tf.shape(y_true)[1]

        y_pred = tf.reshape(y_pred, [batch, grid, grid, 3 ,5+num_class])
        xy  = tf.sigmoid(y_pred[...,0:2])
        wh = y_pred[...,2:4]
        c = y_pred[...,4:5]
        p = y_pred[...,5:]
        y_pred = tf.concat([xy, wh, c, p], axis=-1)

        mask_object = tf.cast(y_true[...,4:5]==1.0, tf.float32)
        mask_no_object = tf.cast(y_true[...,4:5]==0, tf.float32)

        # box = tf.expand_dims(tf.keras.losses.MSE(y_true[...,0:4], y_pred[...,0:4]),axis=-1) * mask_object
        # box_loss = tf.math.reduce_sum(box)

        xywh_true = t_xywh_to_xyxy(y_true[...,0:4], grid, anchors)
        xywh_pred = t_xywh_to_xyxy(y_pred[...,0:4], grid, anchors)

        giou = tf.expand_dims(bbox_giou(xywh_true, xywh_pred), axis=-1)
        bbox_loss_scale = 2.0 - 1.0 * xywh_true[:, :, :, :, 2:3] * xywh_true[:, :, :, :, 3:4]
        giou_loss = (1-giou) * mask_object * bbox_loss_scale
        box_loss = tf.reduce_sum(giou_loss)

        ious = compute_iou_for_yolo(xywh_true, xywh_pred)
        ignore_mask = tf.cast(ious < IGNORE_THRESH, tf.float32)

        confident_all_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true[...,4], logits=y_pred[...,4])
        confident_loss = tf.reduce_sum(tf.expand_dims(confident_all_loss,axis=-1) * mask_object)

        confident_non_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true[...,4], logits=y_pred[...,4])
        confident_no_loss = tf.reduce_sum(tf.expand_dims(confident_non_loss,axis=-1) * mask_no_object * tf.expand_dims(ignore_mask, axis=-1))


        loss_p = tf.nn.sigmoid_cross_entropy_with_logits(y_true[...,5:] , y_pred[...,5:])
        loss_p = tf.reduce_sum(loss_p * mask_object)


        loss = (weight[0]*box_loss + weight[1]*confident_loss + confident_no_loss*weight[2] + loss_p*weight[3]) / tf.cast(batch, tf.float32)
        return loss
    return loss_function