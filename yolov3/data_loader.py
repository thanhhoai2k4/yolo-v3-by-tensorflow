import os
import numpy as np
import cv2
import random
from yolov3.config import anchors, class_ids, class_mapping_decoder, class_mapping_encoder, num_class, image_width, image_height

xml_list = os.listdir("data/annotations") # lay danh sach cac file xml
xml_list = [os.path.join(os.getcwd(),"data/annotations",xml) for xml in xml_list]
xml_list_val = os.listdir("val/annotations")
xml_list_val = [os.path.join(os.getcwd(),"val/annotations",xml) for xml in xml_list_val]




def parse_xml(path: str, trainorval = True):
    try:
        from xml.etree import ElementTree as ET
        tree = ET.parse(path)
        root = tree.getroot()

        width_image = int(root.find('size')[0].text)
        height_image = int(root.find('size')[1].text)

        name = root.find('filename').text
        folder = root.find('folder').text

        # x = 2 if x < 2 else x
        if trainorval:
            path_image = os.path.join(os.getcwd(), "data", folder, name)
        else:
            path_image = os.path.join(os.getcwd(), "val", folder, name)

        boxes = []
        for obj in root.iter('object'):
            xmin = int(obj.find('bndbox')[0].text)
            ymin = int(obj.find('bndbox')[1].text)
            xmax = int(obj.find('bndbox')[2].text)
            ymax = int(obj.find('bndbox')[3].text)
            name = obj.find('name').text
            id = class_mapping_encoder[name]

            x_center = (xmin + xmax) / 2 / width_image
            y_center = (ymin + ymax) / 2 / height_image
            width = (xmax - xmin) / width_image
            height = (ymax - ymin) / height_image

            row = np.array([x_center, y_center, width, height, id], dtype=np.float32)
            boxes.append(row)

        return path_image, np.array(boxes, dtype=np.float32)
    except Exception as e:
        return None

def box_corner_to_center(boxes):
    """
    Convert box corners[xmin, ymin, xmax, ymax] to center coordinates[x_center, y_center, width, height].
    boxes : shape[number box, 4]
    return : shape[number box, 4]
    """

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    # center_x
    cx = (x1 + x2) / 2
    # center_y
    cy = (y1 + y2) / 2
    # width
    w = (x2 - x1)
    # height
    h = (y2 - y1)
    boxes = np.stack((cx, cy, w, h), axis=-1)
    return boxes


def box_center_to_corner(boxes):
    """
    apllay for  npArray
    Convert box center coordinates[x_center, y_center, width, height] to corners[xmin, ymin, xmax, ymax].
    boxes: shape[number box, 4]
    return: shape[number box, 4]
    """

    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    boxes = np.stack((x1, y1, x2, y2), axis=-1)
    return boxes


def bboxes_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious

def padding(anchors_grid, number_box, x=True):
    anchors_grid = np.array(anchors_grid)
    zeros = np.zeros((number_box, 4), dtype=np.float32)
    if x==True:
        zeros[...,2:4] = anchors_grid
    else:
        anchors_grid[...,0:2] = 0
        zeros  = anchors_grid
    return zeros
def sigmoid_inverse(x):
    return np.log(x/(1-x))
def sigmoid(x):
    return 1/(1+np.exp(-x))


# ... (giữ nguyên các hàm và biến ở đầu file data_loader.py)

def encode_boxes(boxes: np.ndarray, grid_size_list: list[int, int, int] = [13, 26, 52],
                 number_class: int = 2):
    # Khởi tạo các mảng y_true cho mỗi đầu ra (head)
    y_true_13 = np.zeros((grid_size_list[0], grid_size_list[0], 3, 5 + number_class), dtype=np.float32)
    y_true_26 = np.zeros((grid_size_list[1], grid_size_list[1], 3, 5 + number_class), dtype=np.float32)
    y_true_52 = np.zeros((grid_size_list[2], grid_size_list[2], 3, 5 + number_class), dtype=np.float32)

    y_true_list = [y_true_13, y_true_26, y_true_52]

    for box in boxes:
        # box: [x_center, y_center, width, height, class_id]
        x, y , w, h ,id = box
        x = np.clip(x, 1e-6, 0.999)
        y = np.clip(y, 1e-6, 0.999)
        # cap nhat lai box
        box = [x,y,w,h,id]

        
        # Tìm anchor phù hợp nhất cho box này trên tất cả các scale
        box_wh = box[2:4]
        # Mở rộng chiều để broadcasting
        box_wh = np.expand_dims(box_wh, 0)  # shape (1, 2)
        anchors_wh = anchors.reshape(-1, 2)  # shape (9, 2)

        # Tính IoU giữa box và tất cả 9 anchors
        box_area = box_wh[:, 0] * box_wh[:, 1]
        anchor_area = anchors_wh[:, 0] * anchors_wh[:, 1]

        # Intersection
        inter_w = np.minimum(box_wh[:, 0], anchors_wh[:, 0])
        inter_h = np.minimum(box_wh[:, 1], anchors_wh[:, 1])
        inter_area = inter_w * inter_h

        # Union
        union_area = box_area + anchor_area - inter_area
        iou = inter_area / union_area

        best_anchor_index = np.argmax(iou)

        # Xác định scale và anchor_index trong scale đó
        scale_index = best_anchor_index // 3  # 0, 1, or 2
        anchor_index_in_scale = best_anchor_index % 3  # 0, 1, or 2

        grid_size = grid_size_list[scale_index]
        y_true = y_true_list[scale_index]

        # Xác định ô grid chịu trách nhiệm cho box này
        grid_x = int(grid_size * box[0])
        grid_y = int(grid_size * box[1])

        # Lấy class_id
        class_id = int(box[4])

        # Lấy anchor tương ứng
        selected_anchor_wh = anchors[scale_index][anchor_index_in_scale]

        # Tính toán các giá trị target (tx, ty, tw, th)
        tx = grid_size * box[0] - grid_x
        ty = grid_size * box[1] - grid_y
        tw = np.log(box[2] / selected_anchor_wh[0] + 1e-6)
        th = np.log(box[3] / selected_anchor_wh[1] + 1e-6)

        # Gán giá trị vào mảng y_true tương ứng
        y_true[grid_y, grid_x, anchor_index_in_scale, 0:4] = [tx, ty, tw, th]
        y_true[grid_y, grid_x, anchor_index_in_scale, 4] = 1.0  # Confidence
        y_true[grid_y, grid_x, anchor_index_in_scale, 5 + class_id] = 1.0  # One-hot encoding cho class

    return y_true_13, y_true_26, y_true_52

def scale_image_and_boxes(image, boxes, scale_factor):
    """
    Args:
        image: anh width, width, 3
        boxes: n,4
        scale_factor:  ti le scale >0

    Returns: tra ve 1 anh va box tuong ung duoc scale.

    """
    # kich thuoc cu
    (h, w) = image.shape[:2]

    # kich thuoc moi
    new_w = int(w*scale_factor)
    new_h = int(h*scale_factor)

    resized_image = cv2.resize(image, (new_w, new_h))

    # tao 1 cai nen mau xam 128/255.0
    canvas = np.full(shape=(h,w,3), fill_value=128.0/255.0, dtype=np.float32)

    x_offset = (w-new_w)//2
    y_offset = (h-new_h)//2

    if scale_factor < 1.0:
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image
        final_image = canvas
    else:
        final_image = resized_image[abs(y_offset):abs(y_offset) + h, abs(x_offset):abs(x_offset) + w]

    # Su ly image thanh cong

    new_boxes = []
    if len(boxes) > 0:
        for box in boxes:
            x_center, y_center, width, height, class_id = box # dang o dang chuan hoa

            new_box_w = width * scale_factor  # widht moi
            new_box_h = height * scale_factor #height moi

            new_x_center = (x_center * scale_factor) + (x_offset / w)
            new_y_center = (y_center * scale_factor) + (y_offset / h)

            # --- KIỂM TRA TÍNH HỢP LỆ CỦA BOX MỚI ---
            # Chuyển đổi box mới về dạng (xmin, ymin, xmax, ymax) để dễ tính toán.
            xmin = new_x_center - new_box_w / 2
            ymin = new_y_center - new_box_h / 2
            xmax = new_x_center + new_box_w / 2
            ymax = new_y_center + new_box_h / 2

            # Rất quan trọng khi phóng to, vì một phần box có thể bị cắt mất.
            xmin = max(0.0, xmin)
            ymin = max(0.0, ymin)
            xmax = min(1.0, xmax)
            ymax = min(1.0, ymax)

            if xmax > xmin and ymax > ymin:
                final_x_center = (xmin + xmax) / 2
                final_y_center = (ymin + ymax) / 2
                final_width = xmax - xmin
                final_height = ymax - ymin
                new_boxes.append([final_x_center, final_y_center, final_width, final_height, class_id])

    return final_image, np.array(new_boxes, dtype=np.float32)

def data_agrument_flip(image, boxes):
    """

        tra ve anh va boxes da duuoc fliped
    Args:
        image: width, height, channel
        boxes: da o dang chuan hoa

    Returns: image boxes

    """
    if np.random.random() > 0.5:
        temp_box = []
        image = np.flip(image, 1)
        for box in boxes:
            x_center, y_center, width, height, id = box
            flipped_x_center = 1.0 - x_center
            temp_box.append([flipped_x_center, y_center, width, height, id])

        return image, np.array(temp_box)
    return image, boxes

def rotate_image_and_boxes(image, angle, boxes):
    """
    Xoay ảnh và các bounding box tương ứng.

    Args:
        image (np.array): Ảnh đầu vào (H, W, C).
        angle (float): Góc xoay (đơn vị: độ).
        boxes (list): Danh sách các bounding box, mỗi box có dạng [x_center, y_center, width, height]. o dang chuan hoa

    Returns:
        rotated_image: Ảnh đã xoay.
        new_boxes: Danh sách các bounding box mới.
    """

    ids = boxes[...,4:5]
    boxes = boxes[...,0:4] # xywh
    boxes = box_center_to_corner(boxes.reshape(-1,4)) # xyxy

    # Lấy kích thước ảnh
    h, w = image.shape[:2]
    # Lấy tâm xoay
    center = (w // 2, h // 2)

    # 1. Tạo ma trận xoay
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 2. Xoay ảnh
    rotated_image = cv2.warpAffine(image, M, (w, h))

    valid_new_boxes = []
    valid_ids = []
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box
        # Lấy tọa độ 4 góc của bounding box
        corners = np.array([
            [x_min*w, y_min*h],
            [x_max*w, y_min*h],
            [x_max*w, y_max*h],
            [x_min*w, y_max*h]
        ])

        # Thêm 1 vào cuối để thực hiện phép nhân ma trận
        ones = np.ones(shape=(len(corners), 1))
        points_ones = np.hstack([corners, ones])

        # 3. Xoay tọa độ các góc
        transformed_points = M.dot(points_ones.T).T

        # 4. Tìm bounding box mới bao trọn các góc đã xoay
        new_x_min = min(transformed_points[:, 0])
        new_y_min = min(transformed_points[:, 1])
        new_x_max = max(transformed_points[:, 0])
        new_y_max = max(transformed_points[:, 1])

        # Đảm bảo bounding box không vượt ra ngoài ảnh
        new_x_min = max(0, new_x_min)
        new_y_min = max(0, new_y_min)
        new_x_max = min(w, new_x_max)
        new_y_max = min(h, new_y_max)

        if new_x_max - new_x_min < 10 or new_y_max - new_y_min < 10:
            continue

        valid_new_boxes.append([new_x_min / w, new_y_min / h, new_x_max / w, new_y_max / h])
        valid_ids.append(ids[i])
    # neu khong ton tai! boxes nao thi tra ve
    if not valid_new_boxes:
        return rotated_image, np.array([])

    new_boxes = np.array(valid_new_boxes)# xyxy
    final_xywh = box_corner_to_center(new_boxes.reshape(-1,4)) #xywh
    final_ids = np.array(valid_ids)
    rows = np.concatenate([final_xywh, final_ids], axis=1)

    return rotated_image, rows

# translate: dich chuyen
def translate_normalized_yolo(image, bboxes, max_translate_ratio=0.1):
    """
    Thực hiện phép dịch chuyển ngẫu nhiên cho ảnh ĐÃ CHUẨN HÓA và các bounding box
    ở định dạng YOLO ĐÃ CHUẨN HÓA [x_center, y_center, width, height].

    Args:
        image (np.array): Ảnh đầu vào đã chuẩn hóa (giá trị pixel từ 0 đến 1).
        bboxes (list of lists): Danh sách các bounding box ở định dạng YOLO chuẩn hóa.
        max_translate_ratio (float): Tỷ lệ dịch chuyển tối đa.

    Returns:
        tuple: (Ảnh đã dịch chuyển, danh sách các bounding box YOLO đã được cập nhật).
    """
    # Nếu không có bounding box nào thì không làm gì cả
    if len(bboxes) == 0:
        return image, bboxes

    # Lấy chiều cao và chiều rộng pixel thực tế của ảnh
    (height, width) = image.shape[:2]

    # 1. Tính toán khoảng cách dịch chuyển THEO TỶ LỆ (normalized)
    # tx, ty bây giờ là các giá trị trong khoảng [-max_translate_ratio, max_translate_ratio]
    tx = random.uniform(-max_translate_ratio, max_translate_ratio)
    ty = random.uniform(-max_translate_ratio, max_translate_ratio)

    # 2. Tạo ma trận biến đổi affine
    # Phải nhân tx, ty với kích thước ảnh thực tế để cv2.warpAffine hiểu
    M = np.float32([[1, 0, tx * width],
                    [0, 1, ty * height]])

    # 3. Áp dụng phép dịch chuyển lên ảnh
    translated_image = cv2.warpAffine(image, M, (width, height))

    # 4. Cập nhật tọa độ cho các bounding box
    updated_bboxes = []
    for bbox in bboxes:
        x_center, y_center, w, h, id = bbox

        # Áp dụng trực tiếp phép dịch chuyển chuẩn hóa vào tâm của box
        new_x_center = x_center + tx
        new_y_center = y_center + ty

        # 5. Xử lý các box bị dịch chuyển một phần hoặc hoàn toàn ra khỏi ảnh

        # Chuyển từ xywh về xyxy để kiểm tra và cắt xén
        x_min = new_x_center - w / 2
        y_min = new_y_center - h / 2
        x_max = x_min + w
        y_max = y_min + h

        # Ràng buộc tọa độ trong khoảng [0, 1]
        clipped_x_min = np.clip(x_min, 0.0, 1.0)
        clipped_y_min = np.clip(y_min, 0.0, 1.0)
        clipped_x_max = np.clip(x_max, 0.0, 1.0)
        clipped_y_max = np.clip(y_max, 0.0, 1.0)

        # Tính lại width và height mới sau khi cắt xén
        new_w = clipped_x_max - clipped_x_min
        new_h = clipped_y_max - clipped_y_min

        # Nếu box còn lại có diện tích > 0 thì mới giữ lại
        if new_w > 0 and new_h > 0:
            # Tính lại tâm mới và chuyển về định dạng YOLO
            final_x_center = clipped_x_min + new_w / 2
            final_y_center = clipped_y_min + new_h / 2
            updated_bboxes.append([final_x_center, final_y_center, new_w, new_h, id])

    return translated_image, updated_bboxes

def datagenerator():
    for xml in xml_list:
        parsed_data  = parse_xml(xml)
        if parsed_data is None:
            print(f"Cảnh báo: Bỏ qua file annotation bị lỗi hoặc rỗng: {xml}")
            continue


        path_image, boxes = parsed_data
        img = cv2.imread(path_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # lat anh theo chieu ngnang ti le 0.5
        img, boxes = data_agrument_flip(img, boxes)
        # ---- doan code de scale
        if np.random.random() > 0.5:
            scale_factor = np.random.uniform(low=0.8, high=1.2, size=None) # None thi tra ve scalar
            img, boxes = scale_image_and_boxes(img, boxes, scale_factor)
        # ---- xoay anh
        if np.random.random() > 0.5:
            angle = 5
            img, boxes = rotate_image_and_boxes(img, angle, boxes)
        # translate image: dich chuyen anh va boxes
        if np.random.random() > 0.5:
            img, boxes = translate_normalized_yolo(img, boxes,max_translate_ratio=0.1)


        img = cv2.resize(img, (416, 416)) / 255.0
        head13, head26, head52 = encode_boxes(boxes, number_class=num_class)
        yield np.array(img), (np.array(head13,dtype=np.float32), np.array(head26,dtype=np.float32), np.array(head52,dtype=np.float32))



def datagenerator_cache():
    imgs = []
    head13s = []
    head26s = []
    head52s = []
    for xml in xml_list:
        path_image, boxes = parse_xml(xml)
        head13, head26, head52 = encode_boxes(boxes, number_class=num_class)
        img = cv2.imread(path_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (416, 416)) / 255.0
        imgs.append(np.array(img))
        head13s.append(np.array(head13,dtype=np.float32))
        head26s.append(np.array(head26,dtype=np.float32))
        head52s.append(np.array(head52,dtype=np.float32))
    return np.array(imgs,dtype=np.float32), np.array(head13s, dtype=np.float32), np.array(head26s, dtype=np.float32), np.array(head52s,dtype=np.float32)



# val
def datagenerator_val():
    for xml in xml_list_val:
        path_image, boxes = parse_xml(xml,False)
        head13, head26, head52 = encode_boxes(boxes, number_class=num_class)
        img = cv2.imread(path_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (416, 416)) / 255.0
        yield np.array(img), (np.array(head13,dtype=np.float32), np.array(head26,dtype=np.float32), np.array(head52,dtype=np.float32))


def datagenerator_test():
    for xml in xml_list:
        parsed_data  = parse_xml(xml)
        if parsed_data is None:
            print(f"Cảnh báo: Bỏ qua file annotation bị lỗi hoặc rỗng: {xml}")
            continue


        path_image, boxes = parsed_data
        img = cv2.imread(path_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # lat anh theo chieu ngnang ti le 0.5
        img, boxes = data_agrument_flip(img, boxes)
        # ---- doan code de scale
        if np.random.random() > 0.5:
            scale_factor = np.random.uniform(low=0.8, high=1.2, size=None) # None thi tra ve scalar
            img, boxes = scale_image_and_boxes(img, boxes, scale_factor)
        # ---- xoay anh
        if np.random.random() > 0.5:
            angle = 20
            img, boxes = rotate_image_and_boxes(img, angle, boxes)
        # translate image: dich chuyen anh va boxes
        if np.random.random() > 0.5:
            img, boxes = translate_normalized_yolo(img, boxes,max_translate_ratio=0.2)


        img = cv2.resize(img, (416, 416)) / 255.0
        head13, head26, head52 = encode_boxes(boxes, number_class=num_class)
        yield np.array(img), boxes, path_image