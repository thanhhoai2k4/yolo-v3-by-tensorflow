import cv2
import numpy as np
import matplotlib.pyplot as plt
class_ids = ["mask_weared_incorrect", "without_mask","with_mask"]
class_mapping_decoder = dict(zip( range(len(class_ids)), class_ids ))
class_mapping_encoder = dict(zip(class_ids, range(len(class_ids))))
def show_image_with_boxes_cv2(image, boxes, window_name="Image with Boxes"):
    """
    Hiển thị ảnh và vẽ các bounding box bằng OpenCV.

    Args:
        image (np.array): Ảnh đầu vào (định dạng BGR mà OpenCV sử dụng).
        boxes (np.array): Mảng các bounding box, mỗi box có định dạng YOLO
                          [x_center, y_center, width, height, class_id].
        window_name (str): Tên của cửa sổ hiển thị.
    """
    # Tạo một bản sao của ảnh để không vẽ trực tiếp lên ảnh gốc
    display_image = image.copy()

    # Lấy chiều cao và chiều rộng của ảnh
    (h, w) = display_image.shape[:2]

    # Kiểm tra xem có bounding box nào để vẽ không
    if boxes is not None and len(boxes) > 0:
        # Lặp qua từng bounding box
        for box in boxes:
            # Lấy thông tin từ box
            x_center, y_center, width, height, class_id = box

            # --- Chuyển đổi từ tọa độ YOLO chuẩn hóa sang tọa độ pixel ---
            # Tính toán chiều rộng và chiều cao của box bằng pixel
            box_w_px = int(width * w)
            box_h_px = int(height * h)

            # Tính toán tâm của box bằng pixel
            x_center_px = int(x_center * w)
            y_center_px = int(y_center * h)

            # Tính toán tọa độ góc trên bên trái (xmin, ymin)
            xmin = int(x_center_px - (box_w_px / 2))
            ymin = int(y_center_px - (box_h_px / 2))

            # --- Vẽ lên ảnh ---
            # 1. Vẽ hình chữ nhật (bounding box)
            # cv2.rectangle(ảnh, điểm_bắt_đầu, điểm_kết_thúc, màu_sắc, độ_dày)
            color = (0, 255, 0)  # Màu xanh lá cây (BGR)
            thickness = 2
            cv2.rectangle(display_image, (xmin, ymin), (xmin + box_w_px, ymin + box_h_px), color, thickness)

            # 2. Vẽ nhãn tên lớp (class label)
            # Lấy tên lớp từ class_id
            label = class_mapping_decoder.get(int(class_id), 'Unknown')
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 1

            # Đặt vị trí cho văn bản ngay phía trên bounding box
            text_position = (xmin, ymin - 10) if ymin - 10 > 10 else (xmin, ymin + 20)

            cv2.putText(display_image, label, text_position, font, font_scale, color, font_thickness)

    # Hiển thị ảnh trong một cửa sổ
    cv2.imshow(window_name, display_image)

    # Đợi người dùng nhấn một phím bất kỳ để đóng cửa sổ
    # Nếu tham số là 0, nó sẽ đợi vô hạn.
    cv2.waitKey(0)

    # Đóng tất cả các cửa sổ đã mở bởi OpenCV
    cv2.destroyAllWindows()

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
    boxes = box_center_to_corner(boxes) # xyxy

    # Lấy kích thước ảnh
    h, w = image.shape[:2]
    # Lấy tâm xoay
    center = (w // 2, h // 2)

    # 1. Tạo ma trận xoay
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 2. Xoay ảnh
    rotated_image = cv2.warpAffine(image, M, (w, h))

    new_boxes = []
    for box in boxes:
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

        new_boxes.append([new_x_min/w, new_y_min/h, new_x_max/w, new_y_max/h])

    new_boxes = np.array(new_boxes)
    new_boxes = box_corner_to_center(new_boxes)
    rows = np.concatenate([new_boxes, ids], axis=1)

    return rotated_image, rows


# --- Ví dụ sử dụng ---
# Tạo một ảnh đen đơn giản
image = cv2.imread("../data/images/maksssksksss0.png")/ 255.0
# Bounding box ban đầu (màu xanh)
original_box = np.array([
    [94/512, 123.5/366, 30/512,37/366 , 0],
]) # xywh

show_image_with_boxes_cv2(image, original_box)
rotated_image, rows = rotate_image_and_boxes(image, 20, original_box)
show_image_with_boxes_cv2(rotated_image,rows)