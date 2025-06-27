import cv2
import numpy as np
import random
import matplotlib.pyplot as plt


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


# --- VÍ DỤ SỬ DỤNG ---
if __name__ == '__main__':
    # 1. Tạo ảnh mẫu và chuẩn hóa nó về khoảng [0, 1]
    image = cv2.imread("../data/images/maksssksksss0.png")
    height, width = image.shape[:2]
    boxes = np.array([
        [76.37, 140, 24.3, 42, 0],
        [167, 128.7, 33.3, 50, 0],
    ]) / 416
    for row in boxes:
        x,y,w,h = row[:4]

        xmin = int((x-w/2)* width )
        ymin = int((y - h / 2) *height)
        xmax = int((x+ w/2) * width)
        ymax = int((y+h/2)*height )

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
        cv2.imshow("image", image)

    # 3. Áp dụng hàm augmentation
    translated_img, translated_bboxes = translate_normalized_yolo(image, boxes,
                                                                  max_translate_ratio=0.1)

    for row in translated_bboxes:
        x,y,w,h = row[:4]

        xmin = int((x-w/2)* width )
        ymin = int((y - h / 2) *height)
        xmax = int((x+ w/2) * width)
        ymax = int((y+h/2)*height )

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
        cv2.imshow("image", translated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()