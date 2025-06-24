import numpy as np
import cv2
class_ids = ["mask_weared_incorrect", "without_mask","with_mask"]
class_mapping_decoder = dict(zip( range(len(class_ids)), class_ids ))
class_mapping_encoder = dict(zip(class_ids, range(len(class_ids))))
def scale_image_and_boxes(image, boxes, scale_factor):

    # kich thuoc cu
    (h, w) = image.shape[:2]

    # kich thuoc moi
    new_w = int(w*scale_factor)
    new_h = int(h*scale_factor)

    resized_image = cv2.resize(image, (new_w, new_h))

    # tao 1 cai nen mau xam 128
    canvas = np.full(shape=(h,w,3), fill_value=128.0 / 255.0, dtype=np.float32)

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
                # Nếu hợp lệ, tính lại tọa độ tâm và kích thước cuối cùng từ các giá trị đã cắt.
                final_x_center = (xmin + xmax) / 2
                final_y_center = (ymin + ymax) / 2
                final_width = xmax - xmin
                final_height = ymax - ymin
                # Thêm box đã xử lý hoàn chỉnh vào danh sách
                new_boxes.append([final_x_center, final_y_center, final_width, final_height, class_id])

    # Trả về ảnh cuối cùng và danh sách các box mới dưới dạng mảng NumPy.
    return cv2.resize(final_image, (416, 416)), np.array(new_boxes, dtype=np.float32)







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


final_image = cv2.imread("../data/images/maksssksksss0.png")/ 255.0
boxes = np.array([
    [76.37, 140, 24.3,42, 0],
    [167, 128.7, 33.3,50, 0],
])/ 416

final_image, boxes = scale_image_and_boxes(final_image, boxes, 0.8)
show_image_with_boxes_cv2(final_image, boxes)