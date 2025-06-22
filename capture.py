import cv2
import numpy as np 
import tensorflow as tf
from tensorflow.keras.models import load_model
from yolov3.yolo_v3_model import decode_predictions
from yolov3.config import anchors, num_class, class_mapping_decoder

# --- TÙY CHỈNH CÁC THAM SỐ ---
MODEL_PATH = 'model.h5'
# Thay đổi kích thước ảnh đầu vào của mô hình (nếu cần)
MODEL_INPUT_WIDTH = 416
MODEL_INPUT_HEIGHT = 416

# -----------------------------
# -----------------------------

def preprocess_frame(frame):
    """
    Tiền xử lý khung hình để phù hợp với đầu vào của mô hình.
    """
    # Thay đổi kích thước khung hình
    resized_frame = cv2.resize(frame, (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT))
    # Chuẩn hóa giá trị pixel (ví dụ: về khoảng [0, 1])
    normalized_frame = resized_frame / 255.0
    # Mở rộng chiều để tạo thành một batch (1, width, height, channels)
    input_tensor = np.expand_dims(normalized_frame, axis=0)
    return input_tensor

def draw_predictions(frame, predictions):
    
    grid_size = [13, 26, 52]
    for i in tf.range(3):
        ac = anchors[i].reshape(-1, 2)
        decoded_preds = decode_predictions(predictions[i], ac, grid_size[i], num_class)[0]
        confidences = decoded_preds[..., 4]
        mask = confidences >= 0.8
        decoded_preds = decoded_preds[mask]
        scores = confidences[mask]

        nms = tf.image.non_max_suppression(
            boxes=decoded_preds[...,0:4], scores=scores, max_output_size=20, iou_threshold=0.5
        )

        for j in nms:
            x,y,w,h = decoded_preds[j][:4] * 416
            c = decoded_preds[j][4]
            p = decoded_preds[j][5:]

            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)

            x_left = int(x - w/2)
            y_top = int(y - h/2)
            x_right = int(x + w/2)
            y_bottom = int(y + h/2)

            nameclass = class_mapping_decoder[np.argmax(p)]

            cv2.rectangle(frame, (x_left,y_top), (x_right,y_bottom), (0,255,0), 2)
            cv2.putText(frame, nameclass, (x_left, y_top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def main():
    # Tải mô hình đã được huấn luyện
    try:
        print("Đang tải mô hình...")
        model = load_model(MODEL_PATH)
        print("Tải mô hình thành công!")
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return

    # Khởi tạo webcam
    # Số 0 thường là webcam tích hợp sẵn. Nếu bạn có nhiều webcam, hãy thử các số khác (1, 2, ...)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Lỗi: Không thể mở webcam.")
        return

    while True:
        # Đọc từng khung hình từ webcam
        ret, frame = cap.read()
        if not ret:
            print("Không thể nhận khung hình. Kết thúc...")
            break

        # Tiền xử lý khung hình
        input_tensor = preprocess_frame(frame)

        # Thực hiện dự đoán
        predictions = model.predict(input_tensor)

        # Vẽ kết quả dự đoán lên khung hình
        # Lưu ý: Cần tùy chỉnh hàm draw_predictions cho phù hợp
        output_frame = draw_predictions(frame, predictions)

        # Hiển thị khung hình kết quả
        cv2.imshow('Webcam Detection', output_frame)

        # Nhấn 'q' để thoát khỏi vòng lặp
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()