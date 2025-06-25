import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from yolov3.yolo_v3_model import decode_predictions
from yolov3.config import anchors, num_class, class_mapping_decoder
import os

# --- TÙY CHỈNH CÁC THAM SỐ ---
MODEL_PATH = 'model.h5'
VIDEO_PATH = 'RESULT/video/video.mp4'  # <<< THAY ĐƯỜNG DẪN ĐẾN VIDEO CỦA BẠN
OUTPUT_PATH = 'RESULT/video/videoresult.mp4'  # <<< ĐƯỜNG DẪN LƯU VIDEO KẾT QUẢ
MODEL_INPUT_WIDTH = 416
MODEL_INPUT_HEIGHT = 416

### MỚI ###
# Thiết lập khoảng thời gian detection. Ví dụ: 10 nghĩa là cứ 10 frame mới detect một lần.
# Bạn có thể tăng/giảm giá trị này để cân bằng giữa hiệu năng và độ chính xác.
DETECTION_INTERVAL = 100



def inference_XXXX(frame, num_classes, model):

    img = frame
    grid_size = [13, 26, 52]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (416, 416))/255.0
    img_expanded = np.expand_dims(img, axis=0)
    kq = model.predict(img_expanded)

    all = []
    for i in range(3):
        ac = anchors[i].reshape(-1,2)
        decoded_preds = decode_predictions(kq[i], ac, grid_size[i], num_classes)[0]
        confidences = decoded_preds[..., 4]
        mask = confidences >= 0.6
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
    return img
def main():
    try:
        print("Đang tải mô hình...")
        model = load_model(MODEL_PATH)
        print("Tải mô hình thành công!")
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Lỗi: Không thể mở video.")
        return

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


    frame_count = 0
    last_predictions = None

    while True:
        ret , frame = cap.read()
        frame = cv2.resize(frame, (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT))
        if not ret:
            print("Kết thúc video.")
            break

        if (frame_count % DETECTION_INTERVAL) == 0:
            frame = inference_XXXX(frame, num_class, model)
            last_predictions = frame


        frame_count += 1
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()