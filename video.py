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
DETECTION_INTERVAL = 10



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
    if all.shape[0] == 0:
        return []
    nms = tf.image.non_max_suppression(
        boxes=all[..., 0:4], scores=all[...,4], max_output_size=20, iou_threshold=0.5
    )

    final_predictions = []
    for index in nms:
        prediction = all[index]
        final_predictions.append(prediction)
    return final_predictions

def draw_predictions(frame, predictions, width_video_scale, height_video_scale):
    for row in predictions:
        x,y,w,h = row[:4] * MODEL_INPUT_WIDTH
        c = row[4]
        p = row[5:]
        xmin = int((x - w/2)*width_video_scale)
        ymin = int((y - h/2)*height_video_scale)
        xmax = int((x + w/2)*width_video_scale)
        ymax = int((y + h/2)*height_video_scale)
        nameclass = class_mapping_decoder[np.argmax(p)]


        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, nameclass+ " : " + str(c), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame


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
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width_video = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_video = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, input_fps, (width_video, height_video))

    width_video_scale = width_video / MODEL_INPUT_WIDTH
    height_video_scale = height_video / MODEL_INPUT_HEIGHT
    frame_count = 0
    last_predictions = None

    while True:
        ret , frame = cap.read()

        if not ret:
            print("Kết thúc video.")
            break

        if (frame_count % DETECTION_INTERVAL) == 0:
            final_predictions = inference_XXXX(frame, num_class, model)
            last_predictions = final_predictions
            frame = draw_predictions(frame, final_predictions, width_video_scale, height_video_scale)
        else:
            frame = draw_predictions(frame, last_predictions, width_video_scale, height_video_scale)


        frame_count += 1
        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()