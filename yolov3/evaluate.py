import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from yolov3.yolo_v3_model import decode_predictions
from yolov3.data_loader import parse_xml
from yolov3.config import anchors, num_class, class_ids, image_width, image_height, class_mapping_decoder
from yolov3.losses import compute_iou_for_yolo

# --- CẤU HÌNH ---
MODEL_PATH = 'model.h5'
VAL_ANNOTATIONS_DIR = 'val/annotations'
IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5  # Ngưỡng tin cậy để xem xét một dự đoán


def get_model_predictions(model, image_path):
    """Thực hiện suy luận trên một ảnh và trả về các dự đoán đã được giải mã."""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (image_width, image_height)) / 255.0
    img_expanded = np.expand_dims(img_resized, axis=0)

    raw_predictions = model.predict(img_expanded)

    grid_sizes = [13, 26, 52]
    all_decoded_preds = []

    for i in range(3):
        anchor_group = anchors[i].reshape(-1, 2)
        decoded_preds = decode_predictions(raw_predictions[i], anchor_group, grid_sizes[i], num_class)[0]
        all_decoded_preds.append(decoded_preds)

    all_decoded_preds = np.concatenate(all_decoded_preds, axis=0)

    # Áp dụng Non-Max Suppression
    boxes = all_decoded_preds[..., 0:4]
    confidences = all_decoded_preds[..., 4]

    # Lọc theo ngưỡng tin cậy ban đầu
    mask = confidences >= CONFIDENCE_THRESHOLD
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_probs = all_decoded_preds[mask][..., 5:]

    if len(boxes) == 0:
        return [], [], []

    # Chuyển đổi boxes từ (center_x, center_y, w, h) sang (y1, x1, y2, x2) cho NMS
    cy, cx, h, w = boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]
    y1 = cy - h / 2
    x1 = cx - w / 2
    y2 = cy + h / 2
    x2 = cx + w / 2
    nms_boxes = tf.stack([y1, x1, y2, x2], axis=-1)

    selected_indices = tf.image.non_max_suppression(
        boxes=nms_boxes,
        scores=confidences,
        max_output_size=50,
        iou_threshold=IOU_THRESHOLD
    )

    final_boxes = tf.gather(boxes, selected_indices).numpy()
    final_scores = tf.gather(confidences, selected_indices).numpy()
    final_class_ids = tf.argmax(tf.gather(class_probs, selected_indices), axis=-1).numpy()

    return final_boxes, final_scores, final_class_ids


def calculate_ap(rec, prec):
    """Tính toán AP từ precision và recall."""
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # Đảm bảo precision không giảm
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Tìm các điểm thay đổi trên trục x (recall)
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Tính diện tích dưới đường cong
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def main():
    # Tải mô hình
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Tải mô hình thành công!")
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return

    # Lấy danh sách file validation
    val_xml_files = [f for f in os.listdir(VAL_ANNOTATIONS_DIR) if f.endswith('.xml')]

    # Lưu trữ tất cả các dự đoán và ground truths
    all_predictions = {class_id: [] for class_id in range(num_class)}
    all_ground_truths = {class_id: 0 for class_id in range(num_class)}

    print("Bắt đầu đánh giá trên tập validation...")
    for xml_file in tqdm(val_xml_files):
        xml_path = os.path.join(VAL_ANNOTATIONS_DIR, xml_file)
        image_path, gt_boxes = parse_xml(xml_path, trainorval=False)

        if not os.path.exists(image_path):
            print(f"Cảnh báo: Không tìm thấy ảnh {image_path} cho file {xml_file}")
            continue

        # Đếm số lượng ground truth cho mỗi lớp
        for gt_box in gt_boxes:
            class_id = int(gt_box[4])
            all_ground_truths[class_id] += 1

        # Lấy dự đoán từ mô hình
        pred_boxes, pred_scores, pred_class_ids = get_model_predictions(model, image_path)

        # Lưu trữ các dự đoán
        for i in range(len(pred_boxes)):
            class_id = pred_class_ids[i]
            # [confidence, box, image_filename]
            all_predictions[class_id].append([pred_scores[i], pred_boxes[i], xml_file])

    # Tính toán AP cho mỗi lớp
    average_precisions = {}
    for class_id, class_name in class_mapping_decoder.items():
        predictions = sorted(all_predictions[class_id], key=lambda x: x[0], reverse=True)
        num_gt = all_ground_truths[class_id]

        if num_gt == 0:
            average_precisions[class_name] = 0.0 if not predictions else 0.0
            continue

        tp = np.zeros(len(predictions))
        fp = np.zeros(len(predictions))

        # Theo dõi các ground truth đã được khớp
        gt_matched = {}

        for i, (confidence, pred_box, img_file) in enumerate(predictions):
            # Lấy ground truths cho ảnh hiện tại
            _, current_gt_boxes = parse_xml(os.path.join(VAL_ANNOTATIONS_DIR, img_file), trainorval=False)

            # Chỉ lấy các gt box của lớp hiện tại
            class_gt_boxes = [box[:4] for box in current_gt_boxes if int(box[4]) == class_id]

            best_iou = 0
            best_gt_idx = -1

            if len(class_gt_boxes) > 0:
                # Tìm iou tốt nhất với các ground truth box
                for j, gt_box in enumerate(class_gt_boxes):
                    iou = compute_iou_for_yolo(np.expand_dims(pred_box, axis=0), np.expand_dims(gt_box, axis=0))[0]
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

            # Kiểm tra xem có khớp không
            if best_iou >= IOU_THRESHOLD:
                # Kiểm tra xem ground truth này đã được khớp trước đó chưa
                if gt_matched.get((img_file, best_gt_idx)) is None:
                    tp[i] = 1.
                    gt_matched[(img_file, best_gt_idx)] = True
                else:
                    fp[i] = 1.
            else:
                fp[i] = 1.

        # Tính precision và recall
        fp_cumsum = np.cumsum(fp)
        tp_cumsum = np.cumsum(tp)

        recall = tp_cumsum / (num_gt + 1e-16)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)

        ap = calculate_ap(recall, precision)
        average_precisions[class_name] = ap

    # Tính mAP
    mean_ap = np.mean(list(average_precisions.values()))

    print("\n--- KẾT QUẢ ĐÁNH GIÁ ---")
    for class_name, ap in average_precisions.items():
        print(f"AP for class '{class_name}': {ap:.4f}")

    print(f"\nMean Average Precision (mAP) @{IOU_THRESHOLD} IoU: {mean_ap:.4f}")


if __name__ == '__main__':
    main()