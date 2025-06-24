import os
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm

ANNOTATIONS_PATH  = "../data/annotations"
NUM_CLUSTERS = 9
NETWORK_INPUT_SIZE = 416



def load_all_box_dimensions(path):
    """
    Đọc tất cả các file XML trong thư mục, trích xuất chiều rộng và chiều cao
    của mỗi bounding box.
    """
    box_dims = []
    print(f"Đang đọc các file annotation từ: {path}")
    for filename in tqdm(os.listdir(path)):
        if not filename.endswith('.xml'):
            continue

        filepath = os.path.join(path, filename)

        tree = ET.parse(filepath)
        root = tree.getroot()

        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            width = xmax - xmin
            height = ymax - ymin
            box_dims.append([width, height])

    return np.array(box_dims)


def iou(boxes, clusters):
    """
    Tính toán Intersection over Union (IoU) giữa một tập các box và các tâm cụm.
    - boxes: (N, 2) array, với N là số lượng box.
    - clusters: (K, 2) array, với K là số lượng cụm (anchor).
    """
    n = boxes.shape[0]
    k = clusters.shape[0]

    box_area = boxes[:, 0] * boxes[:, 1]
    # Lặp lại box_area K lần để có shape (N, K)
    box_area = box_area.repeat(k).reshape(n, k)

    cluster_area = clusters[:, 0] * clusters[:, 1]
    # Chuyển vị và lặp lại để có shape (N, K)
    cluster_area = np.tile(cluster_area, [n, 1])

    # Tính diện tích giao nhau (intersection)
    box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
    cluster_w_matrix = np.tile(clusters[:, 0], (n, 1))
    min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

    box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
    cluster_h_matrix = np.tile(clusters[:, 1], (n, 1))
    min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)

    inter_area = min_w_matrix * min_h_matrix

    # Tính diện tích hợp nhất (union)
    union_area = box_area + cluster_area - inter_area

    return inter_area / union_area


def run_kmeans(boxes, k):
    """
    Chạy thuật toán K-means với thước đo 1 - IoU.
    """
    num_boxes = boxes.shape[0]
    distances = np.empty((num_boxes, k))
    last_clusters = np.zeros((num_boxes,))

    # Khởi tạo tâm cụm: chọn ngẫu nhiên k box từ tập dữ liệu
    np.random.seed(42)  # Để kết quả có thể tái tạo
    clusters = boxes[np.random.choice(num_boxes, k, replace=False)]

    print("Bắt đầu chạy K-means...")
    while True:
        # Tính khoảng cách từ mỗi box đến mỗi tâm cụm
        distances = 1 - iou(boxes, clusters)

        # Gán mỗi box vào cụm gần nhất
        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break  # Hội tụ khi không còn box nào thay đổi cụm

        # Cập nhật tâm cụm bằng cách lấy trung vị (median) của các box trong cụm
        for cluster_idx in range(k):
            # Lấy trung vị để tránh bị ảnh hưởng bởi các giá trị ngoại lai
            clusters[cluster_idx] = np.median(boxes[nearest_clusters == cluster_idx], axis=0)

        last_clusters = nearest_clusters

    return clusters


if __name__ == '__main__':
    # 1. Tải tất cả chiều rộng và chiều cao của các box
    all_boxes = load_all_box_dimensions(ANNOTATIONS_PATH)
    print(f"\nTìm thấy tổng cộng {len(all_boxes)} bounding box.")

    # 2. Chạy K-means để tìm ra các anchor
    anchors = run_kmeans(all_boxes, NUM_CLUSTERS)

    # Sắp xếp các anchor theo diện tích từ nhỏ đến lớn
    areas = anchors[:, 0] * anchors[:, 1]
    anchors = anchors[np.argsort(areas)]

    anchors = anchors.reshape(3,3,2)
    anchors = np.flip(anchors, axis=0)
    print(anchors)
