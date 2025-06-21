import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path



def create_structured_validation_set(source_annot_dir, source_image_dir, val_annot_dir, val_image_dir, val_split=0.2):
    """
    Di chuyển một phần tệp từ các thư mục nguồn có cấu trúc (annotations, images)
    sang các thư mục validation tương ứng.

    Args:
        source_annot_dir (str): Thư mục chứa annotation nguồn.
        source_image_dir (str): Thư mục chứa ảnh nguồn.
        val_annot_dir (str): Thư mục chứa annotation đích (validation).
        val_image_dir (str): Thư mục chứa ảnh đích (validation).
        val_split (float): Tỷ lệ dữ liệu cần chuyển.
    """
    # Chuyển đổi đường dẫn sang đối tượng Path
    src_annot_path = Path(source_annot_dir)
    src_img_path = Path(source_image_dir)
    val_annot_path = Path(val_annot_dir)
    val_img_path = Path(val_image_dir)

    # Tạo các thư mục đích nếu chưa có
    val_annot_path.mkdir(parents=True, exist_ok=True)
    val_img_path.mkdir(parents=True, exist_ok=True)
    print(f"Các thư mục đích '{val_annot_path}' và '{val_img_path}' đã sẵn sàng.")

    # Lấy danh sách tệp XML từ thư mục nguồn
    all_xml_files = list(src_annot_path.glob('*.xml'))
    if not all_xml_files:
        print(f"Lỗi: Không tìm thấy tệp XML nào trong '{src_annot_path}'")
        return

    random.shuffle(all_xml_files)
    num_to_move = int(len(all_xml_files) * val_split)
    files_to_move = all_xml_files[:num_to_move]

    print(f"\nTổng số cặp file: {len(all_xml_files)}")
    print(f"Sẽ di chuyển {num_to_move} cặp vào thư mục validation...")

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    moved_count = 0

    for xml_path in files_to_move:
        # Tìm ảnh tương ứng trong thư mục ảnh nguồn
        found_image = False
        for ext in image_extensions:
            image_path = src_img_path / xml_path.with_suffix(ext).name
            if image_path.exists():
                # Đường dẫn đích
                dest_xml_path = val_annot_path / xml_path.name
                dest_image_path = val_img_path / image_path.name

                # Di chuyển file
                shutil.move(str(image_path), str(dest_image_path))
                shutil.move(str(xml_path), str(dest_xml_path))


                moved_count += 1
                found_image = True
                break

        if not found_image:
            print(f"Cảnh báo: Không tìm thấy ảnh cho tệp {xml_path.name}")

    print(f"\nHoàn tất! Đã di chuyển {moved_count} cặp ảnh/XML.")


if __name__ == '__main__':
    # ----- CẤU HÌNH -----
    # Đường dẫn đến các thư mục dữ liệu huấn luyện
    SOURCE_ANNOTATIONS = 'data/annotations'
    SOURCE_IMAGES = 'data/images'

    # Đường dẫn đến các thư mục validation
    VAL_ANNOTATIONS = 'val/annotations'
    VAL_IMAGES = 'val/images'

    # Tỷ lệ dữ liệu validation (ví dụ: 0.2 là 20%)
    VALIDATION_SPLIT = 0.2
    # -------------------

    create_structured_validation_set(
        SOURCE_ANNOTATIONS,
        SOURCE_IMAGES,
        VAL_ANNOTATIONS,
        VAL_IMAGES,
        val_split=VALIDATION_SPLIT
    )