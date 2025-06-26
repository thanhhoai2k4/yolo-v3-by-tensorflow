import os
import xml.etree.ElementTree as ET


def rename_labels_in_xml_files(directory_path):
    """
    Duyệt qua tất cả các tệp XML trong một thư mục, tìm các đối tượng có nhãn
    'mask_weared_incorrect' và đổi tên chúng thành 'with_mask'.

    Args:
        directory_path (str): Đường dẫn đến thư mục chứa các tệp XML.
    """
    # Biến đếm để theo dõi số lượng file và nhãn đã thay đổi
    changed_files_count = 0
    changed_labels_count = 0

    print(f"Bắt đầu quá trình quét thư mục: {directory_path}\n")

    # Lặp qua tất cả các tệp trong thư mục được cung cấp
    for filename in os.listdir(directory_path):
        if not filename.endswith('.xml'):
            continue  # Bỏ qua nếu không phải file XML

        file_path = os.path.join(directory_path, filename)

        try:
            # Phân tích cú pháp tệp XML
            tree = ET.parse(file_path)
            root = tree.getroot()

            file_was_changed = False

            # Tìm tất cả các thẻ 'object' trong file
            for obj in root.findall('object'):
                name_tag = obj.find('name')
                if name_tag is not None and name_tag.text == 'mask_weared_incorrect':
                    # Thay đổi nội dung của thẻ <name>
                    print(f"  -> Tìm thấy '{name_tag.text}' trong file: {filename}. Đang đổi thành 'with_mask'.")
                    name_tag.text = 'with_mask'

                    # Đánh dấu rằng file này đã bị thay đổi và tăng biến đếm
                    file_was_changed = True
                    changed_labels_count += 1

            # Nếu file có sự thay đổi, lưu lại nội dung mới
            if file_was_changed:
                tree.write(file_path, encoding='utf-8')
                changed_files_count += 1

        except ET.ParseError as e:
            print(f"Lỗi: Không thể phân tích cú pháp tệp {filename}. Lỗi: {e}")
        except Exception as e:
            print(f"Lỗi không xác định khi xử lý tệp {filename}. Lỗi: {e}")

    print("\n----- HOÀN TẤT -----")
    print(f"Tổng số tệp đã được thay đổi: {changed_files_count}")
    print(f"Tổng số nhãn đã được đổi tên: {changed_labels_count}")
    print("--------------------")


# --- CÁCH SỬ DỤNG ---
# 1. Thay thế 'duong_dan_den_thu_muc_xml_cua_ban' bằng đường dẫn thực tế.
#    Ví dụ: 'data/annotations'
# 2. Chạy đoạn mã.

if __name__ == '__main__':
    # THAY ĐỔI ĐƯỜNG DẪN NÀY
    path_to_your_annotations = 'data/annotations/'

    # Kiểm tra xem đường dẫn có tồn tại không
    if not os.path.isdir(path_to_your_annotations):
        print(f"Lỗi: Thư mục '{path_to_your_annotations}' không tồn tại.")
        print("Vui lòng cập nhật biến 'path_to_your_annotations' với đường dẫn chính xác.")
    else:
        rename_labels_in_xml_files(path_to_your_annotations)