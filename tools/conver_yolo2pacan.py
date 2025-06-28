import os
import cv2
import xml.etree.ElementTree as ET
from xml.dom import minidom
from yolov3.config import class_ids, class_mapping_decoder
import shutil
from tqdm import tqdm

def create_xml_annotation(image_filename, widthz, heightz, objects, output_folder):

    os.makedirs(output_folder, exist_ok=True)
    annotation = ET.Element('annotation')

    folder = ET.SubElement(annotation, 'folder')
    folder.text = "images"

    filename = ET.SubElement(annotation, 'filename')
    filename.text = image_filename

    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width')
    height = ET.SubElement(size, 'height')
    depth = ET.SubElement(size, 'depth')
    width.text = str(widthz)
    height.text = str(heightz)
    depth.text = "3"



    # ghi nhung object
    for obj in objects:
        x_center, y_center, width, height, id = obj

        className = class_mapping_decoder[id] # ten cua lop

        xmin_number = int(x_center - width / 2)
        ymin_number = int(y_center - height / 2)
        xmax_number = int(x_center + width / 2)
        ymax_number = int(y_center + height / 2)

        if xmax_number > xmin_number and ymax_number > ymin_number:
            # tao 1 the co ten object
            obj_xml = ET.SubElement(annotation, 'object')

            name = ET.SubElement(obj_xml, 'name')
            name.text = className

            bndbox = ET.SubElement(obj_xml, 'bndbox')

            xmin = ET.SubElement(bndbox, 'xmin')
            ymin = ET.SubElement(bndbox, 'ymin')

            xmax = ET.SubElement(bndbox, 'xmax')
            ymax = ET.SubElement(bndbox, 'ymax')

            xmin.text = str(xmin_number)
            ymin.text = str(ymin_number)

            xmax.text = str(xmax_number)
            ymax.text = str(ymax_number)
        else:
            continue

    # Ghi file XML
    xml_str = ET.tostring(annotation)
    # Định dạng lại XML cho đẹp mắt
    pretty_xml_str = minidom.parseString(xml_str).toprettyxml(indent="    ")

    xml_filename  = os.path.splitext(image_filename)[0] + '.xml'
    xml_path  = os.path.join(output_folder, xml_filename)
    with open(xml_path, 'w') as f:
        f.write(pretty_xml_str)


def convert_yolo_to_pascal_voc(xml_folder="data/annotations", image_target_folder="data/images", txt_folder="archive/labels/train", image_folder = "archive/images/train"):

    class_names = class_ids

    # chua tao thi tao con tao roi thi thoi
    os.makedirs(xml_folder, exist_ok=True)
    os.makedirs(image_target_folder, exist_ok=True)

    ds_thumuc_txt = os.listdir(txt_folder) # debug thi copy: set value: ["0000bee39176697a.txt","0000eda1171fe14e.txt"]
    for txt_filename in tqdm(ds_thumuc_txt):
        if not txt_filename.endswith(".txt"):
            continue

        txt_path = os.path.join(txt_folder, txt_filename) # duong dan den file txt
        base_filename = os.path.splitext(txt_filename)[0] # lay ten cua file ko lay duoi
        image_path = None

        # tim ra duong dan cu anh
        for ext in [".jpg", ".jpeg", ".png"]:
            potential_path = os.path.join(image_folder, base_filename + ext)
            if os.path.exists(potential_path):
                image_path = potential_path
                break

        # doc anh
        image = cv2.imread(image_path)
        if image is None:
            print(f"Lỗi: Không thể đọc file ảnh {image_path}")
            continue

        (height,width, _) = image.shape # height width channels
        objects = []

        with open(txt_path, "r")  as f:
            for line in f.readlines():
                parts = line.strip().split(" ")

                class_id = int(parts[0])
                x_center = float(parts[1]) * width
                y_center = float(parts[2]) * height
                w_box = float(parts[3]) * width
                h_box = float(parts[4]) * height

                row = [x_center, y_center, w_box, h_box, class_id]
                objects.append(row)

        source_image_path = image_path
        destination_folder = os.path.join(image_target_folder, os.path.basename(image_path))
        shutil.copy(source_image_path, destination_folder)
        create_xml_annotation(os.path.basename(image_path), width, height, objects,xml_folder)

convert_yolo_to_pascal_voc(
    xml_folder="data/annotations",
    image_target_folder="data/images",
    txt_folder="archive/labels/train",
    image_folder = "archive/images/train")


convert_yolo_to_pascal_voc(
    xml_folder="val/annotations",
    image_target_folder="val/images",
    txt_folder="archive/labels/val",
    image_folder = "archive/images/val")