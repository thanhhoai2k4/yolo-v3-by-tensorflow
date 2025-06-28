import cv2
from yolov3.data_loader import *
from yolov3.config import class_mapping_decoder



def showimage(path_xml):

    path_img, boxes = parse_xml(path_xml)


    img = cv2.imread(path_img)
    height_image, width_image = img.shape[:2]
    for box in boxes:
        x_center, y_center, width, height, id = box

        xmin = int((x_center - width/2) * width_image)
        ymin = int((y_center - height/2) * height_image)
        xmax = int((x_center + width/2) * width_image)
        ymax = int((y_center + height/2) * height_image)

        name_class = class_mapping_decoder[id]
        cv2.rectangle(img, (int(xmin),int(ymin)), (int(xmax),int(ymax)), (0,255,0), 2)
        cv2.putText(img, name_class, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


showimage("data/annotations/4707554000891476.xml")
showimage("data/annotations/0445381710947770.xml")