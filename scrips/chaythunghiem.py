from pathlib import Path
import shutil
from xml.etree import ElementTree as ET



def converFolderToStringimages(folder = "data/annotations"):
    folder = Path(folder)
    xmls = list(folder.glob('*.xml'))

    count = 0

    for xml in xmls:
        tree = ET.parse(xml)
        root = tree.getroot()
        folderElement = root.find('folder')
        if folderElement.text != "images":
            folderElement.text = "images"
            count += 1
            tree.write(xml, encoding="utf-8", xml_declaration=True)
            print(f"\nĐã ghi các thay đổi vào file: {xml}")
        else:
             continue
    print("So luong da thay doi:" , count)
converFolderToStringimages()