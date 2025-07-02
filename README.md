# yolo-v3-by-tensorflow



## install package
> pip install -r requirements.txt


## Training model: [model.h5](https://drive.google.com/file/d/1s-yeFvDI54ixLgIbfbxGSFRpSGR_buWf/view?usp=sharing)
> python3 training.py
ban có thể sử dụng model của tôi huấn luyện trước để sử dụng inference.

## inference: model.h5
> python3 inference.py


## resultset
![image](RESULT/anh005.png)
![image](RESULT/anh008.png)
![image](RESULT/anh009.png)
![image](RESULT/maksssksksss250.png)
![image](RESULT/maksssksksss757.png)
![image](RESULT/maksssksksss762.png)
![image](RESULT/maksssksksss787.png)
![image](RESULT/maksssksksss788.png)


- AP for class 'with_mask': 0.8531
- AP for class 'without_mask': 0.6263
- AP for class 'mask_weared_incorrect': 0.4489
- Mean Average Precision (mAP) @0.5 IoU: 0.6428

Nhận xét từ chính bản thân:

- Dữ liêu: face-mask-detection ở trên kaggle với dữ liệu khá lệch với with-mask(cao), mithout-mask (trung bình), mask_weared_incorrect(rất thấp). Từ đó khiến mô hình ko dữ đoán chính xác được các vật thể của mask_weared_incorrect. nên sử dụng các kỷ thuật đẻ cân bằng nhãn như phạt nằng vào các nhãn có tỉ lệ thấp và thấp với nhãn có tỉ lệ cao.
- Dữ liệu được load và xử lý bằng numpy. vì thế ko thể tạo đồ thị tính toán của tensorflow. Vì thế lúc khởi động huấn luyện thì nó mất 1 lúc lâu để tạo luồng dữ liệu.
- Loss cho box: là GIOU.

