import os

val_image_dir = "D:/pycharm 1/Py_Projects/ultralytics-8.3.2/coco128/val2017/images"
val_label_dir = "D:/pycharm 1/Py_Projects/ultralytics-8.3.2/coco128/val2017/labels"

image_files = set(os.listdir(val_image_dir))
label_files = set(os.listdir(val_label_dir))

# 检查图像与标签是否一一对应
missing_labels = [f for f in image_files if f.replace(".jpg", ".txt") not in label_files]
if missing_labels:
    print(f"缺失标签文件：{missing_labels[:5]}...")
else:
    print("标签文件完整")