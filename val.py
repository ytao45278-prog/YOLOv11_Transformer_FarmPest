# import warnings
#
# warnings.filterwarnings('ignore')
# from ultralytics import YOLO
#
# if __name__ == '__main__':
#     model = YOLO(r'D:\pycharm 1\Py_Projects\ultralytics-8.3.2\runs\train\exp33\weights\best.pt')
#     model.val(data=r'D:\pycharm 1\Py_Projects\ultralytics-8.3.2\mydataori.yaml',
#               imgsz=640,
#               batch=16,
#               split='test',
#               workers=2,
#               device='0',
#               )

# import warnings
#
# warnings.filterwarnings('ignore')
# from ultralytics import YOLO
#
# if __name__ == '__main__':
#     model = YOLO(r'D:\pycharm 1\Py_Projects\ultralytics-8.3.2\runs\train\exp32\weights\best.pt')
#     model.val(data=r'D:\pycharm 1\Py_Projects\ultralytics-8.3.2\data.yaml',
#               imgsz=640,
#               batch=16,
#               split='test',
#               workers=2,
#               device='0',
#               )
import warnings
import pandas as pd

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载模型
    model = YOLO(r'D:\pycharm 1\Py_Projects\ultralytics-8.3.2\runs\train\exp33\weights\best.pt')

    # 运行验证
    results = model.val(
        data=r'D:\pycharm 1\Py_Projects\ultralytics-8.3.2\mydataori.yaml',
        imgsz=640,
        batch=16,
        split='test',
        workers=2,
        device='0',
    )

    # 提取混淆矩阵数据
    cm = results.confusion_matrix.matrix

    # 保存为 CSV 文件
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv('confusion_matrix.csv', index=False, header=False)

    print("混淆矩阵已保存为 confusion_matrix.csv")