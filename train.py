import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    #D:\pycharm 1\Py_Projects\ultralytics-8.3.2\ultralytics\cfg\models\11\yolo11.yaml
    #model.load(r'D:\pycharm 1\Py_Projects\ultralytics-8.3.2\yolo11n.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    # model = YOLO(model=r'D:\pycharm 1\Py_Projects\ultralytics-8.3.2\ultralytics\cfg\models\11\yolo11.yaml',task='detect')
    model = YOLO("yolo11n.yaml")
    model.train(data=r'D:\pycharm 1\Py_Projects\ultralytics-8.3.2\coco128.yaml',
                imgsz=640,
                epochs=500,
                batch=8,
                workers=0,
                device=''
                # optimizer='SGD',
                # close_mosaic=10,
                # resume=True,
                # project='runs/train',
                # name='exp',
                # single_cls=False,
                # cache=False
                )