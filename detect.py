# -*- coding: utf-8 -*-
"""
@Auth ： 挂科边缘
@File ：detect.py
@IDE ：PyCharm
@Motto:学习新思想，争做新青年
@Email ：179958974@qq.com
"""

from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'D:\pycharm 1\Py_Projects\ultralytics-8.3.2\runs\train\exp32\weights\best.pt')
    model.predict(source=r'D:\pycharm 1\Py_Projects\ultralytics-8.3.2\20241002024400.mp4',
                  save=True,
                  show=True,
                  )
#
# from ultralytics import YOLO
# model = YOLO(model=r'D:\pycharm 1\Py_Projects\ultralytics-8.3.2\runs\train\exp32\weights\best.pt')
# results = model.predict(source=r'D:\pycharm 1\Py_Projects\ultralytics-8.3.2\output\test\images\20200416005155-O.jpg',  # 如图片路径、视频路径、0表示摄像头等
#                         verbose=True)  # 可设置是否打印详细信息
# # 访问results中的speed属性获取时间信息，进而计算FPS
# if hasattr(results, 'names'):  # 确保推理成功有结果
#     speed_dict = results[0].speed
#     inference_time = speed_dict['inference']  # 推理时间（单位：ms）
#     fps = 1000 / inference_time  # 换算成每秒帧数
#     print(f"FPS: {fps}")