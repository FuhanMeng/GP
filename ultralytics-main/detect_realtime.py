import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/exp/weights/best.pt')  # select your model.pt path
    model.predict(source=0,
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  conf=0.25,
                  iou=0.7,
                  # 推理论据
                  # 可视化参数
                  show=True,
                  show_labels=True,
                  save=True,
                  show_conf=True,
                  show_boxes=True,
                  # visualize=True # visualize model features maps
                )