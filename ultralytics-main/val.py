import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs_test/prune_test/runs_test_eh10/yolov8n-EfficientHead10-lamp-exp-finetune/weights/best_notv2.pt')
    model.val(data='./dataset/person_test/data_test.yaml',
              split='val',
              imgsz=640,
              batch=16,
              # rect=False,
              save_json=False,  # if you need to cal coco metrice
              #project='runs/val',
              #name='exp',
              )