import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('./runs_test/distill_test/yolov8n-cwd-exp3/weights/best.pt')
    model.val(data='./dataset/person_test/data_test.yaml',
              split='val',
              imgsz=640,
              batch=16,
              # rect=False,
              save_json=False,  # if you need to cal coco metrice
              project='./runs_test/distill_test/yolov8n-cwd-exp3/weights/test',
              name='exp',
              )
