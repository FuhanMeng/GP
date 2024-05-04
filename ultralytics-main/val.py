import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('./deploy/exp_0503_s-20240503T141606Z-001/exp_0503_s/weights/best.pt')
    model.val(data='./ultralytics-main/dataset/person_datasets1/data1.yaml',
              split='val',
              imgsz=640,
              batch=16,
              # rect=False,
              save_json=True,  # if you need to cal coco metrice
              project='./runs_test/distill_test/yolov8n-cwd-exp3/weights/test',
              name='exp',
              )
