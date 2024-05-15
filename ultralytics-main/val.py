import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('../deploy/yolov8n-efficienthead-lamp-prune-exp-0504-finetune/weights/best.pt')
    model.val(data='./ultralytics-main/dataset/person_datasets1/data1.yaml',
              split='test',
              imgsz=640,
              batch=16,
              # rect=False,
              save_json=True,  # if you need to cal coco metrice
              project='../deploy',
              name='yolov8n-efficienthead-lamp-prune-exp-0504-finetune-test',
              )
