import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('./deploy/1yolov8n-0517-exp/weights/best.pt')  # select your model.pt path
    model.predict(source='./dataset/person_datasets1/images/train/s3764.jpg',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  save=True,
                  # conf=0.2,
                  # visualize=True # visualize model features maps
                )