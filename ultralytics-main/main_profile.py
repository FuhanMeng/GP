import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # choose your yaml file
    model = YOLO('D:/a桌面文件存放/Git Demo/GP/ultralytics-main/dataset/yolov8n.yaml')
    # model = YOLO('D:/a桌面文件存放/Git Demo/GP/ultralytics-main/dataset/yolov8n-EfficientHead.yaml')
    # model = YOLO('D:/a桌面文件存放/Git Demo/GP/ultralytics-main/dataset/yolov8n-fasternet.yaml')
    # model = YOLO('D:/a桌面文件存放/Git Demo/GP/ultralytics-main/dataset/yolov8n-fasternet-EfficientHead.yaml')
    model.info(detailed=True)
    model.profile(imgsz=[640, 640])
    model.fuse()