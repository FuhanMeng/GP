import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # YOLOv8n summary: 225 layers, 3157200 parameters, 3157184 gradients, 8.9 GFLOPs
    model = YOLO('D:/a桌面文件存放/Git Demo/GP/ultralytics-main/dataset/yolov8n.yaml')
    # model = YOLO('D:/a桌面文件存放/Git Demo/GP/ultralytics-main/dataset/yolov8n-EfficientHead.yaml')
    # model = YOLO('D:/a桌面文件存放/Git Demo/GP/ultralytics-main/dataset/yolov8n-fasternet.yaml')
    # model = YOLO('D:/a桌面文件存放/Git Demo/GP/ultralytics-main/dataset/yolov8n-fasternet-EfficientHead.yaml')

    # prune
    # YOLOv8n-EfficientHead summary: 226 layers, 2424163 parameters, 2424147 gradients, 5.6 GFLOPs
    # model = YOLO('../deploy/yolov8n-efficienthead-lamp-prune-exp-0504-prune4/weights/model_c2f_v2.pt')
    # YOLOv8n-EfficientHead summary: 226 layers, 542800 parameters, 542784 gradients, 2.8 GFLOPs
    # model = YOLO('../deploy/yolov8n-efficienthead-lamp-prune-exp-0504-prune4/weights/prune.pt')
    # YOLOv8n-EfficientHead summary: 203 layers, 542800 parameters, 542784 gradients, 2.8 GFLOPs
    # model = YOLO('../deploy/yolov8n-efficienthead-lamp-prune-exp-0504-prune4/weights/prune_notv2.pt')

    # distill
    # YOLOv8n-EfficientHead summary: 203 layers, 542800 parameters, 0 gradients, 2.8 GFLOPs
    # model = YOLO('../deploy/yolov8n-distill-cwd-0506-exp/weights/best.pt')

    model.info(detailed=True)
    model.profile(imgsz=[640, 640])
    model.fuse()