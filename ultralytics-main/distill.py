import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.distill import DetectionDistiller

if __name__ == '__main__':
    param_dict = {
        # origin
        'model': '/content/GP/deploy/yolov8n-efficienthead-lamp-prune-exp-0504-prune4/weights/prune_notv2.pt',
        'data': '/content/GP/ultralytics-main/dataset/person_datasets1/data1.yaml',
        'imgsz': 640,
        'epochs': 200,
        'batch': 64,
        'workers': 8,
        'cache': True,
        'optimizer': 'SGD',
        'device': '0',
        'close_mosaic': 20,
        'project': '/content/drive/MyDrive/finished1',
        'name': 'yolov8n-distill-cwd-exp-0505',
        
        # distill
        'prune_model': True,
        'teacher_weights': '/content/GP/deploy/exp_0503_s/weights/best.pt',
        'teacher_cfg': '/content/GP/ultralytics-main/dataset/yolov8s-EfficientHead.yaml',
        'kd_loss_type': 'all',
        'kd_loss_decay': 'constant',  # 常量，不进行衰减
        # logical distillation settings
        'logical_loss_type': 'l2',
        'logical_loss_ratio': 1.0,
        # feature distillation settings
        'teacher_kd_layers': '15,18,21',
        'student_kd_layers': '15,18,21',
        'feature_loss_type': 'cwd',
        'feature_loss_ratio': 1.0
    }
    
    model = DetectionDistiller(overrides=param_dict)
    model.distill()
