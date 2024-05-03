import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.distill import DetectionDistiller

if __name__ == '__main__':
    param_dict = {
        # origin
        'model': 'ultralytics/cfg/models/v8/yolov8n.pt',
        'data': '/home/hjj/Desktop/dataset/dataset_visdrone/data_exp.yaml',
        'imgsz': 640,
        'epochs': 200,
        'batch': 32,
        'workers': 8,
        'cache': True,
        'optimizer': 'SGD',
        'device': '0',
        'close_mosaic': 20,
        'project': 'runs/distill',
        'name': 'yolov8n-cwd-exp3-0503',
        
        # distill
        'prune_model': True,
        'teacher_weights': 'runs/train/yolov8s/weights/best.pt',
        'teacher_cfg': 'ultralytics/cfg/models/v8/yolov8s.yaml',
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