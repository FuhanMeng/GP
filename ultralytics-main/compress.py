import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.compress import DetectionCompressor, DetectionFinetune

def compress(param_dict):
    with open(param_dict['sl_hyp'], errors='ignore') as f:
        sl_hyp = yaml.safe_load(f)
    param_dict.update(sl_hyp)
    param_dict['name'] = f'{param_dict["name"]}-prune'
    param_dict['patience'] = 0
    compressor = DetectionCompressor(overrides=param_dict)
    prune_model_path = compressor.compress()
    return prune_model_path

def finetune(param_dict, prune_model_path):
    param_dict['model'] = prune_model_path
    param_dict['name'] = f'{param_dict["name"]}-finetune'
    trainer = DetectionFinetune(overrides=param_dict)
    trainer.train()

if __name__ == '__main__':
    param_dict = {
        # origin
        'model': 'D:/a桌面文件存放/Git Demo/GP/ultralytics-main/runs_test_fn_eh/best.pt',
        'data':'D:/a桌面文件存放/Git Demo/GP/ultralytics-main/dataset/person_test/data_test.yaml',
        'imgsz': 640,
        'epochs': 10,  # 250-300
        'batch': 1,  # 16
        'workers': 1,  # 8
        'cache': False,  # True
        'optimizer': 'SGD',
        'device': 'cpu',
        'close_mosaic': 0,
        'project':'ultralytics-main/runs_test_fn_eh/prune',
        'name':'yolov8n-fasternet-EfficientHead-lamp-exp0',
        
        # prune
        'prune_method':'lamp',
        'global_pruning': True,
        'speed_up': 1.1,
        'reg': 0.0005,
        'sl_epochs': 10,
        'sl_hyp': 'D:/a桌面文件存放/Git Demo/GP/ultralytics-main/ultralytics/cfg/hyp.scratch.sl.yaml',
        'sl_model': None,
    }
    
    prune_model_path = compress(copy.deepcopy(param_dict))
    finetune(copy.deepcopy(param_dict), prune_model_path)