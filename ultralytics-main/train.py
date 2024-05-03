import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8n.yaml')
    model.load('yolov8n.pt')  # loading pretrain weights
    model.train(data='dataset/data.yaml',
                imgsz=640,
                epochs=100,  #100
                batch=64,  # 16
                close_mosaic=10,  #10
                patience=100,
                cache=True,
                workers=8,  # 8
                device=0,
                optimizer='SGD', # using SGD\Adam, AdamW, NAdam, RAdam, RMSProp
                # save=True,
                # save_period=10,
                # resume=True, # last.pt path
                # lr_0=0.01, # SGD=1E-2，Adam=1E-3
                # cos_lr, #
                # amp=False, # close amp
                # fraction=0.2,
                # profile, # ONNX
                # momentum\weight_decay\warmup.....
                val=True,
                plots=True,
                project='runs/train',
                name='exp_0503',
                )