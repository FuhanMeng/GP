import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# onnx onnxsim onnxruntime onnxruntime-gpu

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')  # 直接指定权重文件，不要yaml
    model.export(format='onnx', simplify=True, opset=13)
