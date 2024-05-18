import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# onnx onnxsim onnxruntime onnxruntime-gpu

if __name__ == '__main__':
    model = YOLO('./deploy/4yolov8n-pconvhead-distill-cwd-L2-0506-exp/weights/best.pt')  # 直接指定权重文件，不要yaml
    # model.export(format='onnx', simplify=True, opset=13)
    model.export(format='torchscript')
