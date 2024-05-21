import time
import torch
import argparse
from torchvision import transforms
from PIL import Image
import numpy as np
import onnxruntime as ort
import os
import platform
from tqdm import tqdm

# Load PyTorch model
def load_pt_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    if 'model' in checkpoint:
        model = checkpoint['model']
    else:
        model = checkpoint
    model = model.to(device)
    model.eval()
    return model

# Load TorchScript model
def load_torchscript_model(model_path, device):
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model

# Load ONNX model
def load_onnx_model(model_path):
    session = ort.InferenceSession(model_path)
    return session

# Preprocess image
def preprocess_image(image, img_size):
    preprocess = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)

# Run model
def run_model(model, model_type, input_tensor):
    if model_type == 'pt' or model_type == 'torchscript':
        with torch.no_grad():
            output = model(input_tensor)
    elif model_type == 'onnx':
        input_name = model.get_inputs()[0].name
        if input_tensor.is_cuda:
            input_tensor = input_tensor.cpu()  # 将CUDA张量移到CPU上
        output = model.run(None, {input_name: input_tensor.numpy()})
    else:
        raise ValueError("Unsupported model type. Use 'pt', 'torchscript', or 'onnx'.")
    return output

# Measure FPS and Latency
def measure_performance(model, model_type, input_tensor, warmup, testtime):
    # Warm up
    print("begin warmup...")
    for _ in tqdm(range(warmup), desc="warmup", ncols=100):
        run_model(model, model_type, input_tensor)

    print("begin test latency...")
    start_time = time.time()
    latencies = []
    for _ in tqdm(range(testtime), desc="test latency", ncols=100):
        start = time.time()
        run_model(model, model_type, input_tensor)
        end = time.time()
        latencies.append(end - start)
    end_time = time.time()

    avg_latency = sum(latencies) / len(latencies)
    std_latency = np.std(latencies)
    avg_time_per_run = (end_time - start_time) / testtime
    fps = 1 / avg_time_per_run
    return avg_latency, std_latency, fps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./deploy/4yolov8n-pconvhead-distill-cwd-L2-0506-exp/weights/best.onnx',
                        help='trained weights path')  # Updated model path
    parser.add_argument('--batch', type=int, default=8, help='total batch size for all GPUs')  # Updated batch size
    parser.add_argument('--imgs', nargs='+', type=int, default=[640, 640], help='[height, width] image sizes')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--warmup', default=20, type=int, help='warmup time')  # Updated warmup time
    parser.add_argument('--testtime', default=100, type=int, help='test time')  # Updated test time
    parser.add_argument('--half', action='store_true', default=False, help='fp16 mode.')
    opt = parser.parse_args()

    # Print environment information
    print(f"Ultralytics YOLOv8 🚀 Python-{platform.python_version()} torch-{torch.__version__} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Set device
    if not opt.device:
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(opt.device)

    if opt.half:
        torch.set_default_dtype(torch.float16)

    # Create a 640x640 image with random content
    img_size = opt.imgs
    image = Image.fromarray(np.uint8(np.random.rand(img_size[0], img_size[1], 3) * 255))

    input_tensor = preprocess_image(image, img_size).to(device)

    # Load model
    model_type = None
    if opt.weights.endswith('.pt'):
        model_type = 'pt'
        model = load_pt_model(opt.weights, device)
    elif opt.weights.endswith('.torchscript'):
        model_type = 'torchscript'
        model = load_torchscript_model(opt.weights, device)
    elif opt.weights.endswith('.onnx'):
        model_type = 'onnx'
        model = load_onnx_model(opt.weights)
    else:
        raise ValueError("Unsupported model type. Use a .pt, .torchscript, or .onnx file.")

    print(f"Loaded {opt.weights}")

    # Measure performance
    avg_latency, std_latency, fps = measure_performance(model, model_type, input_tensor, opt.warmup, opt.testtime)
    model_size = os.path.getsize(opt.weights) / (1024 * 1024)  # Model size in MB

    print(f"model weights: {opt.weights} size: {model_size:.2f} MB (bs: {opt.batch})")
    print(f"Latency: {avg_latency:.5f}s ± {std_latency:.5f}s fps: {fps:.1f}")
