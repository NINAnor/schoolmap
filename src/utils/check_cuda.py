import torch

def check_tensor_cores():
    device_name = torch.cuda.get_device_name()
    tensor_cores_devices = ["V100", "A100", "A40", "T4", "RTX 20", "RTX 30", "RTX 40"]
    
    if any(core in device_name for core in tensor_cores_devices):
        torch.set_float32_matmul_precision("high")
    else:
        print(f"Your CUDA device ('{device_name}') does not appear to have Tensor Cores.")