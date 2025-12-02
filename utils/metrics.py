"""
Metrics computation utilities
"""
import time
import torch
import numpy as np
from ptflops import get_model_complexity_info
import lightning as L

def compute_model_complexity(model: torch.nn.Module, input_size: tuple = (3, 224, 224)):
    """
    Compute model FLOPs and parameters
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (C, H, W)
        
    Returns:
        Dictionary with num_params and flops
    """
    try:
        macs, params = get_model_complexity_info(
            model,
            input_size,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False
        )
        flops = macs * 2
        print(f"✓ Model Parameters: {params:,}")
        print(f"✓ FLOPs: {flops:,}")
        return {"num_params": params, "flops": flops}
    except Exception as e:
        print(f"⚠ FLOPs estimation failed: {e}")
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model Parameters (fallback): {num_params:,}")
        return {"num_params": num_params, "flops": 0}


def measure_inference_latency(
    model: L.LightningModule,
    dataloader: torch.utils.data.DataLoader,
    num_batches: int = 50,
    warmup_batches: int = 5
):
    """
    Measure inference latency
    
    Args:
        model: Lightning module
        dataloader: Test dataloader
        num_batches: Number of batches to measure
        warmup_batches: Number of warmup batches
        
    Returns:
        Dictionary with avg and std latency in milliseconds
    """
    device = next(model.parameters()).device
    model.eval()
    
    print("\nMeasuring inference latency...")
    
    # Warmup
    with torch.inference_mode():
        for i, (x, _) in enumerate(dataloader):
            if i >= warmup_batches:
                break
            x = x.to(device)
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
    
    # Measurement
    latencies = []
    with torch.inference_mode():
        for i, (x, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            
            x = x.to(device)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = model(x)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            latencies.append((end - start) / x.size(0))
    
    if latencies:
        avg_latency_ms = np.mean(latencies) * 1000
        std_latency_ms = np.std(latencies) * 1000
        print(f"✓ Average Inference Latency: {avg_latency_ms:.2f} ± {std_latency_ms:.2f} ms/sample")
        return {
            "avg_inference_latency_ms": avg_latency_ms,
            "std_inference_latency_ms": std_latency_ms
        }
    else:
        return {
            "avg_inference_latency_ms": float('nan'),
            "std_inference_latency_ms": float('nan')
        }
