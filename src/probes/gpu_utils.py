"""
GPU detection and configuration utilities
Supports CUDA, AMD ROCm, and Apple Metal Performance Shaders (MPS)
"""

import os
import sys
import subprocess
from typing import Tuple, Optional


def detect_device() -> Tuple[str, Optional[str]]:
    """
    Detect the best available compute device.

    Returns:
        Tuple of (device_type, device_info)
        device_type: "cuda", "mps", "cpu", or "rocm"
        device_info: Optional string with device details
    """
    # Import torch here to check device availability
    try:
        import torch

        # Check for CUDA (NVIDIA GPUs)
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            return "cuda", f"CUDA device: {device_name}"

        # Check for MPS (Apple Silicon M1/M2/M3/M4)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps", "Apple Metal Performance Shaders (MPS)"

    except ImportError:
        pass

    # Check for AMD ROCm
    if detect_amd_gpu():
        return "rocm", "AMD GPU with ROCm support"

    # Default to CPU
    return "cpu", "CPU only (no GPU acceleration)"


def detect_amd_gpu() -> bool:
    """
    Detect if an AMD GPU is present in the system.

    Returns:
        True if AMD GPU detected, False otherwise
    """
    try:
        # Check for rocm-smi (ROCm System Management Interface)
        result = subprocess.run(['rocm-smi'], capture_output=True, text=True, timeout=2)
        if result.returncode == 0 and 'AMD' in result.stdout:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    try:
        # Check lspci for AMD GPU
        result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=2)
        if 'VGA compatible controller: Advanced Micro Devices' in result.stdout or \
           'Display controller: Advanced Micro Devices' in result.stdout:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return False


def configure_amd_gpu() -> None:
    """
    Configure AMD GPU environment variables if an AMD GPU is detected.
    Must be called before importing torch.
    """
    if detect_amd_gpu():
        print("AMD GPU detected - configuring ROCm environment variables")
        # PyTorch is compiled for gfx1100, not gfx1101
        # Override to use gfx1100 kernels for gfx1101 GPU (RX 7700/7800 XT)
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
        os.environ["HIP_VISIBLE_DEVICES"] = "0"
        os.environ["AMD_SERIALIZE_KERNEL"] = "3"
        os.environ["TORCH_USE_HIP_DSA"] = "1"
        # Force use of gfx1100 architecture (closest match in PyTorch)
        os.environ["PYTORCH_ROCM_ARCH"] = "gfx1100"
        # Enable expandable memory segments for better memory management
        os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
        # Disable async kernel launches for better error reporting
        os.environ["HIP_LAUNCH_BLOCKING"] = "1"

        print(f"  HSA_OVERRIDE_GFX_VERSION: {os.environ['HSA_OVERRIDE_GFX_VERSION']}")
        print(f"  PYTORCH_ROCM_ARCH: {os.environ['PYTORCH_ROCM_ARCH']}")
        print(f"  TORCH_USE_HIP_DSA: {os.environ['TORCH_USE_HIP_DSA']}")
        print(f"  HIP_LAUNCH_BLOCKING: {os.environ['HIP_LAUNCH_BLOCKING']}")


def get_optimal_device() -> str:
    """
    Get the optimal PyTorch device string for the current system.
    Automatically detects and returns "cuda", "mps", or "cpu".

    Returns:
        Device string compatible with torch.device()
    """
    device_type, device_info = detect_device()

    if device_info:
        print(f"Detected compute device: {device_info}")

    # For ROCm, use "cuda" as PyTorch uses the same API
    if device_type == "rocm":
        return "cuda"

    return device_type


def is_mps_available() -> bool:
    """
    Check if Apple MPS is available on this system.

    Returns:
        True if MPS is available, False otherwise
    """
    try:
        import torch
        return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    except ImportError:
        return False


def configure_device_for_inference() -> str:
    """
    Configure and return the best device for inference/training.
    Handles all device-specific configurations automatically.

    Returns:
        Device string ("cuda", "mps", or "cpu")
    """
    # Configure AMD if present (must be before torch import)
    configure_amd_gpu()

    # Get optimal device
    device = get_optimal_device()

    # Device-specific optimizations
    if device == "mps":
        print("Using Apple MPS acceleration")
        print("  Note: Some operations may fall back to CPU if not supported by MPS")
    elif device == "cuda":
        try:
            import torch
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
        except:
            pass
    else:
        print("Using CPU (no GPU acceleration)")
        print("  Consider using a GPU for faster processing")

    return device
