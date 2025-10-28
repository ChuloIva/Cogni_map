"""
GPU detection and configuration utilities for AMD ROCm
Based on the main project's gpu_utils.py
"""

import os
import subprocess
from typing import Tuple, Optional


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
        print("🎮 AMD GPU detected - configuring ROCm environment variables")
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
        os.environ["HIP_LAUNCH_BLOCKING"] = "0"  # Set to 1 for debugging

        print(f"  ✓ HSA_OVERRIDE_GFX_VERSION: {os.environ['HSA_OVERRIDE_GFX_VERSION']}")
        print(f"  ✓ PYTORCH_ROCM_ARCH: {os.environ['PYTORCH_ROCM_ARCH']}")
        print(f"  ✓ TORCH_USE_HIP_DSA: {os.environ['TORCH_USE_HIP_DSA']}")
    else:
        print("ℹ️  No AMD GPU detected, will use CPU or CUDA if available")


def get_optimal_device() -> str:
    """
    Get the optimal PyTorch device string for the current system.
    Automatically detects and returns "cuda" (for both NVIDIA and AMD ROCm) or "cpu".

    Returns:
        Device string compatible with torch.device()
    """
    import torch

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"✅ Using GPU: {device_name}")
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            print(f"  ROCm version: {torch.version.hip}")
        elif torch.version.cuda:
            print(f"  CUDA version: {torch.version.cuda}")
        return "cuda"
    else:
        print("⚠️  No GPU available - using CPU")
        print("  Consider installing ROCm PyTorch for AMD GPU support")
        return "cpu"
