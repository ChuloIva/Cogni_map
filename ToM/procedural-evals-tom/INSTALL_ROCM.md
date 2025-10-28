# Installing PyTorch with ROCm 6.0 for AMD GPU

Your system has an **AMD Radeon RX 7800 XT** GPU. To use it with PyTorch, you need the ROCm version.

## Current Issue

You currently have PyTorch with CUDA 12.8:
```
torch 2.9.0+cu128 (CUDA available: False)
```

This won't work with AMD GPUs. You need PyTorch with ROCm 6.0.

## Solution: Install PyTorch with ROCm

### Option 1: Using pip (Recommended)

```bash
cd ToM/procedural-evals-tom

# Uninstall current PyTorch
uv pip uninstall torch torchvision torchaudio

# Install PyTorch with ROCm 6.0
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Verify installation
uv run python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA/ROCm available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Option 2: Install all requirements with ROCm

```bash
cd ToM/procedural-evals-tom

# Create a clean requirements file
cat > requirements_rocm.txt << 'EOF'
langchain>=0.1.0
langchain-openai
crfm-helm
scipy
seaborn
matplotlib
pandas
pydantic
python-dotenv
openai>=1.0.0
tqdm
transformers>=4.40.0
accelerate>=0.20.0
# PyTorch with ROCm 6.0 will be installed separately
EOF

# Install requirements
uv pip install -r requirements_rocm.txt

# Install PyTorch with ROCm 6.0
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

## Verify Installation

After installation, verify that PyTorch can see your AMD GPU:

```bash
uv run python -c "
import torch
print('='*60)
print('PyTorch Version:', torch.__version__)
print('CUDA/ROCm Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device Name:', torch.cuda.get_device_name(0))
    print('Device Count:', torch.cuda.device_count())
    if hasattr(torch.version, 'hip') and torch.version.hip:
        print('ROCm Version:', torch.version.hip)
print('='*60)
"
```

Expected output:
```
============================================================
PyTorch Version: 2.x.x+rocm6.0
CUDA/ROCm Available: True
Device Name: AMD Radeon RX 7800 XT
Device Count: 1
ROCm Version: 6.0.xxxxx
============================================================
```

## What the GPU Configuration Does

The `gpu_utils.py` file automatically configures your AMD GPU when you run the evaluation:

- **HSA_OVERRIDE_GFX_VERSION=11.0.0**: Your RX 7800 XT is gfx1101, but PyTorch compiles for gfx1100
- **PYTORCH_ROCM_ARCH=gfx1100**: Use gfx1100 kernels (compatible with your GPU)
- **TORCH_USE_HIP_DSA=1**: Enable HIP Direct Shared Access
- **PYTORCH_HIP_ALLOC_CONF=expandable_segments:True**: Better memory management

## Troubleshooting

### If GPU is still not detected:

1. **Check if ROCm drivers are installed:**
   ```bash
   rocm-smi
   ```
   If this fails, you may need to install ROCm drivers from AMD.

2. **Check lspci output:**
   ```bash
   lspci | grep -i vga
   ```
   Should show: `Advanced Micro Devices, Inc. [AMD/ATI] Navi 32 [Radeon RX 7700 XT / 7800 XT]`

3. **Check environment variables:**
   ```bash
   uv run python -c "
   import os
   print('HSA_OVERRIDE_GFX_VERSION:', os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'Not set'))
   print('PYTORCH_ROCM_ARCH:', os.environ.get('PYTORCH_ROCM_ARCH', 'Not set'))
   "
   ```

### If you need to install ROCm drivers:

Visit: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/

For Ubuntu/Debian:
```bash
wget https://repo.radeon.com/amdgpu-install/6.0/ubuntu/jammy/amdgpu-install_6.0.60000-1_all.deb
sudo apt install ./amdgpu-install_6.0.60000-1_all.deb
sudo amdgpu-install --usecase=rocm
```

## Running Evaluation After Installation

Once PyTorch with ROCm is installed correctly:

```bash
cd ToM/procedural-evals-tom/code/src
uv run python evaluate_gemma.py --num_samples 10  # Test with 10 samples first
```

You should see:
```
🎮 AMD GPU detected - configuring ROCm environment variables
  ✓ HSA_OVERRIDE_GFX_VERSION: 11.0.0
  ✓ PYTORCH_ROCM_ARCH: gfx1100
  ✓ TORCH_USE_HIP_DSA: 1

============================================================
Loading model: google/gemma-3-4b-it
Target device: cuda
============================================================
✅ Using GPU: AMD Radeon RX 7800 XT
  ROCm version: 6.0.xxxxx
```

## Performance Tips

With ROCm properly configured, you should see:
- **Loading speed**: Model loads in ~5-10 seconds (vs minutes on CPU)
- **Inference speed**: ~10-30 tokens/second (vs <1 token/second on CPU)
- **Memory usage**: ~8GB VRAM for Gemma 3 4B IT

## Summary

1. Uninstall current PyTorch (CUDA version)
2. Install PyTorch with ROCm 6.0
3. Run evaluation - AMD GPU will be auto-detected and configured
4. Enjoy fast GPU-accelerated inference!