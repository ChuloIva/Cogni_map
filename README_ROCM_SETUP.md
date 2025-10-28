# ROCm PyTorch Setup for UV Projects

This guide helps you set up ROCm with PyTorch in UV-based Python projects.

## Quick Setup

### Option 1: Using the setup script

1. **Set environment variables** (in your current shell):
   ```bash
   source setup_rocm_pytorch.sh
   ```

2. **Install PyTorch with ROCm** in your UV environment:
   ```bash
   ./setup_rocm_pytorch.sh install
   ```

3. **Verify installation**:
   ```bash
   ./setup_rocm_pytorch.sh verify
   ```

### Option 2: Manual UV installation

1. **Create/activate UV environment**:
   ```bash
   uv venv
   source .venv/bin/activate  # or use: uv run
   ```

2. **Install PyTorch with ROCm**:
   ```bash
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
   ```

3. **Set environment variables** (add to your shell profile or use direnv):
   ```bash
   source .envrc
   ```

### Option 3: Using direnv (automatic environment loading)

1. **Install direnv** (if not already installed):
   ```bash
   sudo apt install direnv  # or appropriate package manager
   # Add to your ~/.bashrc or ~/.zshrc: eval "$(direnv hook bash)"
   ```

2. **Allow direnv** in this directory:
   ```bash
   direnv allow
   ```

The `.envrc` file will automatically load ROCm environment variables when you enter the directory.

## For Other Projects

Copy these files to your other UV-based projects:

```bash
# Copy to your other project directory
cp setup_rocm_pytorch.sh /path/to/other/project/
cp .envrc /path/to/other/project/
```

Then follow the Quick Setup steps in that project.

## Environment Variables Explained

- `HSA_OVERRIDE_GFX_VERSION=11.0.0` - Tells ROCm to use RDNA 3 architecture (RX 7700 XT)
- `PYTORCH_ROCM_ARCH=gfx1100` - Specifies the exact GPU architecture
- `ROCM_VERSION=6.0` - ROCm version to use
- `HIP_VISIBLE_DEVICES=0` - Makes GPU 0 visible to HIP/ROCm
- `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` - Memory optimization

## Verification

After setup, verify ROCm is working:

```bash
python -c "import torch; print(f'ROCm available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
ROCm available: True
GPU: AMD Radeon RX 7700 XT
```

## Troubleshooting

### ROCm not detected in UV environment

1. Ensure you've installed PyTorch with the ROCm index URL
2. Source the environment variables: `source .envrc`
3. Restart your shell/terminal

### GPU not visible

Check if environment variables are set:
```bash
echo $HSA_OVERRIDE_GFX_VERSION
echo $PYTORCH_ROCM_ARCH
```

If empty, source the `.envrc` file again.

## pyproject.toml Configuration

Add this to your `pyproject.toml` for UV to use the correct PyTorch index:

```toml
[tool.uv]
index-url = "https://download.pytorch.org/whl/rocm6.0"

# Or use extra-index-url to keep PyPI as primary
[[tool.uv.index]]
name = "pytorch-rocm"
url = "https://download.pytorch.org/whl/rocm6.0"
```