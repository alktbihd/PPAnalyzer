# GPU Deployment Guide for PPAnalyzer

This guide shows how to deploy PPAnalyzer with GPU support for **10-20x faster classification** (2 minutes → 5-10 seconds).

## Quick Start

### 1. Deploy GPU Container

```bash
./deploy-gpu-azure.sh
```

This will:
- ✅ Build GPU-enabled Docker image
- ✅ Push to Azure Container Registry  
- ✅ Create Azure Container Instance with K80 GPU
- ✅ Configure environment variables
- ⏱️ Takes ~15-20 minutes

### 2. Manage GPU Container

```bash
# Make control script executable
chmod +x gpu-control.sh

# Start container (begin charging)
./gpu-control.sh start

# Stop container (stop charging)
./gpu-control.sh stop

# View logs
./gpu-control.sh logs

# Check status
./gpu-control.sh status

# Get URL
./gpu-control.sh url
```

## Performance Comparison

| Setup | Classification Time | Concurrent Users | Monthly Cost |
|-------|---------------------|------------------|--------------|
| **CPU (B2)** | ~2 minutes | Limited | $55 |
| **GPU (K80)** | **5-10 seconds** ⚡ | Unlimited | $684 always-on |
| **GPU (on-demand)** | **5-10 seconds** ⚡ | Unlimited | **~$5-20/semester** 💰 |

## GPU Specifications

- **GPU**: NVIDIA K80 (12 GB VRAM)
- **CPU**: 6 cores
- **RAM**: 16 GB
- **Location**: East US
- **Cost**: ~$0.95/hour when running

## Update Frontend

After deploying GPU container, update your frontend to use the new URL:

1. Get the GPU URL:
   ```bash
   ./gpu-control.sh url
   ```

2. Update `.github/workflows/frontend-deploy.yml`:
   ```yaml
   env:
     VITE_API_URL: http://ppanalyzer-gpu.eastus.azurecontainer.io:8000
   ```

3. Or for local testing:
   ```bash
   cd frontend
   VITE_API_URL=http://ppanalyzer-gpu.eastus.azurecontainer.io:8000 npm run dev
   ```

## Cost Management

### Option 1: On-Demand (Recommended for Demos)
```bash
# Before presentation/demo
./gpu-control.sh start

# After presentation/demo
./gpu-control.sh stop
```

**Cost**: ~$5-20 for entire semester

### Option 2: Always On (Production)
Keep container running 24/7.

**Cost**: ~$684/month

### Option 3: Spot Instances (70-90% cheaper)
Modify `deploy-gpu-azure.sh` line 88 to add:
```bash
--priority Spot
```

**Cost**: ~$0.10-0.30/hour

## Verification

After deployment, check that GPU is being used:

```bash
# View logs
./gpu-control.sh logs

# Look for:
✓ Using device: cuda
✓ PrivBERT model loaded successfully on cuda!
```

## Troubleshooting

### Container not starting
```bash
# Check logs
./gpu-control.sh logs

# Restart
./gpu-control.sh restart
```

### Model not found
The model is included in the Docker image via `COPY . .` in `Dockerfile.gpu`. Make sure:
- Model exists at `backend/model/privbert_final/`
- `.dockerignore` doesn't exclude model files

### Out of memory
K80 has 12 GB VRAM. If you get OOM errors:
- Reduce batch size in `classifier.py` (line 130)
- Use smaller model chunks

## Switching Back to CPU

To switch back to the CPU App Service:

1. Stop GPU container:
   ```bash
   ./gpu-control.sh stop
   ```

2. Update frontend URL back to:
   ```
   https://ppanalyzer-backend.azurewebsites.net
   ```

Both can run simultaneously for A/B testing!

## Advanced: Multiple GPU Types

### V100 (10x faster than K80)
Modify `deploy-gpu-azure.sh` line 81:
```bash
--gpu-resource count=1 sku=V100
```

**Cost**: ~$3.06/hour

### Multiple GPUs
```bash
--gpu-resource count=2 sku=K80
```

## Support

For issues or questions, check:
- Container logs: `./gpu-control.sh logs`
- Azure Portal: Container Instances → ppanalyzer-gpu
- Health check: `http://[GPU-URL]:8000/api/health`

