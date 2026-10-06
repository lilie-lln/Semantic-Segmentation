
# Semantic Segmentation API

A FastAPI service for binary semantic segmentation inference.  
It supports two U-Net-based architectures and includes a Gradio web interface.

Model weights are hosted on Hugging Face:  
[`Jolie11/semantic-segmentation-models`](https://huggingface.co/Jolie11/semantic-segmentation-models)

## Features

- Two available models:
  - `unet`: Standard U-Net with DoubleConv blocks and skip connections
  - `resnet34_unet`: ResNet34 encoder + U-Net style decoder
- Input images are resized to 256×256 for inference, then restored to the original size
- Binary mask generated using sigmoid + 0.5 threshold
- REST API endpoint (`/predict`) and Gradio demo (`/demo`)
- Supports both CPU and CUDA

## Project Structure

```text
├── src/
│   ├── models/
│   │   ├── unet.py              # Standard U-Net
│   │   └── resnet34_unet.py     # ResNet34-UNet
│   ├── app.py                   # FastAPI + Gradio entry point
│   ├── inference.py             # Preprocessing and inference
│   └── utils.py                 # Model loading utilities
├── Dockerfile
├── requirements.txt
└── .gitignore
```

### Docker

```bash
docker build -t semantic-segmentation .
docker run --gpus all -p 8000:8000 semantic-segmentation
```

> The Dockerfile uses `pytorch/pytorch:2.14.0-cuda13.2-cudnn9-runtime` as the base image.

###
<img width="528" height="740" alt="ez" src="https://github.com/user-attachments/assets/02bb7dad-ac69-49bf-b7b0-0ad4f16b32f2" />



### GET `/`

Returns service status and the list of available models.

## Model Details

- **unet**: Classic U-Net architecture using DoubleConv (Conv → BN → ReLU) blocks and skip connections.
- **resnet34_unet**: Custom ResNet34 encoder combined with a U-Net style decoder.

Both models output a single channel (binary segmentation). During inference, `torch.sigmoid` is applied followed by a 0.5 threshold.

## Dependencies

Main packages (see `requirements.txt` for the full list):

- FastAPI / Uvicorn
- PyTorch (provided by the Docker base image)
- Pillow / NumPy
- Gradio
- huggingface_hub
