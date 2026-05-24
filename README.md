# Semantic Segmentation API

![Python Version](https://img.shields.io/badge/python-3.11-blue)
![Framework](https://img.shields.io/badge/framework-FastAPI-green)
![Deep Learning](https://img.shields.io/badge/backend-PyTorch-orange)

A lightweight production-ready FastAPI application serving Semantic Segmentation models. 
This API supports binary segmentation tasks using both custom Residual U-Net architectures and ResNet34 backbone networks, handling real-time image preprocessing and mask generation.

---

## Features

- **FastAPI Framework:** High performance async image inference endpoints.
- **Dual Architectures:** Supported options include:
  - `unet`: A custom U-Net network featuring residual blocks (`ResBlock`).
  - `resnet34_unet`: A U-Net decoder integrated with a ResNet34 feature extractor.
- **Dynamic Preprocessing:** Images are automatically rescaled to $256 \times 256$ pixels, processed, and evaluated through a pixel threshold ($> 0.5$).
- **Dockerized:** Fully containerized setup via `python:3.11-slim` for hassle-free deployments.

---

## Repository Structure


```text
├── saved_models/          
│   ├── unet/
│   │   └── best_model.pth
│   └── resnet34_unet/
│       └── best_model.pth
├── src/
│   ├── models/
│   │   ├── unet.py
│   │   └── resnet34_unet.py
│   ├── app.py             # FastAPI entrypoint
│   ├── inference.py       # Preprocessing & inference pipeline
│   ├── utils.py           # Model loading routines
├── Dockerfile             # Docker image configuration
├── requirements.txt       # App dependencies
└── .gitignore
