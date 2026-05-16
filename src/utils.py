import torch
from models.unet import Unet
from models.resnet34_unet import ResNet34_UNet
import os


def get_model(model_name, device):
    if model_name == 'unet':
        return Unet(in_channels=3).to(device)
    elif model_name == 'resnet34_unet':
        return ResNet34_UNet().to(device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def load_model(model_name, model_path, device):
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = get_model(model_name, device)    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model
