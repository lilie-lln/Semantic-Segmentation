import torch
from models.unet import Unet
from models.resnet34_unet import ResNet34_UNet
import os

def dice_score(pred, target, valid_mask, threshold=0.5):

    pred_probs = torch.sigmoid(pred)
    pred_binary = (pred_probs > threshold).float()
    pred_binary = pred_binary * valid_mask
    target *= valid_mask

    dims = tuple(range(1, pred_binary.ndim))

    intersection = torch.sum(
        pred_binary * target,
        dim=dims
    )

    denominator = (
        torch.sum(pred_binary, dim=dims)
        + torch.sum(target, dim=dims)
    )
    
    smooth = 1e-6
    
    # intersection = torch.sum(pred_binary * target)
    # sum_pred_target = torch.sum(pred_binary) + torch.sum(target)

    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    # 預防 empty mask 
    dice = torch.where(denominator < smooth, torch.ones_like(dice), dice)
    # if sum_pred_target < smooth:
    #     return 1.0
    # dice = (2. * intersection + smooth) / (sum_pred_target + smooth)
    return dice

def get_model(model_name, device):
    if model_name == 'unet':
        return Unet(in_channels=3).to(device)
    elif model_name == 'resnet34_unet':
        return ResNet34_UNet().to(device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def load_model(model_path, model_name, device):
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = get_model(model_name, device)    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model
