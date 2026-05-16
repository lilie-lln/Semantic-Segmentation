import torch

from PIL import Image
import numpy as np
from utils import load_model

def load_and_preprocess_image(image):
    #image = Image.open(image).convert("RGB")
    image = image.convert("RGB")
    image_resized = image.resize((256, 256), Image.BILINEAR)
    image_np = np.array(image_resized)
    image_np = np.moveaxis(image_np, -1, 0)
    image_tensor = torch.from_numpy(image_np).float()

    return image_tensor

def predict_mask(model, image, device):
    image_tensor = load_and_preprocess_image(image)

    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        output = model(image_tensor)
        output = torch.sigmoid(output)     
        output = (output > 0.5).float()
        pred = output.squeeze().cpu().numpy()

        pred_img = (pred * 255).astype(np.uint8)

    return Image.fromarray(pred_img)



   
