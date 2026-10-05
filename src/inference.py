import torch
from PIL import Image
import numpy as np
import torchvision.transforms as T


def load_and_preprocess_image(image):
    ##image = Image.open(image).convert("RGB")
    # image = image.convert("RGB")
    # image_resized = image.resize((256, 256), Image.BILINEAR)
    # image_np = np.array(image_resized)
    # image_np = np.moveaxis(image_np, -1, 0)
    # image_tensor = torch.from_numpy(image_np).float()
    
    original_size = image.size
    image = T.Resize((256, 256), interpolation=T.InterpolationMode.BILINEAR
    )(image)

    image_tensor = T.ToTensor()(image)

    return image_tensor,original_size

def predict_mask(model, image, device):
    image_tensor,original_size = load_and_preprocess_image(image)

    with torch.inference_mode():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        output = model(image_tensor)
        output = torch.sigmoid(output)     
        output = (output > 0.5).float()
        pred = output.squeeze().cpu().numpy()

        pred_img = (pred * 255).astype(np.uint8)
        pred_pil = Image.fromarray(pred_img)
        pred_pil = pred_pil.resize(original_size, resample=Image.Resampling.NEAREST)

    return pred_pil



   
