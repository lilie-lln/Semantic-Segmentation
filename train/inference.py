import argparse
import torch
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
from utils import get_model
import torchvision.transforms as T


def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")

    image = T.Resize(
        (256, 256),
        interpolation=T.InterpolationMode.BILINEAR
    )(image)

    image_tensor = T.ToTensor()(image)

    return image_tensor

def inference(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        output = model(image_tensor)
        output = torch.sigmoid(output)     
        output = (output > 0.5).float()
    return output

def main():
    parser = argparse.ArgumentParser(description="Image segmentation inference code")
    parser.add_argument('--model', type=str, default='resnet34_unet', help="Choose the model type")
    parser.add_argument('--model_path', type=str, help="Path to the model weights file")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the image(s) for inference (single file or folder)")
    parser.add_argument('--output_dir', type=str, default='inference_outputs', help="Directory to save inference results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(args.model, device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    image_files = []
    if os.path.isdir(args.image_path):
        for file in os.listdir(args.image_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(os.path.join(args.image_path, file))
    else:
        image_files.append(args.image_path)

    os.makedirs(args.output_dir, exist_ok=True)

    for img_path in image_files:
        image_tensor = load_and_preprocess_image(img_path)
        output = inference(model, image_tensor, device)
        pred = output.squeeze().cpu().numpy()
        pred_img = (pred * 255).astype(np.uint8)
        pred_pil = Image.fromarray(pred_img)
        base_name = os.path.basename(img_path)
        save_path = os.path.join(args.output_dir, f"pred_{base_name}")
        pred_pil.save(save_path)

if __name__ == "__main__":
    main()
