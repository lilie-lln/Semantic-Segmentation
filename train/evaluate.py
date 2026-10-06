import torch
from tqdm import tqdm 
from utils import dice_score

def evaluate(model, val_loader, criterion, device):

    model.eval()
    total_loss = 0.0
    total_valid_pixels = 0
    dice_scores = []

    with torch.no_grad():
        for sample in tqdm(val_loader):
            images = sample['image'].float().to(device)
            masks = sample['mask'].float().to(device)
            valid_masks = sample["valid_mask"].to(device)
            outputs = model(images)

            loss_map = criterion(outputs, masks)
            # loss = (loss_map * valid_masks).sum() / valid_masks.sum().clamp_min(1.0)
            # 扣掉boundary剩下的才有效
            loss_sum = (loss_map * valid_masks).sum()
            valid_pixels = valid_masks.sum()

            total_loss += loss_sum.item() 
            total_valid_pixels += valid_pixels.item()
            
            dice = dice_score(outputs, masks, valid_masks)
            dice_scores.extend(dice.cpu().tolist())
            
    avg_dice = sum(dice_scores) / len(dice_scores)
    avg_loss = total_loss / total_valid_pixels
    return avg_loss, avg_dice