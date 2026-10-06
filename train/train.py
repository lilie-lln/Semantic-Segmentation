import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import os
from tqdm import tqdm
from oxford_pet import load_dataset
from utils import get_model
from evaluate import evaluate
import random
import numpy as np

def train(model, train_loader, val_loader, optimizer, criterion, num_epochs=10, device=None, save_dir='saved_models', model_type='unet'):
    model_save_dir = os.path.join(save_dir, model_type)
    os.makedirs(model_save_dir, exist_ok=True)
    
    best_dice = 0.0
    best_epoch = 0
    loss_log = [0.000] * num_epochs
    dice_log = [0.000] * num_epochs

    for epoch in range(num_epochs):
        model.train()

        total_loss = 0.0
        total_valid_pixels = 0

        for sample in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = sample['image'].to(device)
            masks = sample['mask'].to(device)
            valid_masks = sample["valid_mask"].to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss_map = criterion(outputs, masks)

            loss_sum = (loss_map * valid_masks).sum()
            valid_pixels = valid_masks.sum()

            loss = loss_sum / valid_pixels.clamp_min(1.0)
            # loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss_sum.item()
            total_valid_pixels += valid_pixels.item()

        avg_train_loss = (total_loss / total_valid_pixels)
        loss_log[epoch] = avg_train_loss
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_train_loss:.4f}")

        if val_loader is not None:
            val_loss, val_dice = evaluate(model, val_loader, criterion, device)
            print(f"Validation Loss: {val_loss:.4f}, Dice Score: {val_dice:.4f}")
            dice_log[epoch] = val_dice
            if val_dice > best_dice:
                best_dice = val_dice
                best_epoch = epoch + 1
                best_model_path = os.path.join(model_save_dir, "best_model.pth")
                torch.save(model.state_dict(), best_model_path)

        # save_path = os.path.join(model_save_dir, f"epoch_{epoch+1}.pth")
        # torch.save(model.state_dict(), save_path)

        # 其實可以不用改 就只是全部生成之後存起來 改成 蓋掉上一次的

        last_checkpoint_path = os.path.join(
            model_save_dir,
            "last_checkpoint.pth"
        )

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_dice": best_dice,
        }, last_checkpoint_path)

    for epoch in range(num_epochs):
        print(f" {dice_log[epoch]} \t  {loss_log[epoch]}")
    
    return best_dice, best_epoch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--model', type=str, default='resnet34_unet', help='Which model to train')
    parser.add_argument('--data_path', type=str, default='../dataset/oxford-iiit-pet')
    parser.add_argument('--epochs', '-e', type=int, default=30)
    parser.add_argument('--batch_size', '-b', type=int, default=8, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--save_dir', type=str, default='saved_models/resnet34_unet_', help='directory to save the models')
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = load_dataset(args.data_path, mode="train")
    val_dataset = load_dataset(args.data_path, mode="valid")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = get_model(args.model, device)

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    best_dice, best_epoch = train(model, train_loader, val_loader, optimizer, criterion, 
        num_epochs=args.epochs, device=device, save_dir=args.save_dir, model_type=args.model)
    
    model_save_dir = os.path.join(args.save_dir, args.model)
    final_model_path = os.path.join(model_save_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)

