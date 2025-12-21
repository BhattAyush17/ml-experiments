import torch
import os

def save_checkpoint(path, epoch, model, optimizer):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }, path)

def load_checkpoint(path, model, optimizer, device):
    if not os.path.exists(path):
        return 0
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"] + 1
