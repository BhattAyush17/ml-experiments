import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from datasets import load_dataset
from torch.utils.data import DataLoader

from common.checkpoints import CheckpointManager



# ============================================================
# CONFIG (ENV ONLY)
# ============================================================

def load_config():
    return {
        "epochs": int(os.getenv("EPOCHS", 1)),
        "lr": float(os.getenv("LR", 0.01)),
        "output_dir": os.getenv("OUTPUT_DIR", "./checkpoints"),
    }



# ============================================================
# DATASET (ABSOLUTE CANARY – UNBREAKABLE)
# ============================================================

def build_canary_loader():
    print("[HF] Loading CIFAR-10 dataset...")

    dataset = load_dataset("cifar10", split="train")
    print(f"[HF] Dataset size: {len(dataset)}")

    to_tensor = transforms.ToTensor()

    def unwrap(x):
        while isinstance(x, list):
            x = x[0]
        return x

    def collate_fn(batch):
        item = batch[0]                # single sample
        img = unwrap(item["img"])

        if not torch.is_tensor(img):
            img = to_tensor(img)

        x = img.unsqueeze(0)           # [1, C, H, W]
        y = torch.tensor([item["label"]])

        return x, y

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
    )

    return loader


# ============================================================
# MODEL
# ============================================================

def build_model(device):
    model = models.vgg16(weights=None)
    model.classifier[6] = nn.Linear(4096, 10)
    return model.to(device)


# ============================================================
# TRAIN (CANARY — ONE STEP ONLY)
# ============================================================

def train_canary(model, loader, cfg, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9)

    model.train()

    for epoch in range(cfg["epochs"]):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

            print(f"[TRAIN] Canary loss: {loss.item():.4f}")
            return optimizer   # 🔒 exit immediately



# ============================================================
# ENTRY POINT
# ============================================================

def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    loader = build_canary_loader()
    model = build_model(device)

    optimizer = train_canary(model, loader, cfg, device)

    ckpt = CheckpointManager(cfg["output_dir"])
    ckpt.save(model, optimizer, step=0)

    print("[SUCCESS] PIPELINE CANARY PASSED.")



if __name__ == "__main__":
    main()
