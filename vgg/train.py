import torch, os
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

from common.seed import set_seed
from common.checkpoint import save_checkpoint, load_checkpoint
from common.hf_push import push_to_hf

# ---------------- CONFIG ----------------
set_seed(42)

HF_REPO = "username/vgg-cifar10"
CKPT_PATH = "latest.pth"
EPOCHS = 10
BATCH_SIZE = 64
LR = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- DATA ----------------
dataset = load_dataset("cifar10")

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])

def tf(batch):
    batch["pixel_values"] = [transform(img) for img in batch["img"]]
    return batch

dataset = dataset.with_transform(tf)

loader = DataLoader(
    dataset["train"],
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# ---------------- MODEL ----------------
model = models.vgg11(num_classes=10).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

start_epoch = load_checkpoint(CKPT_PATH, model, optimizer, DEVICE)

# ---------------- TRAIN ----------------
for epoch in range(start_epoch, EPOCHS):
    model.train()
    pbar = tqdm(loader, desc=f"Epoch {epoch}")

    for batch in pbar:
        x = batch["pixel_values"].to(DEVICE)
        y = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=loss.item())

    save_checkpoint(CKPT_PATH, epoch, model, optimizer)
    push_to_hf(CKPT_PATH, HF_REPO)

print("VGG training finished.")
