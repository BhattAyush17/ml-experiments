import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from datasets import load_dataset
from torch.utils.data import DataLoader
from huggingface_hub import HfApi


# -----------------------
# ENV CONFIG (from shell)
# -----------------------
HF_REPO = os.getenv("HF_REPO")
DATASET_NAME = os.getenv("DATASET_NAME", "cifar10")
DATASET_SPLIT = os.getenv("DATASET_SPLIT", "train")  # ✅ FIX
EPOCHS = int(os.getenv("EPOCHS", 1))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))
LR = float(os.getenv("LR", 0.01))

assert HF_REPO is not None, "HF_REPO env var not set"


# -----------------------
# DEVICE
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")


# -----------------------
# DATASET (HF STREAM)
# -----------------------
print("[HF] Requesting dataset from Hugging Face hub...")
dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
print(f"[HF] Dataset loaded: {DATASET_NAME} ({len(dataset)} samples)")
print("[HF] Streaming samples into DataLoader...")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def preprocess(example):
    example["pixel_values"] = transform(example["image"])
    return example

dataset = dataset.with_transform(preprocess)

def collate_fn(batch):
    x = torch.stack([item["pixel_values"] for item in batch])
    y = torch.tensor([item["label"] for item in batch])
    return x, y

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

for x, y in loader:
    print(f"[HF] First batch received: {x.shape}")
    break


# -----------------------
# MODEL (VGG)
# -----------------------
print("[INFO] Initializing VGG16...")

model = models.vgg16(weights=None)
model.classifier[6] = nn.Linear(4096, 10)  # CIFAR-10
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)


# -----------------------
# TRAIN LOOP
# -----------------------
print("[INFO] Starting training...")

model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % 100 == 0:
            print(f"[Epoch {epoch+1}] Step {step} | Loss: {loss.item():.4f}")

    print(f"[Epoch {epoch+1}] Avg Loss: {total_loss / len(loader):.4f}")


# -----------------------
# SAVE CHECKPOINT
# -----------------------
os.makedirs("checkpoints", exist_ok=True)
ckpt_path = "checkpoints/latest.pth"

torch.save(model.state_dict(), ckpt_path)
print(f"[INFO] Checkpoint saved: {ckpt_path}")


# -----------------------
# PUSH TO HUGGINGFACE
# -----------------------
print("[INFO] Uploading checkpoint to HuggingFace...")

api = HfApi()
api.create_repo(
    repo_id=HF_REPO,
    repo_type="model",
    exist_ok=True
)

api.upload_file(
    path_or_fileobj=ckpt_path,
    path_in_repo="latest.pth",
    repo_id=HF_REPO,
    repo_type="model"
)

print("[SUCCESS] Pipeline run complete.")
