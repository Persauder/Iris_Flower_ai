import argparse
import os
from pathlib import Path
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_dataloaders(data_dir: Path, batch_size=16):
    train_t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])
    val_t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_t)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=val_t)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_dl, val_dl, train_ds.classes

def build_model(num_classes: int):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model

def train_one_epoch(model, dl, criterion, optimizer):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n

@torch.no_grad()
def evaluate(model, dl, criterion):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="dataset", help="Root folder with class subfolders or train/val split")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    data_root = Path(args.data)
    # If user put images directly in dataset/<class>/*, create a simple 80/20 split into train/val
    if (data_root / "train").exists() and (data_root / "val").exists():
        train_dl, val_dl, classes = get_dataloaders(data_root, batch_size=args.batch_size)
    else:
        # Auto split
        full_ds = datasets.ImageFolder(data_root, transform=transforms.ToTensor())
        classes = full_ds.classes
        # Simple split: copy indices
        n = len(full_ds)
        idx = torch.randperm(n)
        split = int(0.8 * n)
        train_idx, val_idx = idx[:split], idx[split:]
        # Subset loaders with transforms
        train_t = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ])
        val_t = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        # Manually map indices
        from torch.utils.data import Subset
        base_train = datasets.ImageFolder(data_root, transform=train_t)
        base_val = datasets.ImageFolder(data_root, transform=val_t)
        train_dl = DataLoader(Subset(base_train, train_idx.tolist()), batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_dl = DataLoader(Subset(base_val, val_idx.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(num_classes=len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True, parents=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_dl, criterion)
        print(f"Epoch {epoch:02d}: train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {val_loss:.4f} acc {val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "classes": classes,
            }, models_dir / "iris_cnn.pth")
            print(f"  ✔ Saved best model (acc={best_acc:.3f}) to models/iris_cnn.pth")

    print("Done. Best val acc:", best_acc)

if __name__ == "__main__":
    main()