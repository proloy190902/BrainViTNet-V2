"""
BrainViTNet-V2 — Training Script
"""
import math
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import (BrainViTNetV2, UncertaintyAwareLoss,
                   CrossDomainContrastiveLoss, count_parameters)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Paths (edit as needed) ──────────────────────────────────────
TRAIN_DIR = "/kaggle/input/brain-tumor-dataset/Brain Tumor Data(1)/Brain Tumor Data/Brain Tumor data/Brain Tumor data/Training"
TEST_DIR  = "/kaggle/input/brain-tumor-dataset/Brain Tumor Data(1)/Brain Tumor Data/Brain Tumor data/Brain Tumor data/Testing"

IMG_SIZE    = 224
BATCH_SIZE  = 16
NUM_EPOCHS  = 50
WARMUP      = 5
PATIENCE    = 10
LR          = 1e-4
LAMBDA_CLS  = 1.0
LAMBDA_CONT = 0.1

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
])
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
test_dataset  = datasets.ImageFolder(TEST_DIR,  transform=test_transform)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                           shuffle=True,  num_workers=2, pin_memory=True)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=2, pin_memory=True)

class_names = train_dataset.classes
num_classes = len(class_names)
print(f"Train: {len(train_dataset)}  |  Test: {len(test_dataset)}  |  Classes: {class_names}")

model          = BrainViTNetV2(num_classes=num_classes, num_domains=1).to(device)
criterion_unc  = UncertaintyAwareLoss(gamma=2.0)
criterion_cont = CrossDomainContrastiveLoss(temperature=0.07)
optimizer      = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)


def warmup_cosine_lr(epoch):
    if epoch < WARMUP:
        return (epoch + 1) / WARMUP
    progress = (epoch - WARMUP) / max(1, NUM_EPOCHS - WARMUP)
    return 0.5 * (1 + math.cos(math.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_lr)

total_params, trainable_params = count_parameters(model)
print(f"Params: {total_params/1e6:.2f}M  |  Trainable: {trainable_params/1e6:.2f}M")

best_val_acc = 0
counter = 0
train_losses, val_losses, train_accs, val_accs, cont_losses_log = [], [], [], [], []

print("\nSTARTING TRAINING\n" + "="*60)

for epoch in range(NUM_EPOCHS):
    model.train()
    run_loss = run_cont = correct = total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        logits, log_var, proj = model(images, return_features=True)
        loss_cls  = criterion_unc(logits, log_var, labels)
        loss_cont = criterion_cont(proj, labels)
        loss = LAMBDA_CLS * loss_cls + LAMBDA_CONT * loss_cont

        if not torch.isfinite(loss):
            print(f"  [warn] NaN/Inf loss at epoch {epoch+1}, skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        run_loss  += loss.item()
        run_cont  += loss_cont.item()
        _, pred    = logits.max(1)
        correct   += pred.eq(labels).sum().item()
        total     += labels.size(0)

    scheduler.step()
    avg_train_loss = run_loss / max(len(train_loader), 1)
    train_acc      = 100. * correct / max(total, 1)
    train_losses.append(avg_train_loss)
    train_accs.append(train_acc)
    cont_losses_log.append(run_cont / max(len(train_loader), 1))

    model.eval()
    val_loss = v_correct = v_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits, log_var = model(images)
            val_loss  += criterion_unc(logits, log_var, labels).item()
            _, pred    = logits.max(1)
            v_correct += pred.eq(labels).sum().item()
            v_total   += labels.size(0)

    avg_val_loss = val_loss / len(test_loader)
    val_acc      = 100. * v_correct / v_total
    val_losses.append(avg_val_loss)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
          f"Train {avg_train_loss:.4f}/{train_acc:.2f}% | "
          f"Val {avg_val_loss:.4f}/{val_acc:.2f}% | "
          f"LR {optimizer.param_groups[0]['lr']:.2e}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_brainvitnet_v2.pth")
        counter = 0
        print(f"  ✓ Best model saved! ({best_val_acc:.2f}%)")
    else:
        counter += 1
        if counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

print(f"\nBest Val Accuracy: {best_val_acc:.2f}%")
