"""
BrainViTNet-V2 — Evaluation + Visualization Script
Run after training: python evaluate.py
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_score, recall_score, f1_score)
from sklearn.preprocessing import label_binarize
from scipy import stats
from model import (BrainViTNetV2, TemperatureScaling,
                   GradCAMPlusPlus, apply_colormap_on_image)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE   = 224
BATCH_SIZE = 16

TEST_DIR = "/kaggle/input/brain-tumor-dataset/Brain Tumor Data(1)/Brain Tumor Data/Brain Tumor data/Brain Tumor data/Testing"

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)
class_names  = test_dataset.classes
num_classes  = len(class_names)

# ── Load model ─────────────────────────────────────────────────
model = BrainViTNetV2(num_classes=num_classes, num_domains=1).to(device)
model.load_state_dict(torch.load("best_brainvitnet_v2.pth", map_location=device))
model.eval()

# ── Temperature calibration ────────────────────────────────────
temp_scaler    = TemperatureScaling().to(device)
temp_optimizer = torch.optim.LBFGS([temp_scaler.temperature], lr=0.01, max_iter=50)
all_logits_cal, all_labels_cal = [], []
with torch.no_grad():
    for images, labels in test_loader:
        logits, _ = model(images.to(device))
        all_logits_cal.append(logits)
        all_labels_cal.append(labels.to(device))
all_logits_cal = torch.cat(all_logits_cal)
all_labels_cal = torch.cat(all_labels_cal)


def cal_eval():
    temp_optimizer.zero_grad()
    F.cross_entropy(temp_scaler(all_logits_cal), all_labels_cal).backward()
    return F.cross_entropy(temp_scaler(all_logits_cal), all_labels_cal)


temp_optimizer.step(cal_eval)
print(f"Temperature: {temp_scaler.temperature.item():.4f}")

# ── Inference ──────────────────────────────────────────────────
y_true, y_pred, y_scores = [], [], []
all_epistemic, all_aleatoric = [], []

for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    mean_pred, ep_unc, al_unc = model.predict_with_uncertainty(images, T=10)
    with torch.no_grad():
        logits, _ = model(images)
        cal_probs = torch.softmax(temp_scaler(logits), dim=1)
    _, pred = cal_probs.max(1)
    y_true.extend(labels.cpu().numpy())
    y_pred.extend(pred.cpu().numpy())
    y_scores.append(cal_probs.cpu().numpy())
    all_epistemic.extend(ep_unc.cpu().numpy())
    all_aleatoric.extend(al_unc.cpu().numpy())

y_scores      = np.vstack(y_scores)
all_epistemic = np.array(all_epistemic)
all_aleatoric = np.array(all_aleatoric)
model.eval()

# ── Plot 1: Confusion Matrix ────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names,
            yticklabels=class_names, cmap="Blues", annot_kws={"size": 14})
plt.xlabel("Predicted", fontsize=14, fontweight='bold')
plt.ylabel("True",      fontsize=14, fontweight='bold')
plt.title("Confusion Matrix - BrainViTNet-V2", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("../results/confusion_matrix_v2_300dpi.png", dpi=300, bbox_inches='tight')
plt.show()

# ── Plot 2: ROC Curves ─────────────────────────────────────────
y_true_bin = label_binarize(y_true, classes=range(num_classes))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
plt.figure(figsize=(10, 8))
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
    plt.plot(fpr, tpr, color=colors[i], lw=2.5,
             label=f'{class_names[i]} (AUC={auc(fpr, tpr):.3f})')
plt.plot([0,1],[0,1],'k--', lw=2, label='Random')
plt.xlabel('FPR', fontsize=14, fontweight='bold')
plt.ylabel('TPR', fontsize=14, fontweight='bold')
plt.title('ROC Curves - BrainViTNet-V2', fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../results/roc_curve_v2_300dpi.png", dpi=300, bbox_inches='tight')
plt.show()

# ── Plot 3: Uncertainty Distribution ───────────────────────────
correct_mask = np.array(y_true) == np.array(y_pred)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, unc, title in zip(
        axes[:2], [all_epistemic, all_aleatoric],
        ['Epistemic Uncertainty', 'Aleatoric Uncertainty']):
    ax.hist(unc[correct_mask],  bins=50, alpha=0.7, color='#4ECDC4',
            label='Correct',   density=True)
    ax.hist(unc[~correct_mask], bins=50, alpha=0.7, color='#FF6B6B',
            label='Incorrect', density=True)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(title, fontsize=13, fontweight='bold')
    ax.set_ylabel('Density', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(alpha=0.3)

cls_ep = [np.mean(all_epistemic[np.array(y_true)==i]) for i in range(num_classes)]
cls_al = [np.mean(all_aleatoric[np.array(y_true)==i]) for i in range(num_classes)]
x_pos, w = np.arange(num_classes), 0.35
axes[2].bar(x_pos-w/2, cls_ep, w, label='Epistemic', color='#45B7D1', alpha=0.8)
axes[2].bar(x_pos+w/2, cls_al, w, label='Aleatoric', color='#FFA07A', alpha=0.8)
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(class_names, rotation=30, ha='right')
axes[2].set_title('Per-Class Uncertainty', fontsize=14, fontweight='bold')
axes[2].legend(fontsize=11); axes[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../results/uncertainty_analysis_v2_300dpi.png", dpi=300, bbox_inches='tight')
plt.show()

# ── Statistics ─────────────────────────────────────────────────
print("\n" + "="*60 + "\nSTATISTICAL ANALYSIS\n" + "="*60)
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

precision = precision_score(y_true, y_pred, average=None)
recall    = recall_score(y_true, y_pred, average=None)
f1        = f1_score(y_true, y_pred, average=None)

overall_acc = 100 * np.mean(np.array(y_true) == np.array(y_pred))
macro_f1    = f1_score(y_true, y_pred, average='macro')
weighted_f1 = f1_score(y_true, y_pred, average='weighted')
n   = len(y_true)
se  = np.sqrt((overall_acc/100 * (1-overall_acc/100)) / n)
ci  = stats.norm.interval(0.95, loc=overall_acc/100, scale=se)

print(f"Overall Accuracy : {overall_acc:.2f}%")
print(f"Macro F1         : {macro_f1:.4f}")
print(f"Weighted F1      : {weighted_f1:.4f}")
print(f"Mean Epistemic   : {np.mean(all_epistemic):.6f}")
print(f"Mean Aleatoric   : {np.mean(all_aleatoric):.4f}")
print(f"Temperature      : {temp_scaler.temperature.item():.4f}")
print(f"95% CI           : [{ci[0]*100:.2f}%, {ci[1]*100:.2f}%]")
