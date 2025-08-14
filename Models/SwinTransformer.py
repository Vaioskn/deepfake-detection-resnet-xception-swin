import os
import timm
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_curve, auc, average_precision_score
)

from utils.constants import MAIN_OUTPUT_FOLDER
from utils.constants import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE

class SE1d(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x)
        return x * w 


class SwinTransformerWrapper(nn.Module):
    def __init__(self,
                 model_name='swin_base_patch4_window7_224',
                 num_classes=2,
                 se_reduction=16):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        C = self.backbone.num_features

        self.se     = SE1d(C, reduction=se_reduction)
        self.head = nn.Sequential(
            nn.Linear(C, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        feats = self.backbone(x)
        attn  = self.se(feats)
        out   = self.head(attn)
        return out


def validate_video_level(model, val_loader, device):
    """Video‐level validation with six aggregation methods."""
    model.eval()
    video_dict = defaultdict(lambda: {"labels": [], "probs": [], "preds": []})
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    total = 0

    with torch.no_grad():
        for images, labels, paths in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            total += images.size(0)

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            for i, p in enumerate(paths):
                video_dict[p]["labels"].append(labels[i].item())
                video_dict[p]["probs"].append(probs[i])
                video_dict[p]["preds"].append(preds[i])

    val_loss = running_loss / total if total else 0.0

    methods = {
        'mean_prob':       lambda x: np.mean([p[1] for p in x]),
        'max_prob':        lambda x: np.max([p[1] for p in x]),
        'min_prob':        lambda x: np.min([p[1] for p in x]),
        'median_prob':     lambda x: np.median([p[1] for p in x]),
        'trimmed_mean':    lambda x: np.mean(sorted([p[1] for p in x])[1:-1]) if len(x)>2 else np.mean([p[1] for p in x]),
        'vote_percentage': lambda x: np.mean(x)
    }

    metrics_all = {}
    best_f1 = -1.0
    best_acc = 0.0
    best_method = None

    for name, agg in methods.items():
        y_true, y_pred, scores = [], [], []
        for info in video_dict.values():
            if not info["labels"]:
                continue
            true = info["labels"][0]
            if name == 'vote_percentage':
                score = agg(info["preds"])
            else:
                score = agg(info["probs"])
            pred = 1 if score >= 0.5 else 0
            y_true.append(true)
            y_pred.append(pred)
            scores.append(score)

        if not y_true:
            continue

        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        mcc  = matthews_corrcoef(y_true, y_pred)
        y_bin = np.array(y_true)
        fpr, tpr, _ = roc_curve(y_bin, scores)
        eer     = fpr[np.nanargmin(np.abs((1-tpr)-fpr))]
        roc_auc = auc(fpr, tpr)
        ap      = average_precision_score(y_bin, scores)

        metrics_all[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'mcc': mcc,
            'auc': roc_auc,
            'ap': ap,
            'eer': eer
        }

        if f1 > best_f1:
            best_f1 = f1
            best_acc = acc
            best_method = name

    return val_loss, best_acc, best_f1, best_method, metrics_all


def train_model(model, train_loader, val_loader, device, output_folder):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=0.05
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2
    )
    scaler = GradScaler()
    early_stop = 5
    no_improve = 0
    best_val_loss = float('inf')
    model_path = os.path.join(output_folder, "SwinTransformer.pth")

    n1 = NUM_EPOCHS // 3
    n2 = 2 * NUM_EPOCHS // 3

    if os.path.exists(model_path):
        logging.warning("SwinTransformer already trained; loading weights.")
        model.load_state_dict(torch.load(model_path))
        return model, None

    train_losses, val_losses = [], []
    train_accs, val_accs, val_f1s = [], [], []

    for epoch in range(NUM_EPOCHS):
        if epoch == n1:
            for p in model.backbone.layers[2].parameters():
                p.requires_grad = True
            logging.info("Unfroze Swin backbone layer 2")
        if epoch == n2:
            for p in model.backbone.layers[1].parameters():
                p.requires_grad = True
            logging.info("Unfroze Swin backbone layer 1")

        # ---- TRAIN ----
        model.train()
        running, corr, tot = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        for imgs, lbls, _ in pbar:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            with autocast():
                outs = model(imgs)
                loss = criterion(outs, lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * imgs.size(0)
            preds = outs.argmax(dim=1)
            corr += (preds == lbls).sum().item()
            tot  += lbls.size(0)
            pbar.set_postfix(loss=f"{running/tot:.4f}", acc=f"{corr/tot:.4f}")

        train_losses.append(running/tot)
        train_accs.append(corr/tot)

        # ---- VALIDATE ----
        val_loss, val_acc, val_f1, best_method, metrics_all = validate_video_level(
            model, val_loader, device
        )
        scheduler.step(val_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        logging.info(
            f"Epoch {epoch+1}: Train Loss {train_losses[-1]:.4f}, Acc {train_accs[-1]:.4f} | "
            f"Val Loss {val_loss:.4f}, Acc {val_acc:.4f}, F1 {val_f1:.4f} ({best_method})"
        )

        metrics_file = os.path.join(output_folder, "Swin_validation_metrics.txt")
        with open(metrics_file, "a") as f:
            f.write(f"\nEpoch {epoch+1}\n")
            for m, met in metrics_all.items():
                f.write(
                    f"{m}: Acc {met['accuracy']:.4f}, Prec {met['precision']:.4f}, "
                    f"Rec {met['recall']:.4f}, F1 {met['f1']:.4f}, MCC {met['mcc']:.4f}, "
                    f"AUC {met['auc']:.4f}, AP {met['ap']:.4f}, EER {met['eer']:.4f}\n"
                )
            f.write(f"Best: {best_method}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_method_overall = best_method
            no_improve = 0
            torch.save(model.state_dict(), model_path)
            logging.info(f"Epoch {epoch+1}: new best val loss {val_loss:.4f}, model saved.")
        else:
            no_improve += 1
            if no_improve >= early_stop:
                logging.info("Early stopping.")
                break

    # ---- PLOT TRAINING CURVES ----
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1),   val_losses,   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Loss Curves'); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'training_loss.png'))
    plt.close()

    plt.figure()
    plt.plot(range(1, len(train_accs)+1), train_accs, label='Train Acc')
    plt.plot(range(1, len(val_accs)+1),   val_accs,   label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title('Accuracy Curves'); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'training_accuracy.png'))
    plt.close()

    plt.figure()
    plt.plot(range(1, len(val_f1s)+1), val_f1s, label='Val F1')
    plt.xlabel('Epoch'); plt.ylabel('F1 Score')
    plt.title('F1 Score Curve'); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'validation_f1.png'))
    plt.close()

    plt.figure()
    plt.plot(val_losses, val_f1s, marker='o')
    plt.xlabel('Validation Loss')
    plt.ylabel('F1 Score')
    plt.title('F1 vs Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'f1_vs_loss.png'))
    plt.close()

    model.best_agg_method = best_method_overall
    return model, best_method_overall


def SwinTransformer_main(train_ds, val_ds, test_ds, device, output_folder):
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = SwinTransformerWrapper().to(device)

    for name, p in model.named_parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True
    for p in model.backbone.layers[3].parameters():
        p.requires_grad = True
    for p in model.se.parameters():
        p.requires_grad = True

    model, best_method = train_model(model, train_loader, val_loader, device, output_folder)
    return test_loader, model, best_method
