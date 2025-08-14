import os
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from torchvision import models
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_curve, auc, average_precision_score
)

from utils.constants import MAIN_OUTPUT_FOLDER
from utils.constants import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE


def validate_video_level(model, val_loader, device):
    """Video-level validation with multiple aggregation methods"""
    model.eval()
    video_dict = defaultdict(lambda: {"labels": [], "probs": [], "preds": []})
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels, video_paths in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            for i, vpath in enumerate(video_paths):
                video_dict[vpath]["labels"].append(labels[i].item())
                video_dict[vpath]["probs"].append(probs[i].cpu().numpy())
                video_dict[vpath]["preds"].append(preds[i].item())

    val_loss = running_loss / total_samples if total_samples > 0 else 0.0

    methods = {
        'mean_prob':         lambda x: np.mean([p[1] for p in x]),
        'max_prob':          lambda x: np.max([p[1] for p in x]),
        'min_prob':          lambda x: np.min([p[1] for p in x]),
        'median_prob':       lambda x: np.median([p[1] for p in x]),
        'trimmed_mean_prob': lambda x: np.mean(sorted([p[1] for p in x])[1:-1])
                                     if len(x) > 2 else np.mean([p[1] for p in x]),
        'vote_percentage':   lambda x: np.mean(x)
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
            score = agg(info["preds"]) if name == 'vote_percentage' else agg(info["probs"])
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
        eer     = fpr[np.nanargmin(np.abs((1 - tpr) - fpr))]
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
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2
    )


    scaler = GradScaler()
    early_stop_patience = 5
    no_improve_counter = 0
    best_val_loss = float('inf')
    model_path = os.path.join(output_folder, "ResNet50.pth")

    if os.path.exists(model_path):
        logging.warning("ResNet50 model already trained. Loading weights.")
        model.load_state_dict(torch.load(model_path))
        return model, None

    train_losses, train_accs = [], []
    val_losses, val_accs, val_f1s = [], [], []

    for epoch in range(NUM_EPOCHS):
        # ---- TRAINING ----
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False)
        for images, labels, _ in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

            train_bar.set_postfix({
                "loss": f"{running_train_loss/total_train:.4f}",
                "acc":  f"{correct_train/total_train:.4f}"
            })

        epoch_train_loss = running_train_loss / total_train
        epoch_train_acc  = correct_train / total_train

        # ---- VALIDATION ----
        val_loss, val_acc, val_f1, best_method, metrics_all = validate_video_level(
            model, val_loader, device
        )
        scheduler.step(val_loss)

        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        logging.info(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
            f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f} ({best_method})"
        )
        with open(os.path.join(output_folder, "ResNet50_validation_metrics.txt"), "a+") as f:
            f.write(f"\nEpoch {epoch+1}\n")
            for name, met in metrics_all.items():
                f.write(
                    f"{name} -> Acc: {met['accuracy']:.4f}, Prec: {met['precision']:.4f}, "
                    f"Rec: {met['recall']:.4f}, F1: {met['f1']:.4f}, "
                    f"MCC: {met['mcc']:.4f}, AUC: {met['auc']:.4f}, "
                    f"AP: {met['ap']:.4f}, EER: {met['eer']:.4f}\n"
                )
            f.write(f"Best: {best_method}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_method_overall = best_method
            no_improve_counter = 0
            torch.save(model.state_dict(), model_path)
            logging.info(f"Epoch {epoch+1}: new best loss {val_loss:.4f}. Model saved.")
        else:
            no_improve_counter += 1
            if no_improve_counter >= early_stop_patience:
                logging.info("Early stopping triggered.")
                break

    
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1),   val_losses,   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Loss Curves'); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(output_folder, 'training_loss.png')); plt.close()

    plt.figure()
    plt.plot(range(1, len(train_accs)+1), train_accs, label='Train Acc')
    plt.plot(range(1, len(val_accs)+1),   val_accs,   label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title('Accuracy Curves'); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(output_folder, 'training_accuracy.png')); plt.close()

    plt.figure()
    plt.plot(range(1, len(val_f1s)+1), val_f1s, label='Val F1')
    plt.xlabel('Epoch'); plt.ylabel('F1 Score')
    plt.title('F1 Curve'); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(output_folder, 'validation_f1.png')); plt.close()

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


def ResNet50_main(train_dataset, val_dataset, test_dataset, device, output_folder):
    train_loader = DataLoader(
                        train_dataset, 
                        batch_size=BATCH_SIZE, 
                        shuffle=True, 
                        num_workers=4, 
                        pin_memory=True
                    )

    val_loader   = DataLoader(
                        val_dataset,   
                        batch_size=BATCH_SIZE, 
                        shuffle=False, 
                        num_workers=4, 
                        pin_memory=True
                    )

    test_loader  = DataLoader(
                        test_dataset,  
                        batch_size=BATCH_SIZE, 
                        shuffle=False, 
                        num_workers=4, 
                        pin_memory=True
                    )

    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if any([n in name for n in ['layer2', 'layer3', 'layer4', 'fc']]):
            param.requires_grad = True

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 2)
    )
    model.to(device)

    model, best_method = train_model(model, train_loader, val_loader, device, output_folder)
    return test_loader, model, best_method
