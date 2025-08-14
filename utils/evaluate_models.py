import os, csv, cv2, torch, logging, numpy as np, matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from random import sample
from PIL import Image, ImageDraw
from torchvision import transforms
from collections import defaultdict
from sklearn.preprocessing import label_binarize
from pytorch_grad_cam import GradCAM
from utils.prepare_data import VAL_TEST_TRANSFORM
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef, roc_curve, auc,
    average_precision_score, precision_recall_curve
)

from utils.prepare_data import (
    crop_face_from_frame, sample_frames_fixed_interval
)

from utils.constants import READ_DATASET, FAKE_DATASET

def _collect_frame_predictions(model, test_loader, device):
    model.eval()
    video_dict = defaultdict(lambda: {"labels": [], "probs": [], "preds": []})

    with torch.no_grad():
        for imgs, labels, video_paths in tqdm(test_loader, desc="Testing"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            for i, vpath in enumerate(video_paths):
                video_dict[vpath]["labels"].append(labels[i].item())
                video_dict[vpath]["probs"].append(probs[i].cpu().numpy())
                video_dict[vpath]["preds"].append(preds[i].item())

    return video_dict

def _aggregate_video_predictions(video_dict, method_name):
    """Return metrics dict."""
    agg_fn = {
        "mean_prob":         lambda x: np.mean([p[1] for p in x]),
        "max_prob":          lambda x: np.max([p[1] for p in x]),
        "min_prob":          lambda x: np.min([p[1] for p in x]),
        "median_prob":       lambda x: np.median([p[1] for p in x]),
        "trimmed_mean_prob": lambda x: np.mean(sorted([p[1] for p in x])[1:-1])
                                    if len(x) > 2 else np.mean([p[1] for p in x]),
        "vote_percentage":   lambda x: np.mean(x),
    }[method_name]

    v_labels, v_preds, v_scores = [], [], []
    error_records, correct_records = [], []

    for vpath, info in video_dict.items():
        true = info["labels"][0]
        score = agg_fn(info["preds"] if method_name == "vote_percentage"
                       else info["probs"])
        pred = 1 if score >= .5 else 0

        v_labels.append(true);  v_preds.append(pred);  v_scores.append(score)

        cap = cv2.VideoCapture(vpath); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        row = {
            "video_name": os.path.basename(vpath),
            "total_frames": total,
            "true_label": true,
            "predicted_label": pred,
            "method": method_name,
            "confidence_score": round(score, 4),
            "result_type": ("TP" if pred == 1 else "TN") if pred == true
                           else ("FP" if pred == 1 else "FN"),
        }
        (correct_records if pred == true else error_records).append(row)

    a = accuracy_score(v_labels, v_preds)
    p = precision_score(v_labels, v_preds, zero_division=0)
    r = recall_score(v_labels, v_preds, zero_division=0)
    f1 = f1_score(v_labels, v_preds, zero_division=0)
    mcc = matthews_corrcoef(v_labels, v_preds)
    y_bin = label_binarize(v_labels, classes=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y_bin, v_scores)
    eer = fpr[np.nanargmin(np.abs((1 - tpr) - fpr))]
    ap = average_precision_score(y_bin, v_scores)

    return {
        "accuracy": a, "precision": p, "recall": r, "f1": f1,
        "mcc": mcc, "ap": ap, "eer": eer,
        "labels": v_labels, "preds": v_preds, "scores": v_scores,
        "errors": error_records, "corrects": correct_records,
    }

def _log_metrics(metrics_all, best_method, model_type, out_dir):
    with open(os.path.join(out_dir, f"{model_type}_metrics.txt"), "w") as f:
        f.write("Video-level metrics per aggregation method:\n\n")
        for name, met in metrics_all.items():
            f.write(f"Method: {name}\n")
            f.write(f"  Accuracy : {met['accuracy']:.4f}\n")
            f.write(f"  Precision: {met['precision']:.4f}\n")
            f.write(f"  Recall   : {met['recall']:.4f}\n")
            f.write(f"  F1       : {met['f1']:.4f}\n")
            f.write(f"  MCC      : {met['mcc']:.4f}\n")
            f.write(f"  AP       : {met['ap']:.4f}\n")
            f.write(f"  EER      : {met['eer']:.4f}\n\n")
        f.write(f"Best aggregation (by F1): {best_method}\n")

def _write_csv(rows, name, model_type, out_dir):
    path = os.path.join(out_dir, f"{model_type}_{name}.csv")
    if not rows:
        logging.info(f"No records for {name}.")
        return
    with open(path, "w", newline="") as f:
        hdr = ["video_name","total_frames","true_label","predicted_label",
               "method","confidence_score","result_type"]
        csv.DictWriter(f, hdr).writeheader();  csv.DictWriter(f, hdr).writerows(rows)
    logging.info(f"Saved {name} to {path}")

def get_target_layer(model, model_type):
    """Get target layer for Grad-CAM based on model architecture"""
    if model_type == "ResNet50":
        return [model.layer4[-1]]
    elif model_type == "Xception":
        return [model.base.conv4]
    elif model_type == "SwinTransformer":
        return [model.backbone.layers[-1].blocks[-1].norm1]
    else:
        return [list(model.children())[-2]] 

def generate_heatmap_grid(model, model_type, device, video_path, transform, face_crop):
    """
    Generate a grid of heatmaps for a video (5x6 grid of frames)
    Returns: PIL Image of the grid
    """
    frame_indices = sample_frames_fixed_interval(video_path)
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        if face_crop:
            frame_pil = crop_face_from_frame(frame)
            if frame_pil is None:
                continue
            orig_frame = np.array(frame_pil)
        else:
            orig_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            orig_frame = cv2.resize(orig_frame, (224, 224))
            frame_pil = Image.fromarray(orig_frame)
        
        input_tensor = transform(frame_pil).unsqueeze(0).to(device)
        frames.append((orig_frame, input_tensor, idx))
    
    cap.release()
    
    if not frames:
        return None
    
    target_layer = get_target_layer(model, model_type)
    cam = GradCAM(model=model, target_layers=target_layer)
    
    heatmap_images = []
    percentages = []
    
    for orig_frame, input_tensor, idx in frames:
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.softmax(output, dim=1)[0]
            pred_class = torch.argmax(output).item()
            pred_percent = prob[pred_class].item() * 100
        
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        
        frame_norm = orig_frame.astype(np.float32) / 255
        
        visualization = show_cam_on_image(
            frame_norm, 
            grayscale_cam,
            use_rgb=True
        )
        heatmap_images.append(visualization)
        percentages.append(f"{pred_percent:.1f}%")
    
    grid_img = Image.new('RGB', (6 * 224, 5 * (224 + 30)), color='white')
    draw = ImageDraw.Draw(grid_img)
    
    for i, (img, percent) in enumerate(zip(heatmap_images, percentages)):
        if i >= 30: 
            break
            
        row = i // 6
        col = i % 6
        
        img_pil = Image.fromarray(img)
        
        x = col * 224
        y = row * (224 + 30)
        
        grid_img.paste(img_pil, (x, y))
        
        bbox = draw.textbbox((0, 0), percent)
        text_width = bbox[2] - bbox[0]
        draw.text(
            (x + (224 - text_width) // 2, y + 224 + 5),
            percent,
            fill='black'
        )
    
    if hasattr(cam, 'activations_and_grads'):
        cam.activations_and_grads.release()
    if hasattr(cam, 'model'):
        cam.model.zero_grad()
    
    return grid_img


def _test_model(model, test_loader, device, model_type, out_dir, face_crop, agg_method=None):
    os.makedirs(out_dir, exist_ok=True)
    video_dict = _collect_frame_predictions(model, test_loader, device)

    valid_methods = [
        "mean_prob", "max_prob", "min_prob",
        "median_prob", "trimmed_mean_prob", "vote_percentage"
    ]

    if agg_method not in valid_methods:
        raise ValueError(f"Invalid aggregation method '{agg_method}'. Must be one of: {valid_methods}")

    metrics = _aggregate_video_predictions(video_dict, agg_method)
    metrics_all = {agg_method: metrics}

    _log_metrics(metrics_all, agg_method, model_type, out_dir)
    _write_csv(metrics["errors"],  "errors",  model_type, out_dir)
    _write_csv(metrics["corrects"], "correct", model_type, out_dir)

    labels_bin = label_binarize(metrics["labels"], classes=[0, 1]).ravel()
    cm = confusion_matrix(metrics["labels"], metrics["preds"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({agg_method})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_type}_confusion_matrix.png"))
    plt.close()

    fpr, tpr, _ = roc_curve(labels_bin, metrics["scores"])
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC Curve"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_type}_roc_curve.png"))
    plt.close()

    prec, rec, _ = precision_recall_curve(labels_bin, metrics["scores"])
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label=f"AP = {metrics['ap']:.2f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_type}_pr_curve.png"))
    plt.close()

    cam_dir = os.path.join(out_dir, 'cams')
    os.makedirs(cam_dir, exist_ok=True)
    categories = ['FN', 'FP', 'TP', 'TN']
    for cat in categories:
        os.makedirs(os.path.join(cam_dir, cat), exist_ok=True)

    selected_videos = {cat: [] for cat in categories}
    for record in metrics["errors"] + metrics["corrects"]:
        cat = record['result_type']
        if len(selected_videos[cat]) < 3:
            selected_videos[cat].append(record)

    for cat, records in selected_videos.items():
        for record in records:
            video_path = os.path.join(
                READ_DATASET if record['true_label'] == 0 else FAKE_DATASET,
                record['video_name']
            )
            if not os.path.exists(video_path):
                logging.warning(f"Video not found: {video_path}")
                continue

            grid = generate_heatmap_grid(
                model=model,
                model_type=model_type,
                device=device,
                video_path=video_path,
                transform=VAL_TEST_TRANSFORM,
                face_crop=face_crop
            )

            if grid:
                output_path = os.path.join(cam_dir, cat, f"{record['video_name']}.png")
                grid.save(output_path)
                logging.info(f"Saved heatmap grid for {record['video_name']} to {output_path}")



def test_model_faces(model, test_loader, device, model_type, output_folder, agg_method=None):
    """Evaluation routine for face‐cropped data."""
    _test_model(model, test_loader, device, model_type,
                output_folder, face_crop=True, agg_method=agg_method)

def test_model_full_frame(model, test_loader, device, model_type, output_folder, agg_method=None):
    """Evaluation routine for full‐frame data."""
    _test_model(model, test_loader, device, model_type,
                output_folder, face_crop=False, agg_method=agg_method)
