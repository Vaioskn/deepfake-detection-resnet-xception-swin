import os, cv2, glob, random, logging, torch, numpy as np
from PIL import Image
import mediapipe as mp
from facenet_pytorch import MTCNN
from torchvision import transforms
from torch.utils.data import Dataset

from utils.constants import (
    FRAMES_PER_VIDEO, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
)

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_mp_detectors = [
    mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.3),
    mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3),
]
_mtcnn = MTCNN(keep_all=False, device=_device)

def _detect_face_bbox(frame: np.ndarray):
    """(x1, y1, x2, y2) or None."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    for det in _mp_detectors:
        res = det.process(rgb)
        if res.detections:
            bb = res.detections[0].location_data.relative_bounding_box
            h, w, _ = frame.shape
            return bb.xmin * w, bb.ymin * h, (bb.xmin + bb.width) * w, (bb.ymin + bb.height) * h

    boxes, _ = _mtcnn.detect(rgb)
    if boxes is not None:
        return tuple(boxes[0])
    return None

def crop_face_from_frame(frame: np.ndarray, fallback_size=(224, 224), margin=0.2):
    """Return PIL.Image of the face crop or None."""
    bbox = _detect_face_bbox(frame)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    h, w, _ = frame.shape
    dw, dh = (x2 - x1) * margin, (y2 - y1) * margin
    x1, y1 = int(max(x1 - dw, 0)),            int(max(y1 - dh, 0))
    x2, y2 = int(min(x2 + dw, w)),            int(min(y2 + dh, h))

    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None
    return Image.fromarray(cv2.resize(face, fallback_size))

def sample_frames_fixed_interval(video_path: str, frames_per_video: int = FRAMES_PER_VIDEO):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total <= 0:
        return []
    step = max(total // frames_per_video, 1)
    return [i * step for i in range(frames_per_video) if i * step < total]

class DeepFakeFrameDataset(Dataset):
    """
    Returns (tensor_image, label, video_path).
    If `face_crop=True` tries to crop a face; otherwise uses the whole frame.
    """

    def __init__(self, frame_info_list, transform=None, face_crop=True):
        self.frame_info_list = frame_info_list
        self.transform = transform
        self.face_crop = face_crop

    def __len__(self):
        return len(self.frame_info_list)

    def __getitem__(self, idx):
        video_path, frame_num, label = self.frame_info_list[idx]
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_pil, attempt, ok = None, 0, False
        while attempt < 5 and frame_pil is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ok, frame = cap.read()
            if not ok:
                break

            if self.face_crop:
                frame_pil = crop_face_from_frame(frame)
            else:
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if frame_pil is None:
                frame_num = random.randint(0, max(total_frames - 1, 0))
                attempt += 1
        cap.release()

        if frame_pil is None and ok:
            frame_pil = Image.fromarray(cv2.resize(frame, (224, 224)))

        if self.transform:
            frame_pil = self.transform(frame_pil)
        return frame_pil, torch.tensor(label, dtype=torch.long), video_path


def _gather_frames(video_list, require_face):
    info = []
    for vid_path, label in video_list:
        for idx in sample_frames_fixed_interval(vid_path):
            if not require_face:
                info.append((vid_path, idx, label))
                continue
            cap = cv2.VideoCapture(vid_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            cap.release()
            if ok and crop_face_from_frame(frame) is not None:
                info.append((vid_path, idx, label))
    return info

_aug_resize_256 = transforms.Resize((256, 256))
_to_tensor_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

FACE_TRAIN_TRANSFORM = transforms.Compose([
    _aug_resize_256,
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    _to_tensor_norm,
])

FRAME_TRAIN_TRANSFORM = transforms.Compose([
    _aug_resize_256,
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    _to_tensor_norm,
])

VAL_TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    _to_tensor_norm,
])


def prepare_data_faces(real_dataset_path: str, fake_dataset_path: str):
    """
    Face-cropped frames pipeline (exactly your original behaviour).
    Returns: train_ds, val_ds, test_ds
    """
    random.seed(RANDOM_SEED)
    real_videos = sorted(glob.glob(os.path.join(real_dataset_path, "*.mp4")))
    fake_videos = sorted(glob.glob(os.path.join(fake_dataset_path, "*.mp4")))
    
    vids = [(v, 0) for v in real_videos] + [(v, 1) for v in fake_videos]
    random.shuffle(vids)

    n_total = len(vids)
    n_train, n_val = int(TRAIN_RATIO * n_total), int(VAL_RATIO * n_total)
    train_v, val_v, test_v = vids[:n_train], vids[n_train:n_train+n_val], vids[n_train+n_val:]
    logging.info(f"[faces] split: {len(train_v)} train / {len(val_v)} val / {len(test_v)} test")

    train_info = _gather_frames(train_v, require_face=True)
    val_info   = _gather_frames(val_v,   require_face=True)
    test_info  = _gather_frames(test_v,  require_face=True)
    logging.info(f"[faces] frames: train={len(train_info)} val={len(val_info)} test={len(test_info)}")

    train_ds = DeepFakeFrameDataset(train_info, FACE_TRAIN_TRANSFORM, face_crop=True)
    val_ds   = DeepFakeFrameDataset(val_info,   VAL_TEST_TRANSFORM,  face_crop=True)
    test_ds  = DeepFakeFrameDataset(test_info,  VAL_TEST_TRANSFORM,  face_crop=True)
    return train_ds, val_ds, test_ds


def prepare_data_full_frame(real_dataset_path: str, fake_dataset_path: str):
    """
    Full-frame pipeline (no face detection, faster).
    Returns: train_ds, val_ds, test_ds
    """
    random.seed(RANDOM_SEED)
    real_videos = sorted(glob.glob(os.path.join(real_dataset_path, "*.mp4")))
    fake_videos = sorted(glob.glob(os.path.join(fake_dataset_path, "*.mp4")))

    vids = [(v, 0) for v in real_videos] + [(v, 1) for v in fake_videos]
    random.shuffle(vids)

    n_total = len(vids)
    n_train, n_val = int(TRAIN_RATIO * n_total), int(VAL_RATIO * n_total)
    train_v, val_v, test_v = vids[:n_train], vids[n_train:n_train+n_val], vids[n_train+n_val:]
    logging.info(f"[frames] split: {len(train_v)} train / {len(val_v)} val / {len(test_v)} test")

    train_info = _gather_frames(train_v, require_face=False)
    val_info   = _gather_frames(val_v,   require_face=False)
    test_info  = _gather_frames(test_v,  require_face=False)
    logging.info(f"[frames] frames: train={len(train_info)} val={len(val_info)} test={len(test_info)}")

    train_ds = DeepFakeFrameDataset(train_info, FRAME_TRAIN_TRANSFORM, face_crop=False)
    val_ds   = DeepFakeFrameDataset(val_info,   VAL_TEST_TRANSFORM,    face_crop=False)
    test_ds  = DeepFakeFrameDataset(test_info,  VAL_TEST_TRANSFORM,    face_crop=False)
    return train_ds, val_ds, test_ds
