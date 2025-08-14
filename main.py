import logging
import os
import torch
from torch.utils.data import DataLoader

import Models.Xception as xcep
import Models.ResNet50 as resnet
import Models.SwinTransformer as swin
import Models.ResNetLSTM as reslstm

from utils.constants import *
from utils.prepare_data import prepare_data_faces, prepare_data_full_frame
from utils.evaluate_models import test_model_faces, test_model_full_frame

DATA_PREP_FUNCS = {
    "faces":  prepare_data_faces,
    "frames": prepare_data_full_frame,
}


TEST_FUNCS = {
    "faces":  test_model_faces,
    "frames": test_model_full_frame,
}


def train_model(model_type, train_ds, val_ds, test_ds,
                device, training_out, *, prep_name:str):
    """
    Returns (test_loader, model). For ResNetLSTM the *_ds are ignored because
    it builds its own sequence dataset; we forward the `prep_name` so it can
    decide face-crop vs full-frame.
    """
    if model_type == "ResNet50":
        return resnet.ResNet50_main(train_ds, val_ds, test_ds,
                                    device, training_out)

    if model_type == "Xception":
        return xcep.Xception_main(train_ds, val_ds, test_ds,
                                  device, training_out)

    if model_type == "SwinTransformer":
        return swin.SwinTransformer_main(train_ds, val_ds, test_ds,
                                         device, training_out)

    raise ValueError(f"Unknown model type: {model_type}")


def run_one_test(model, test_loader, test_fn,
                 device, model_type, test_out):
    os.makedirs(test_out, exist_ok=True)
    test_fn(model, test_loader, device, model_type, test_out)



def main():
    os.makedirs(MAIN_OUTPUT_FOLDER, exist_ok=True)
    logging.basicConfig(
        filename = os.path.join(MAIN_OUTPUT_FOLDER, "DeepFakeDetection.log"),
        filemode = "w+",
        level    = logging.INFO,
        format   = "%(asctime)s — %(levelname)s — %(message)s",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # ---------------------------
    # PHASE 1: PREPARE “faces” ONCE
    # ---------------------------
    logging.info("===== PHASE 1: PREPARING ‘faces’ DATA … =====")
    train_ds_faces, val_ds_faces, test_ds_faces = prepare_data_faces(
        READ_DATASET, FAKE_DATASET
    )

    _, _, test_ds_frames_for_phase1 = prepare_data_full_frame(
        READ_DATASET, FAKE_DATASET
    )

    for model_type in MODELS:
        logging.info(f"========== PHASE 1 (FACES) → MODEL: {model_type} ==========")

        training_out = os.path.join(MAIN_OUTPUT_FOLDER, model_type, "faces", "training")
        os.makedirs(training_out, exist_ok=True)

        faces_test_loader, model = train_model(
            model_type,
            train_ds_faces,
            val_ds_faces,
            test_ds_faces,
            device,
            training_out,
            prep_name="faces"
        )

        combo_name = "faces_on_faces"
        logging.info(f"    └─▶ TESTING {combo_name} …")
        test_out_dir = os.path.join(MAIN_OUTPUT_FOLDER, model_type, combo_name, "testing")
        run_one_test(model, faces_test_loader, test_model_faces,
                     device, model_type, test_out_dir)

        combo_name = "faces_on_frames"
        logging.info(f"    └─▶ TESTING {combo_name} …")
        frames_test_loader = DataLoader(
            test_ds_frames_for_phase1,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        test_out_dir = os.path.join(MAIN_OUTPUT_FOLDER, model_type, combo_name, "testing")
        run_one_test(model, frames_test_loader, test_model_full_frame,
                     device, model_type, test_out_dir)


    # ----------------------------
    # PHASE 2: PREPARE “frames” ONCE
    # ----------------------------
    logging.info("===== PHASE 2: PREPARING ‘frames’ DATA … =====")
    train_ds_frames, val_ds_frames, test_ds_frames = prepare_data_full_frame(
        READ_DATASET, FAKE_DATASET
    )

    faces_test_loader_for_phase2 = DataLoader(
        test_ds_faces,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    for model_type in MODELS:
        logging.info(f"========== PHASE 2 (FRAMES) → MODEL: {model_type} ==========")

        training_out = os.path.join(MAIN_OUTPUT_FOLDER, model_type, "frames", "training")
        os.makedirs(training_out, exist_ok=True)

        frames_test_loader, model = train_model(
            model_type,
            train_ds_frames,
            val_ds_frames,
            test_ds_frames,
            device,
            training_out,
            prep_name="frames"
        )

        combo_name = "frames_on_faces"
        logging.info(f"    └─▶ TESTING {combo_name} …")
        test_out_dir = os.path.join(MAIN_OUTPUT_FOLDER, model_type, combo_name, "testing")
        run_one_test(model, faces_test_loader_for_phase2, test_model_faces,
                     device, model_type, test_out_dir)

        combo_name = "frames_on_frames"
        logging.info(f"    └─▶ TESTING {combo_name} …")
        test_out_dir = os.path.join(MAIN_OUTPUT_FOLDER, model_type, combo_name, "testing")
        run_one_test(model, frames_test_loader, test_model_full_frame,
                     device, model_type, test_out_dir)

if __name__ == "__main__":
    main()