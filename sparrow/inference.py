#!/usr/bin/env python
"""
This script uses Triton Inference Server to perform object detection using the MegaDetectorV6 model,
and then for each "animal" detection, it crops the bounding box and sends it to a classification
model (e.g., AI4GAmazonClassification) for species classification.

Results are:
- Logged to CSV
- For JPEG outputs, all bounding boxes (with labels & scores) are stored as JSON in EXIF
  UserComment so server-side can turn overlays on/off later.

By default, the saved image pixels are on (boxes drawn). You can disable drawing with:
    DRAW_BOXES=flase
"""

import os
import time
import csv
import json
import logging
import threading
from datetime import datetime

from PIL import Image, ImageFile, ImageDraw, ImageFont
import numpy as np
import tritonclient.http as httpclient
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import requests
from filelock import FileLock
from utils.sparrow_id import get_hardware_id
from utils.detection_utils import non_max_suppression, scale_boxes
import piexif  # EXIF metadata

# Setup Logging & Folders
LOGS_DIR = "/app/logs"
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "inference.log")),
        logging.StreamHandler()
    ]
)
model_logger = logging.getLogger("model_settings")
log = logging.getLogger("inference")

ONLY_SAVE_ANIMALS = os.getenv("ONLY_SAVE_ANIMALS", "false").strip().lower() == "true"
DRAW_BOXES = os.getenv("DRAW_BOXES", "true").strip().lower() == "true"

# Model Config Sync
CONFIG_DIR = "/app/config"
MODEL_CONFIG_FILE = os.path.join(CONFIG_DIR, "model_settings.json")
MODEL_CONFIG_LOCK = f"{MODEL_CONFIG_FILE}.lock"

SERVER_BASE_URL = os.getenv("SERVER_BASE_URL", "https://server.sparrow-earth.com").rstrip("/")
MODEL_ENDPOINT = f"{SERVER_BASE_URL}/model_settings"
model_logger.info(f"Model settings endpoint: {MODEL_ENDPOINT}")
AUTH_KEY_PATH = "/app/config/access_key.txt"

DEFAULT_MODEL_CONFIG = {
    "selected_model": "AI4GAmazonClassification",
    "lables": {
        "0": "Dasyprocta",
        "1": "Bos",
        "2": "Pecari",
        "3": "Mazama",
        "4": "Cuniculus",
        "5": "Leptotila",
        "6": "Human",
        "7": "Aramides",
        "8": "Tinamus",
        "9": "Eira",
        "10": "Crax",
        "11": "Procyon",
        "12": "Capra",
        "13": "Dasypus",
        "14": "Sciurus",
        "15": "Crypturellus",
        "16": "Tamandua",
        "17": "Proechimys",
        "18": "Leopardus",
        "19": "Equus",
        "20": "Columbina",
        "21": "Nyctidromus",
        "22": "Ortalis",
        "23": "Emballonura",
        "24": "Odontophorus",
        "25": "Geotrygon",
        "26": "Metachirus",
        "27": "Catharus",
        "28": "Cerdocyon",
        "29": "Momotus",
        "30": "Tapirus",
        "31": "Canis",
        "32": "Furnarius",
        "33": "Didelphis",
        "34": "Sylvilagus",
        "35": "Unknown"
    },
    "classification_enabled": True,
    "keep_blanks": False,
    "detection_threshold": 0.4
}

os.makedirs(CONFIG_DIR, exist_ok=True)


def load_model_config():
    """Load model_settings.json (create default if missing)."""
    if not os.path.isfile(MODEL_CONFIG_FILE):
        model_logger.info("No model_settings.json found. Creating default.")
        save_model_config(DEFAULT_MODEL_CONFIG)
        return DEFAULT_MODEL_CONFIG.copy()
    try:
        with FileLock(MODEL_CONFIG_LOCK, timeout=5):
            with open(MODEL_CONFIG_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        model_logger.error(f"Failed to load model_settings.json: {e}")
        return DEFAULT_MODEL_CONFIG.copy()


def save_model_config(config):
    """Atomically save model_settings.json."""
    tmp_path = f"{MODEL_CONFIG_FILE}.tmp"
    try:
        with FileLock(MODEL_CONFIG_LOCK, timeout=5):
            with open(tmp_path, "w") as f:
                json.dump(config, f, indent=4)
            os.replace(tmp_path, MODEL_CONFIG_FILE)
    except Exception as e:
        model_logger.error(f"Failed to save model_settings.json: {e}")


def fetch_model_settings(unique_id, auth_key):
    """
    Fetch updated model settings from the server.
    If different from local, update local file.
    """
    payload = {"unique_id": unique_id, "auth_key": auth_key}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(MODEL_ENDPOINT, json=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            server_model_config = response.json()
            current_config = load_model_config()
            if server_model_config != current_config:
                model_logger.info("Model settings have changed. Updating local config.")
                save_model_config(server_model_config)
            else:
                model_logger.info("Model settings unchanged.")
        else:
            model_logger.warning(f"Model settings fetch failed: {response.status_code} - {response.text}")
    except Exception as e:
        model_logger.warning(f"Could not fetch model settings: {e}")


def model_settings_fetch_loop(unique_id, auth_key):
    """Background thread to periodically fetch model settings."""
    model_logger.info("Started model settings background fetch thread.")
    while True:
        fetch_model_settings(unique_id, auth_key)
        time.sleep(60)


def get_current_model_name():
    """Get current classification model name."""
    return load_model_config().get("selected_model", "AI4GAmazonClassification")


def get_current_labels():
    """Get current label dictionary."""
    return load_model_config().get("lables", DEFAULT_MODEL_CONFIG["lables"])


def is_classification_enabled():
    """Whether classification is enabled."""
    return load_model_config().get("classification_enabled", True)


def is_keep_blanks_enabled():
    """Whether blank images should be kept."""
    return load_model_config().get("keep_blanks", False)


def get_detection_threshold():
    """Get detection confidence threshold."""
    cfg = load_model_config()
    return cfg.get("detection_threshold", DEFAULT_MODEL_CONFIG["detection_threshold"])


# Image & Preprocess Utils
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_font():
    """Return a Pillow built-in bitmap font."""
    return ImageFont.load_default()


def letterbox(im, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, stride=32):
    """Resize and pad image to meet stride-multiple constraints."""
    if isinstance(im, Image.Image):
        im = T.ToTensor()(im)
    shape = im.shape[1:]  # [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = dw % stride, dh % stride
    elif scaleFill:
        dw, dh = 0, 0
        new_unpad = new_shape
        r = (new_shape[1] / shape[1], new_shape[0] / shape[0])
    dw /= 2
    dh /= 2
    # Resize
    if shape[::-1] != new_unpad:
        resize_transform = T.Resize(new_unpad[::-1], interpolation=T.InterpolationMode.BILINEAR, antialias=False)
        im = resize_transform(im)
    # Pad
    padding = (
        int(round(dw - 0.1)), int(round(dw + 0.1)),
        int(round(dh + 0.1)), int(round(dh - 0.1))
    )
    im = F.pad(im * 255.0, padding, value=114) / 255.0
    return im


# MegaDetector classes
class_name_to_id = {0: "animal", 1: "person", 2: "vehicle"}
colors = ["red", "blue", "purple"]


def preprocess_classification(img):
    """
    Preprocess a PIL image for classification:
    Resizes to 224x224, converts to tensor -> numpy [1,3,224,224].
    """
    img = img.resize((224, 224))
    img_tensor = T.ToTensor()(img)
    img_np = img_tensor.numpy()
    img_np = np.expand_dims(img_np, axis=0).astype(np.float32)
    return img_np


# Triton / IO Setup
TRITON_URL = (os.getenv("TRITON_SERVER_URL") or os.getenv("TRITON_URL", "http://triton:8000")).strip().rstrip("/")
if TRITON_URL.startswith(("http://", "https://")):
    TRITON_URL = TRITON_URL.split("://", 1)[1]

client = httpclient.InferenceServerClient(url=TRITON_URL, network_timeout=600.0)

# Megadetector model
megadetector_model_name = "megadetectorv6"

# Input and output directories
input_dir = "/app/images/"
output_dir = "/app/static/gallery/"
os.makedirs(output_dir, exist_ok=True)

# CSV for logging detections
csv_file = '/app/static/data/detections.csv'
os.makedirs(os.path.dirname(csv_file), exist_ok=True)


def write_to_csv(image_name, detection, confidence, date):
    """Append detection results to CSV."""
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Image Name', 'Detection', 'Confidence Score', 'Date'])
        writer.writerow([image_name, detection, confidence, date])


def save_jpeg_with_boxes(img, boxes_meta, out_path):
    """
    Save a JPEG with bounding boxes stored as JSON in EXIF UserComment.

    boxes_meta: list of dicts, each like:
      {
        "x1": float (normalized 0-1),
        "y1": float,
        "x2": float,
        "y2": float,
        "label": str,
        "score": float,
        "class_id": int,
        "source": str,
        "model": str or None
      }
    """
    exif_bytes_in = img.info.get("exif", b"")
    if exif_bytes_in:
        try:
            exif_dict = piexif.load(exif_bytes_in)
        except Exception:
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    else:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

    payload = json.dumps(boxes_meta).encode("utf-8")
    # EXIF UserComment should start with an encoding prefix
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = b"ASCII\0\0\0" + payload

    exif_bytes_out = piexif.dump(exif_dict)
    img.save(out_path, format="JPEG", exif=exif_bytes_out)


# Background Settings Fetch
try:
    with open(AUTH_KEY_PATH, "r") as f:
        AUTH_KEY = f.read().strip()
except Exception as e:
    model_logger.error(f"Failed to read auth key: {e}")
    AUTH_KEY = None

try:
    UNIQUE_ID = get_hardware_id()
    model_logger.info(f"Hardware ID loaded: {UNIQUE_ID}")
except Exception as e:
    model_logger.error(f"Failed to get hardware ID: {e}")
    UNIQUE_ID = None

if AUTH_KEY and UNIQUE_ID:
    model_thread = threading.Thread(
        target=model_settings_fetch_loop,
        args=(UNIQUE_ID, AUTH_KEY),
        daemon=True
    )
    model_thread.start()

# Main Processing Loop
while True:
    for image_name in os.listdir(input_dir):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, image_name)
            print(f"Opening {image_path}")
            attempt = 0
            while attempt < 3:
                if attempt != 0:
                    time.sleep(5)
                print(f"Attempt {attempt}")
                try:
                    image = Image.open(image_path).convert("RGB")
                    break
                except Exception as e:
                    print(e)
                    attempt += 1
            if attempt == 3:
                os.remove(image_path)
                print(f"Removed {image_path} without processing.")
                continue

            # Prepare image for MegaDetector
            img_lb = letterbox(image, new_shape=(640, 640), auto=False, stride=32)
            image_np = img_lb.numpy()
            image_np = np.expand_dims(image_np, axis=0)

            md_inputs = [httpclient.InferInput("images", image_np.shape, "FP32")]
            md_inputs[0].set_data_from_numpy(image_np)
            md_outputs = [httpclient.InferRequestedOutput("output0")]

            md_results = client.infer(megadetector_model_name, md_inputs, outputs=md_outputs)
            output_data = md_results.as_numpy("output0")
            print(f"MegaDetector output shape: {output_data.shape}")

            # Extract datetime from filename
            date_str = image_name.split('_')[-1].split('.')[0][:14]
            date = datetime.strptime(date_str, '%Y%m%d%H%M%S')

            # Non-max suppression
            conf_thres = get_detection_threshold()
            pred = non_max_suppression(
                torch.tensor(output_data),
                conf_thres=conf_thres,
                iou_thres=0.5,
                agnostic=False
            )[0].numpy()
            print(f"MegaDetector predictions: {pred}")

            # Handle blank
            if pred.size == 0:
                if is_keep_blanks_enabled():
                    write_to_csv(image_name, "blank", 1.0, date)
                    os.makedirs(output_dir, exist_ok=True)
                    image.save(os.path.join(output_dir, image_name))
                    print(f"Saved blank image to {output_dir}")
                os.remove(image_path)
                print(f"Removed source file {image_path} (blank)")
                continue

            if len(pred) > 0:
                # Scale boxes back to original image size
                pred[:, :4] = scale_boxes([640, 640], pred[:, :4], np.array(image).shape)
                xyxy = pred[:, :4]
                md_confidence = pred[:, 4]
                md_class_id = pred[:, 5].astype(int)

                font = load_font()

                drew_any = False            # we had at least one kept detection
                skipped_count = 0           # how many non-animal detections we skip

                # Metadata for EXIF (one dict per detection)
                boxes_meta = []
                img_w, img_h = image.size

                # Only create drawing context if we actually want boxes rendered
                annotated_img = image.copy() if DRAW_BOXES else image
                draw = ImageDraw.Draw(annotated_img) if DRAW_BOXES else None

                for i in range(len(pred)):
                    cls_id = md_class_id[i]

                    # Skip non-animals (person=1, vehicle=2) if ONLY_SAVE_ANIMALS is enabled
                    if ONLY_SAVE_ANIMALS and cls_id in (1, 2):
                        try:
                            x1_s, y1_s, x2_s, y2_s = [float(v) for v in xyxy[i]]
                        except Exception:
                            x1_s = y1_s = x2_s = y2_s = -1.0
                        label_skipped = "person" if cls_id == 1 else "vehicle"
                        conf_s = float(md_confidence[i])
                        log.info(
                            f"Skipping {label_skipped} (conf={conf_s:.2f}) due to ONLY_SAVE_ANIMALS; "
                            f"image={image_name}, box=({x1_s:.1f},{y1_s:.1f},{x2_s:.1f},{y2_s:.1f})"
                        )
                        skipped_count += 1
                        continue

                    md_label = class_name_to_id[cls_id]
                    det_conf = md_confidence[i]
                    x1, y1, x2, y2 = xyxy[i]

                    # Only run classification if it's an "animal" AND classification is enabled
                    if cls_id == 0 and is_classification_enabled():
                        cropped = image.crop((x1, y1, x2, y2))
                        cropped_np = preprocess_classification(cropped)

                        clf_inputs = [httpclient.InferInput("input", cropped_np.shape, "FP32")]
                        clf_inputs[0].set_data_from_numpy(cropped_np)
                        clf_outputs = [httpclient.InferRequestedOutput("output")]

                        current_model_name = get_current_model_name()
                        clf_results = client.infer(current_model_name, clf_inputs, outputs=clf_outputs)

                        clf_output = clf_results.as_numpy("output")  # [1, 36]
                        exp_scores = np.exp(clf_output[0])
                        probs = exp_scores / np.sum(exp_scores)
                        pred_class = int(np.argmax(probs))
                        clf_conf = float(np.max(probs))

                        labels_dict = get_current_labels()
                        detected_class = labels_dict.get(str(pred_class), "Unknown")
                        if clf_conf < 0.8:
                            detected_class = "Unknown"

                        write_to_csv(image_name, detected_class, clf_conf, date)
                        label = f"{detected_class} {clf_conf:.2f}"

                        stored_label = detected_class
                        stored_conf = clf_conf
                        stored_model = current_model_name
                    else:
                        # For person/vehicle, or if classification disabled, use MD label only
                        write_to_csv(image_name, md_label, det_conf, date)
                        label = f"{md_label} {det_conf:.2f}"

                        stored_label = md_label
                        stored_conf = float(det_conf)
                        stored_model = None

                    # Optionally draw bounding box and label
                    if DRAW_BOXES and draw is not None:
                        draw.rectangle(xyxy[i], outline=colors[cls_id], width=2)
                        text_bbox = draw.textbbox((xyxy[i][0], xyxy[i][1] - 20), label, font=font)
                        draw.rectangle(
                            [text_bbox[0], text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2],
                            fill=colors[cls_id]
                        )
                        draw.text((xyxy[i][0] + 2, xyxy[i][1] - 20), label, font=font, fill='white')

                    drew_any = True  # we have at least one kept detection

                    # Store normalized coordinates + label in metadata list
                    norm_x1 = float(x1) / float(img_w)
                    norm_y1 = float(y1) / float(img_h)
                    norm_x2 = float(x2) / float(img_w)
                    norm_y2 = float(y2) / float(img_h)

                    boxes_meta.append(
                        {
                            "x1": norm_x1,
                            "y1": norm_y1,
                            "x2": norm_x2,
                            "y2": norm_y2,
                            "label": stored_label,
                            "score": float(stored_conf),
                            "class_id": int(cls_id),
                            "source": "megadetectorv6",
                            "model": stored_model,
                        }
                    )

                # Per-image summary for skipped detections
                if ONLY_SAVE_ANIMALS and skipped_count:
                    log.info(f"{image_name}: skipped {skipped_count} non-animal detection(s) due to ONLY_SAVE_ANIMALS")

                # If we filtered out everything (e.g., only people/vehicles), treat as blank
                if not drew_any:
                    if is_keep_blanks_enabled():
                        write_to_csv(image_name, "blank", 1.0, date)
                        os.makedirs(output_dir, exist_ok=True)
                        image.save(os.path.join(output_dir, image_name))
                        print(f"Saved blank image to {output_dir}")
                    os.remove(image_path)
                    print(f"Removed source file {image_path} (all detections filtered)")
                    continue

                # Save CLEAN image, embedding boxes in EXIF if JPEG
                out_path = os.path.join(output_dir, image_name)
                img_to_save = image  # always save original pixels

                if image_name.lower().endswith((".jpg", ".jpeg")):
                    save_jpeg_with_boxes(img_to_save, boxes_meta, out_path)
                else:
                    img_to_save.save(out_path)

                print(f"Saved {out_path}")

            # Remove original after processing
            os.remove(image_path)
            print(f"Removed {image_path}")

    # Check new images every 10s
    time.sleep(10)
