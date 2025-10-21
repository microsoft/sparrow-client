#!/usr/bin/env python3
"""
Audio monitoring pipeline that records from a USB microphone (threshold- or schedule-based), 
slices clips into overlapping windows, generates mel-spectrograms, and sends batches to a Triton model for inference. 
Window scores are aggregated to per-second and audio-level results; recordings with confidence -> SUMMARY_THRESHOLD are kept and appended to /app/static/data/audio_detections.csv. 
Settings are loaded from /app/config/audio_settings.json and periodically refreshed from SERVER_BASE_URL using the device unique_id.
"""

import os
import csv
import math
import shutil
from pathlib import Path
from typing import Optional, List, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import logging
import subprocess
import requests
import json
import threading
import time
from filelock import FileLock, Timeout
from datetime import datetime
import tempfile
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import tritonclient.http as httpclient
from utils.dataset_dataloader import ResizeTo
from utils.sparrow_id import get_hardware_id

# Configuration Paths
CONFIG_DIR = "/app/config"
CONFIG_FILE = os.path.join(CONFIG_DIR, "audio_settings.json")
LOG_FILE = "/app/logs/audio_script.log"
TEMP_CONFIG_FILE = f"{CONFIG_FILE}.tmp"

# CSV path (detections only)
CSV_PATH = "/app/static/data/audio_detections.csv"

# Default Configuration (device capture)
DEFAULT_CONFIG = {
    "mode": "schedule",
    "SAMPLE_RATE": 44100,
    "MONITOR_DURATION": 1,
    "RECORD_DURATION": 2,
    "THRESHOLD": 2500,
    "selected_model": "megadetector_birds_v1",
    "SCHEDULE": {
        "interval_minutes": 5,
        "duration_minutes": 1
    }
}

# Pipeline / Model Config
BATCH_SIZE = 16
NUM_WORKERS = 0
SAMPLE_RATE = 48_000 # pipeline sample rate for windows/spectrograms
WINDOW_SIZE_SEC = 5.0
OVERLAP_SEC = 4.0
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 224
TOP_DB = 80.0

# Triton Server
TRITON_URL = (os.getenv("TRITON_SERVER_URL") or os.getenv("TRITON_URL", "http://triton:8000")).strip().rstrip("/")
if TRITON_URL.startswith(("http://", "https://")):
    TRITON_URL = TRITON_URL.split("://", 1)[1]

SELECTED_MODEL = DEFAULT_CONFIG["selected_model"]

# Audio-level summary threshold (probability)
SUMMARY_THRESHOLD = 0.80

# Clean up spectrograms folder after each file
DELETE_SPECTROGRAMS_AFTER = True

# Setup Logging & Folders
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.info(f"CSV will be written to: {CSV_PATH}")

# Configuration Management
def load_config():
    if not os.path.exists(CONFIG_FILE):
        logger.info("Configuration file not found. Creating default configuration.")
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()
    try:
        with FileLock(f"{CONFIG_FILE}.lock", timeout=5):
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
        logger.info("Configuration loaded successfully.")
        return config
    except Timeout:
        logger.error("Timeout acquiring config lock.")
        return DEFAULT_CONFIG.copy()
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in config: {e}. Using defaults.")
        return DEFAULT_CONFIG.copy()
    except Exception as e:
        logger.error(f"Unexpected error loading config: {e}. Using defaults.")
        return DEFAULT_CONFIG.copy()

def save_config(config):
    try:
        with FileLock(f"{CONFIG_FILE}.lock", timeout=5):
            with open(TEMP_CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=4)
            os.replace(TEMP_CONFIG_FILE, CONFIG_FILE)
        logger.info("Configuration saved successfully.")
    except Timeout:
        logger.error("Timeout acquiring lock to save config.")
    except Exception as e:
        logger.error(f"Unexpected error while saving config: {e}")

# Generate Unique ID (using shared sparrow_id.py)
try:
    UNIQUE_ID = get_hardware_id()
    logger.info(f"Hardware ID loaded: {UNIQUE_ID}")
except Exception as e:
    logger.critical(f"Cannot proceed without valid unique_id: {e}")
    raise SystemExit(1)

# Configuration Fetching
SERVER_BASE_URL = os.getenv("SERVER_BASE_URL", "https://server.sparrow-earth.com").rstrip("/")
SERVER_URL = f"{SERVER_BASE_URL}/get_scheduleaudio"

try:
    with open("/app/config/access_key.txt", "r") as f:
        AUTH_KEY = f.read().strip()
except Exception as e:
    logger.error(f"Failed to read access key from /app/config/access_key.txt: {e}")
    exit(1)

def fetch_settings(unique_id):
    payload = {"unique_id": unique_id, "auth_key": AUTH_KEY}
    try:
        logger.info("Attempting to fetch audio settings from the server.")
        response = requests.post(SERVER_URL, json=payload, timeout=10)
        if response.status_code == 200:
            server_settings = response.json()
            logger.info(f"Audio settings retrieved from server: {server_settings}")
            current = load_config()
            if server_settings != current:
                logger.info("New audio settings detected. Updating configuration.")
                save_config(server_settings)
                update_audio_settings()
            else:
                logger.info("No changes in audio settings.")
        else:
            logger.error(f"Failed to fetch settings. Status Code: {response.status_code}, Detail: {response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error while fetching settings from server: {e}")

# Audio Monitoring Globals
# capture/recorder (device) settings - populated by update_audio_settings()
CAPTURE_SR = None
MONITOR_DURATION = None
RECORD_DURATION = None
THRESHOLD = None
DEVICE = None

# Directories
RECORDINGS_DIR = "/app/recordings"
PROCESSING_DIR = os.path.join(RECORDINGS_DIR, "processing")

MODE = "threshold"
INTERVAL_SEC = 60 * 60
DURATION_SEC = 10 * 60

# Microphone Detection
def detect_usb_microphone():
    try:
        res = subprocess.run(["arecord", "-l"], capture_output=True, text=True, check=True)
        for line in res.stdout.splitlines():
            ll = line.lower()
            if "card" in ll and "device" in ll and "usb" in ll:
                try:
                    card = line.split("card")[1].split(":")[0].strip()
                    dev  = line.split("device")[1].split(":")[0].strip()
                    dev_id = f"plughw:{card},{dev}"
                    logger.info(f"Detected USB audio at {dev_id}")
                    return dev_id
                except Exception:
                    continue
        logger.warning("No USB device found via arecord -l.")
    except Exception as e:
        logger.error(f"Error detecting audio device: {e}")
    return None

# Update audio settings
def update_audio_settings():
    global CAPTURE_SR, MONITOR_DURATION, RECORD_DURATION, THRESHOLD
    global DEVICE, MODE, INTERVAL_SEC, DURATION_SEC, SELECTED_MODEL

    config = load_config()
    CAPTURE_SR      = int(config.get("SAMPLE_RATE",      DEFAULT_CONFIG["SAMPLE_RATE"]))
    MONITOR_DURATION = int(config.get("MONITOR_DURATION", DEFAULT_CONFIG["MONITOR_DURATION"]))
    RECORD_DURATION  = int(config.get("RECORD_DURATION",  DEFAULT_CONFIG["RECORD_DURATION"]))
    THRESHOLD        = int(config.get("THRESHOLD",        DEFAULT_CONFIG["THRESHOLD"]))
    SELECTED_MODEL   = config.get("selected_model", DEFAULT_CONFIG["selected_model"])

    MODE = config.get("mode", "threshold")
    sched = config.get("SCHEDULE", {})
    INTERVAL_SEC = int(sched.get("interval_minutes", 60)) * 60
    DURATION_SEC = int(sched.get("duration_minutes", 10)) * 60

    DEVICE = detect_usb_microphone()
    if DEVICE:
        logger.info(
            f"Mode={MODE} | CAPTURE_SR={CAPTURE_SR} | mon={MONITOR_DURATION}s | rec={RECORD_DURATION}s | "
            f"th={THRESHOLD} | intv={INTERVAL_SEC}s | dur={DURATION_SEC}s | dev={DEVICE} | "
            f"model={SELECTED_MODEL} | PIPELINE_SR={SAMPLE_RATE}"
        )
    else:
        logger.warning("No valid audio device detected.")

# Dataset for .npy specs
class BioacousticsInferenceDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, root: Optional[str] = None,
                 x_col: str = "spec_name", target_size: Optional[List[int]] = None):
        super().__init__()
        self.df = dataframe
        self.root = root
        self.x_col = x_col
        self.paths = [os.path.join(self.root, p) for p in self.df[self.x_col].astype(str).tolist()]
        self._resize = ResizeTo(target_size) if target_size is not None else None

    def __len__(self): return len(self.df)

    def _load_npy(self, idx: int):
        path = self.paths[idx]
        arr = np.load(path)
        return arr, path

    def __getitem__(self, idx: int):
        arr, path = self._load_npy(idx)
        arr = arr.astype(np.float32, copy=False)
        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim == 3 and (arr.shape[0] not in (1, 2, 3) and arr.shape[-1] in (1, 2, 3)):
            arr = np.moveaxis(arr, -1, 0)
        x = torch.from_numpy(arr)  # [C,H,W]
        if self._resize is not None:
            x = self._resize(x)
        return x, path

# Windowing / Spec creation
def build_windows_for_file(audio_path, window_size_sec, overlap_sec, sample_rate):
    window_size = int(window_size_sec * sample_rate)
    hop_size = int((window_size_sec - overlap_sec) * sample_rate)
    windows, window_idx = [], 0

    sound_duration = librosa.get_duration(filename=audio_path)
    duration_samples = int(sound_duration * sample_rate)
    if duration_samples <= 0:
        return windows

    if duration_samples <= window_size:
        windows.append({'window_id': 0, 'sound_path': audio_path, 'start': 0, 'end': duration_samples})
        return windows

    num_windows = max(1, math.ceil((duration_samples - window_size) / hop_size) + 1)
    for i in range(num_windows):
        start = i * hop_size
        end = min(start + window_size, duration_samples)  # include last partial window
        if end <= start:  # safety
            continue
        windows.append({'window_id': window_idx, 'sound_path': audio_path, 'start': start, 'end': end})
        window_idx += 1

    return windows

def compute_all_mel_spectrograms_gpu(windows: List[dict],
                                     sample_rate: int,
                                     n_fft: int,
                                     hop_length: Optional[int],
                                     n_mels: int,
                                     top_db: float,
                                     spectrograms_path: str,
                                     save_npy: bool,
                                     fill_noise: bool,
                                     noise_db_mean: Optional[float],
                                     noise_db_std: float,
                                     random_state: Optional[int],
                                     storage_dtype: str) -> None:
    if hop_length is None:
        hop_length = n_fft // 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    Path(spectrograms_path).mkdir(parents=True, exist_ok=True)

    by_sid = defaultdict(list)
    for idx, win in enumerate(windows):
        by_sid[win["sound_path"]].append((idx, win))

    for audio_file_path, items in tqdm(by_sid.items(), desc="files", leave=False):
        y, orig_sr = sf.read(audio_file_path, dtype="float32", always_2d=False)
        if isinstance(y, np.ndarray) and y.ndim == 2:
            y = y.mean(axis=1)
        wav_cpu = torch.from_numpy(y).unsqueeze(0)  # [1, num_samples]

        if orig_sr != sample_rate:
            wav_cpu = torchaudio.functional.resample(wav_cpu, orig_freq=orig_sr, new_freq=sample_rate)

        mel_tf = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
            f_min=0.0, f_max=sample_rate / 2.0, power=2.0, center=False, norm="slaney", mel_scale="slaney"
        ).to(device)
        to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=top_db).to(device)

        for _, win in items:
            start, end = int(win["start"]), int(win["end"])
            sound_filename = os.path.basename(win["sound_path"]).split(".")[0]
            npy_path = os.path.join(spectrograms_path, f"{sound_filename}_{start}_{end}.npy")
            if os.path.exists(npy_path):
                continue
            wav_win = wav_cpu[:, start:end].to(device)
            S = mel_tf(wav_win).squeeze(0)  # [n_mels, T]
            S_db = to_db(S)
            if save_npy:
                arr = S_db.detach().cpu().numpy()
                if storage_dtype == "float16":
                    arr = arr.astype("float16", copy=False)
                else:
                    arr = arr.astype("float32", copy=False)
                np.save(npy_path, arr)
            del wav_win, S, S_db
            if device.type == "cuda":
                torch.cuda.empty_cache()

# Triton batching (specs -> logits)
def run_inference_triton(dataloader: DataLoader, sample_rate: int) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    client = httpclient.InferenceServerClient(url=TRITON_URL, network_timeout=600.0)
    if not client.is_server_live():
        raise RuntimeError("Triton server not live")

    md = client.get_model_metadata(SELECTED_MODEL)
    in_name, out_name = md["inputs"][0]["name"], md["outputs"][0]["name"]

    all_paths, all_logits = [], []

    for x, paths in tqdm(dataloader, desc="batches", leave=False):
        # x: [B, 1, H, W] torch
        x_np = x.numpy().astype(np.float32, copy=False)
        inp = httpclient.InferInput(in_name, x_np.shape, "FP32")
        inp.set_data_from_numpy(x_np, binary_data=True)
        res = client.infer(model_name=SELECTED_MODEL, inputs=[inp])
        logits = res.as_numpy(out_name)  # [B,1] or [B]
        logits = np.squeeze(logits, axis=-1) if logits.ndim == 2 and logits.shape[1] == 1 else logits
        all_logits.append(logits)
        all_paths.extend(paths)

    audios = ["_".join(os.path.basename(p).replace(".npy", "").split("_")[:-2]) for p in all_paths]
    starts = [int(os.path.basename(p).split("_")[-2]) / sample_rate for p in all_paths]
    ends   = [int(os.path.basename(p).split("_")[-1].replace(".npy", "")) / sample_rate for p in all_paths]

    all_logits = np.concatenate(all_logits) if len(all_logits) else np.array([], dtype=np.float32)
    probabilities = 1 / (1 + np.exp(-all_logits)) if all_logits.size else np.array([], dtype=np.float32)
    predictions  = (probabilities > 0.5).astype(int)    if probabilities.size else np.array([], dtype=np.int32)
    return audios, starts, ends, predictions, probabilities

# Aggregations / Summary
def process_inference_results_per_second(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    unique_audios = df['audio'].unique()
    all_results = []
    for audio in unique_audios:
        audio_df = df[df['audio'] == audio].copy()
        if audio_df.empty:
            continue
        min_start = int(np.floor(audio_df['start(s)'].min()))
        max_end   = int(np.ceil (audio_df['end(s)'].max()))
        for second in range(min_start, max_end):
            overlapping = audio_df[
                ((audio_df['start(s)'] <= second) & (audio_df['end(s)'] > second)) |
                ((audio_df['start(s)'] < second + 1) & (audio_df['end(s)'] >= second + 1))
            ]
            if len(overlapping) == 0:
                continue
            weights = []
            for _, row in overlapping.iterrows():
                overlap_start = max(row['start(s)'], second)
                overlap_end   = min(row['end(s)'], second + 1)
                overlap_duration = max(0, overlap_end - overlap_start)
                weights.append(overlap_duration)
            weights = np.array(weights)
            if weights.sum() == 0:
                continue
            weights = weights / weights.sum()
            avg_pred = np.average(overlapping['prediction'],  weights=weights)
            avg_prob = np.average(overlapping['probability'], weights=weights)
            avg_conf = np.average(overlapping['confidence'],  weights=weights)
            all_results.append({
                'audio': audio,
                'second': second,
                'count_overlaps': len(overlapping),
                'prediction': 1 if avg_pred >= 0.5 else 0,
                'avg_prediction': avg_pred,
                'avg_probability': avg_prob,
                'avg_confidence': avg_conf,
            })
    results_df = pd.DataFrame(all_results).sort_values(['audio', 'second']).reset_index(drop=True)
    out_path = os.path.join(os.path.dirname(csv_path), 'per_second_results.csv')
    results_df.to_csv(out_path, index=False)
    return results_df

def summarize_audio_level(per_second_csv_path: str, threshold: float) -> pd.DataFrame:
    if not os.path.exists(per_second_csv_path): return pd.DataFrame()
    df = pd.read_csv(per_second_csv_path)
    if df.empty: return pd.DataFrame()
    out_rows = []
    for audio, g in df.groupby("audio"):
        g = g.sort_values("second")
        max_prob = float(g["avg_probability"].max()) if not g.empty else 0.0
        pos = g[g["avg_probability"] >= threshold]
        if not pos.empty:
            start = float(pos["second"].min())
            end   = float(pos["second"].max() + 1)
            detected = True
        else:
            start, end, detected = None, None, False
        out_rows.append({"audio": audio, "detected": detected, "start": start, "end": end, "max_prob": max_prob})
    summary = pd.DataFrame(out_rows)[["audio", "detected", "start", "end", "max_prob"]]
    out_path = os.path.join(os.path.dirname(per_second_csv_path), "audio_summary.csv")
    summary.to_csv(out_path, index=False)
    return summary

def cleanup_spectrograms(path: str):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)

# CSV logging (detections only)
def write_to_csv(audio_name, detection, confidence, date):
    lock = FileLock(CSV_PATH + ".lock", timeout=5)
    try:
        with lock:
            header_needed = not os.path.exists(CSV_PATH)
            with open(CSV_PATH, mode="a", newline="") as f:
                writer = csv.writer(f)
                if header_needed and f.tell() == 0:
                    writer.writerow(["Audio Name", "Detection", "Confidence Score", "Date"])
                writer.writerow([audio_name, detection, confidence, date])
    except Exception as e:
        logger.error(f"Failed to write CSV '{CSV_PATH}': {e}")

def log_audio_detection(audio_path: str, prob: float):
    audio_name = os.path.basename(audio_path)
    detection = True
    confidence = round(float(prob), 6)
    date_str = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    write_to_csv(audio_name, detection, confidence, date_str)

# End-to-end classify one file
def classify_audio_file(audio_path: str) -> Tuple[bool, float]:
    """
    Full pipeline on a single recorded file in PROCESSING_DIR:
    window -> mel (npy) -> Triton -> per-second -> audio summary.
    Returns (detected, max_prob).
    """
    base = os.path.splitext(os.path.basename(audio_path))[0]

    # Outputs live under PROCESSING_DIR/inference_output/
    output_dir = os.path.join(PROCESSING_DIR, "inference_output")
    spectrograms_path = os.path.abspath(os.path.join(output_dir, "spectrograms"))
    os.makedirs(spectrograms_path, exist_ok=True)

    # Build windows for just this audio_path
    windows = build_windows_for_file(audio_path, WINDOW_SIZE_SEC, OVERLAP_SEC, SAMPLE_RATE)
    if len(windows) == 0:
        logger.warning(f"No windows produced for {audio_path}")
        return False, 0.0

    # Compute specs
    compute_all_mel_spectrograms_gpu(
        windows=windows,
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        top_db=TOP_DB,
        spectrograms_path=spectrograms_path,
        save_npy=True,
        fill_noise=False,           # no noise fill needed at inference
        noise_db_mean=None,
        noise_db_std=3.0,
        random_state=42,
        storage_dtype="float32",
    )

    # Dataset / loader
    df = pd.DataFrame(windows)
    df['spec_name'] = df.apply(
        lambda row: f"{os.path.basename(row['sound_path']).split('.')[0]}_{row['start']}_{row['end']}.npy",
        axis=1
    )
    n_frames = int(np.ceil((WINDOW_SIZE_SEC * SAMPLE_RATE - N_FFT) / HOP_LENGTH)) + 1
    target_size = (N_MELS, n_frames)

    dataset = BioacousticsInferenceDataset(
        dataframe=df, root=spectrograms_path, x_col="spec_name", target_size=target_size
    )
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Triton inference (batched)
    try:
        audios, starts, ends, predictions, probabilities = run_inference_triton(dataloader, SAMPLE_RATE)
    except Exception as e:
        logger.error(f"Triton inference failed for {audio_path}: {e}")
        return False, 0.0

    # Save window-level results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "inference_results.csv")
    results_df = pd.DataFrame({
        'audio': audios,
        'start(s)': starts,
        'end(s)': ends,
        'prediction': predictions,
        'probability': probabilities,
        'confidence': np.abs(probabilities - 0.5) * 2,
    }).sort_values('confidence', ascending=False)
    results_df.to_csv(results_path, index=False)

    # Per-second & summary
    process_inference_results_per_second(results_path)
    per_second_csv = os.path.join(output_dir, "per_second_results.csv")
    summary = summarize_audio_level(per_second_csv, threshold=SUMMARY_THRESHOLD)

    # Extract decision for this file
    detected, max_prob = False, 0.0
    if not summary.empty:
        row = summary[summary["audio"] == base]
        if not row.empty:
            detected = bool(row["detected"].iloc[0])
            max_prob = float(row["max_prob"].iloc[0])

    # Cleanup specs to save disk
    if DELETE_SPECTROGRAMS_AFTER:
        del dataloader, dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cleanup_spectrograms(spectrograms_path)

    return detected, max_prob

# Recording helpers
def _wait_for_stable_file(path: str, tries: int = 4, sleep_s: float = 0.2) -> bool:
    prev = -1
    for _ in range(tries):
        if not os.path.exists(path):
            time.sleep(sleep_s)
            continue
        size = os.path.getsize(path)
        if size > 0 and size == prev:
            return True
        prev = size
        time.sleep(sleep_s)
    return os.path.exists(path) and os.path.getsize(path) > 0

def _processing_path_for_timestamp(ts: str) -> str:
    return os.path.join(PROCESSING_DIR, f"{ts}.wav")

def _final_recordings_path_for(ts: str) -> str:
    return os.path.join(RECORDINGS_DIR, f"{ts}.wav")

def _record_to_processing(duration_sec: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = _processing_path_for_timestamp(ts)
    cmd = [
        "arecord", "-D", DEVICE,
        "-t", "wav", "-c", "1",
        "-f", "S16_LE", "-r", str(CAPTURE_SR),
        "-d", str(duration_sec), out,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if not _wait_for_stable_file(out):
        raise RuntimeError(f"Recorded file missing/empty: {out}")
    return out

def _keep_move_to_recordings(processing_file: str) -> str:
    ts = os.path.splitext(os.path.basename(processing_file))[0]
    dest = _final_recordings_path_for(ts)
    try:
        os.replace(processing_file, dest)
        logger.info(f"Moved kept file -> {dest}")
        return dest
    except Exception as e:
        logger.error(f"Failed to move kept file to recordings: {e}")
        return processing_file

# Monitoring & Recording
def monitor_audio():
    fd, temp_file = tempfile.mkstemp(prefix="monitor_", suffix=".wav")
    os.close(fd)
    command = ["arecord", "-D", DEVICE, "-t", "wav", "-c", "1", "-f", "S16_LE",
               "-r", str(CAPTURE_SR), "-d", str(MONITOR_DURATION), temp_file]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if not _wait_for_stable_file(temp_file):
            logger.error(f"Monitor file not ready: {temp_file}")
            return None
        data, _ = sf.read(temp_file, dtype="int16")
        return data.flatten()
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        return None
    finally:
        try: os.remove(temp_file)
        except Exception: pass

def record_triggered_audio():
    try:
        path_proc = _record_to_processing(RECORD_DURATION)
        logger.info(f"Trigger recording saved (processing): {path_proc} (pipeline scoring...)")
        detected, max_prob = classify_audio_file(path_proc)
        logger.info(f"Audio summary > detected={detected}, max_prob={max_prob:.4f}")
        if detected and max_prob >= SUMMARY_THRESHOLD:
            dest = _keep_move_to_recordings(path_proc)
            log_audio_detection(dest, max_prob)
        else:
            try:
                os.remove(path_proc)
            except Exception as e:
                logger.warning(f"Failed to delete {path_proc}: {e}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Recording failed (trigger): {e}")
    except Exception as e:
        logger.error(f"Unexpected error during triggered recording: {e}")

# Monitoring Loop
def audio_monitoring_loop():
    logger.info("Audio monitoring thread started.")
    while True:
        update_audio_settings()

        if DEVICE is None:
            time.sleep(10)
            continue

        if MODE == "threshold":
            data = monitor_audio()
            if data is not None:
                max_amp = np.max(np.abs(data))
                logger.info(f"Max amplitude: {max_amp}")
                if max_amp > THRESHOLD:
                    logger.info("Threshold exceeded -> recording")
                    record_triggered_audio()
            time.sleep(MONITOR_DURATION)
        else:  # schedule mode
            try:
                logger.info(f"Scheduled recording ({DURATION_SEC}s)")
                path_proc = _record_to_processing(DURATION_SEC)
                logger.info(f"Scheduled recording saved (processing): {path_proc} (pipeline scoring...)")
                detected, max_prob = classify_audio_file(path_proc)
                logger.info(f"Audio summary -> detected={detected}, max_prob={max_prob:.4f}")
                if detected and max_prob >= SUMMARY_THRESHOLD:
                    dest = _keep_move_to_recordings(path_proc)
                    log_audio_detection(dest, max_prob)
                else:
                    try:
                        os.remove(path_proc)
                    except Exception as e:
                        logger.warning(f"Failed to delete {path_proc}: {e}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Scheduled record failed: {e}")
            except Exception as e:
                logger.error(f"Unexpected error in schedule: {e}")
            to_sleep = max(0, INTERVAL_SEC - DURATION_SEC)
            logger.info(f"Next scheduled record in {to_sleep}s")
            time.sleep(to_sleep)

# Settings Fetch Thread
def settings_fetching_loop(unique_id):
    logger.info("Settings fetching thread started.")
    while True:
        fetch_settings(unique_id)
        time.sleep(120)

# Main
def main():
    logger.info("Audio recording script started.")
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    os.makedirs(PROCESSING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    update_audio_settings()

    threading.Thread(target=audio_monitoring_loop, daemon=True).start()
    threading.Thread(target=settings_fetching_loop, args=(UNIQUE_ID,), daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Audio recording script terminated by user.")
    except Exception as e:
        logger.critical(f"Unexpected error in main thread: {e}")

if __name__ == "__main__":
    main()
