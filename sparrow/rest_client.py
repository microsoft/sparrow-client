"""
Main runtime for Sparrow telemetry.
Handles image/audio uploads, collection and transmission of system metrics,
backlog storage when the server is unavailable, and VE.Direct solar reads.
Logging is centralized to a rotating file so readings from sensors are captured
in the same log.
"""

import os
import time
import requests
import csv
import schedule
import logging
import psutil
from datetime import datetime
import smbus2
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import TimedRotatingFileHandler
import json
import threading
import socket
from urllib.parse import urlparse
import serial
import io
import utils.sensors as sensors
from utils.sparrow_id import get_hardware_id

from utils.sensors import (
    I2C_BUS,
    SENSOR_STATE,
    detect_sensors,
    read_env,
)

# Logger Bootstrap
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

# Configuration
image_output_dir = "/app/static/gallery/"
audio_output_dir = "/app/recordings/"
csv_file = "/app/static/data/detections.csv"
logs_dir = "/app/logs/restclient_logs.log"

SERVER_BASE_URL = os.getenv("SERVER_BASE_URL", "https://server.sparrow-earth.com").rstrip("/")

image_server_url   = f"{SERVER_BASE_URL}/uploads"
audio_server_url   = f"{SERVER_BASE_URL}/audio_uploads"
system_metrics_url = f"{SERVER_BASE_URL}/system_metrics"

try:
    with open("/app/config/access_key.txt", "r") as f:
        auth_key = f.read().strip()
except Exception as e:
    logger.error(f"Failed to read access key from /app/config/access_key.txt: {e}")
    exit(1)

metrics_backlog_file = "/app/static/data/metrics_backlog.jsonl"

sys_path = os.getenv("SYS_PATH", "/sys")
proc_path = os.getenv("PROC_PATH", "/proc")

# Generate Unique ID
try:
    unique_id = get_hardware_id()
    logger.info(f"Generated unique_id: {unique_id}")
except Exception:
    logger.critical("Cannot proceed without a valid unique_id.")
    exit(1)

# VE.Direct (Solar)
try:
    ved = serial.Serial("/dev/ttyUSB0", 19200, timeout=1)
    logger.info("Opened VE.Direct on /dev/ttyUSB0")
except Exception as e:
    logger.error(f"Could not open VE.Direct port: {e}")
    ved = None

# Helper Functions
def is_file_accessible(file_path, mode='r'):
    """Return True if a file is accessible with the given mode."""
    try:
        with open(file_path, mode):
            return True
    except IOError:
        return False

def is_server_online(url, timeout=8):
    parsed = urlparse(url)
    host = parsed.hostname
    scheme = (parsed.scheme or "http").lower()
    port = parsed.port or (443 if scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception as e:
        logger.warning(f"Connectivity check failed for {url}: {e}")
        return False

# CSV Row Reader
REQUIRED_COLS = ["Image Name", "Detection", "Confidence Score", "Date"]

def safe_csv_rows(path):
    """
    Get dict rows from a CSV safely using the existing comma delimiter.
    - Strips NUL bytes that cause 'line contains NUL'
    - Decodes with 'replace' to avoid Unicode errors
    - Skips rows missing required fields
    """
    try:
        with open(path, "rb") as f:
            raw = f.read()
        if b"\x00" in raw:
            logger.warning("Detections CSV contains NUL bytes; sanitizing.")
            raw = raw.replace(b"\x00", b"")
        text = raw.decode("utf-8", "replace")
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames or any(c not in reader.fieldnames for c in REQUIRED_COLS):
            logger.error(f"Unexpected headers in {path}: {reader.fieldnames}")
            return
        for lineno, row in enumerate(reader, start=2):
            try:
                if not all((row.get(c) or "").strip() for c in REQUIRED_COLS):
                    raise ValueError("missing required field(s)")
                yield row
            except Exception as e:
                logger.warning(f"Skipping malformed CSV line {lineno}: {e}; row={row!r}")
    except Exception as e:
        logger.error(f"Failed to read {path}: {e}")

# Solar Generation
def read_solar_generation():
    """Read a single VE.Direct frame and derive PV power, yields, voltage, and load power."""
    if ved is None:
        logger.warning("VE.Direct port not open")
        return None, {}, None, None

    pv = {}
    start = time.time()
    logger.info("Starting VE.Direct read.. (2s timeout)")
    while True:
        raw = ved.readline()
        now = time.time()
        if not raw or now - start > 2:
            break
        text = raw.decode("ascii", "ignore").strip()
        if "\t" in text:
            key, val = text.split("\t", 1)
        elif ":" in text:
            key, val = text.split(":", 1)
        else:
            continue
        key, val = key.strip(), val.strip()
        try:
            pv[key] = float(val)
        except ValueError:
            pv[key] = val

    logger.info(f"Parsed VE.Direct frame: {pv}")

    ppv = pv.get("PPV")

    yields = {}
    for days_ago, reg in [(0, "H20"), (1, "H22")]:
        raw_kwh = pv.get(reg)
        if isinstance(raw_kwh, (int, float)):
            yields[days_ago] = int(raw_kwh * 1000)  # kWh -> Wh
        else:
            yields[days_ago] = None
        logger.info(f"Yield {days_ago} day(s) ago ({reg}): {yields[days_ago]} Wh")

    raw_v  = pv.get("V")
    raw_il = pv.get("IL")

    ved_batt_v = (raw_v  / 1000.0) if isinstance(raw_v, (int, float)) else None
    ved_load_i = (raw_il / 1000.0) if isinstance(raw_il, (int, float)) else None

    ved_load_p = (ved_batt_v * ved_load_i) if (ved_batt_v is not None and ved_load_i is not None) else None

    return ppv, yields, ved_batt_v, ved_load_p

# Metrics Backlog
def append_metric_to_backlog(metric):
    """Append a metrics record to the backlog file as a JSON line."""
    try:
        backlog_dir = os.path.dirname(metrics_backlog_file)
        os.makedirs(backlog_dir, exist_ok=True)
        with open(metrics_backlog_file, "a") as f:
            f.write(json.dumps(metric) + "\n")
        logger.info("Appended current metric to backlog.")
    except Exception as e:
        logger.error(f"Failed to append metric to backlog: {e}")

def send_backlog_metrics():
    """Send backlog metrics to the server, keeping any unsent lines on failure."""
    if not is_server_online(system_metrics_url):
        logger.warning("Server for system metrics appears offline, skipping historical metrics.")
        return

    if not os.path.exists(metrics_backlog_file):
        return

    try:
        records = []
        with open(metrics_backlog_file, "r") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    records.append(record)
                except json.JSONDecodeError:
                    continue

        total_records = len(records)
        if total_records == 0:
            return

        logger.info(f"Starting upload of {total_records} historical metric record(s).")

        unsent_metrics = []
        sent_count = 0

        for record in records:
            try:
                response = requests.post(system_metrics_url, json=record, timeout=10)
                response.raise_for_status()
                sent_count += 1
            except requests.exceptions.RequestException:
                unsent_metrics.append(record)

        with open(metrics_backlog_file, "w") as f:
            for record in unsent_metrics:
                f.write(json.dumps(record) + "\n")

        logger.info(f"Finished uploading historical metrics: {sent_count} sent, {len(unsent_metrics)} remain in backlog.")
    except Exception as e:
        logger.error(f"Error processing backlog metrics: {e}")

# I2C Bus Init
try:
    bus = smbus2.SMBus(I2C_BUS)
    logger.info(f"I2C bus {I2C_BUS} initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize I2C bus {I2C_BUS}: {e}")
    bus = None

# Upload Functions
def upload_image_and_data(image_path, detection_data_list):
    """Upload one image with all matching detection rows; delete CSV rows on success elsewhere."""
    image_name = os.path.basename(image_path)
    if not is_server_online(image_server_url):
        logger.warning(f"Image server appears offline, skipping upload for {image_name}.")
        return False

    success = True
    for detection_data in detection_data_list:
        try:
            with open(image_path, "rb") as image_file:
                files = {"file": (image_name, image_file, "image/jpeg")}
                data = {
                    "auth_key": auth_key,
                    "unique_id": unique_id,
                    "image_name": detection_data["Image Name"],
                    "detection": detection_data["Detection"],
                    "confidence": float(detection_data["Confidence Score"]),
                    "date": detection_data["Date"],
                }
                logger.info(f"Sending image data to server: {data}")
                response = requests.post(image_server_url, files=files, data=data)
                response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to upload {image_name} with detection {detection_data['Detection']}: {e}")
            success = False
        else:
            logger.info(f"Successfully uploaded {image_name} with detection {detection_data['Detection']}")
    return success

def upload_audio_file(audio_path):
    """Upload a single .wav file; return True on success."""
    audio_name = os.path.basename(audio_path)
    try:
        with open(audio_path, "rb") as audio_file:
            files = {"file": (audio_name, audio_file, "audio/wav")}
            data  = {"auth_key": auth_key, "unique_id": unique_id}
            response = requests.post(audio_server_url, files=files, data=data)
            response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to upload audio file {audio_name}: {e}")
        return False
    else:
        logger.info(f"Successfully uploaded audio file: {audio_name}")
        return True

# Processing Functions
def remove_records_from_csv(image_name):
    """Remove all rows for the given image from the detections CSV in an atomic, crash-safe way."""
    logger.info(f"Pruning CSV rows for {image_name}")
    tmp = csv_file + ".tmp"
    try:
        with open(csv_file, "r", newline="") as fin, open(tmp, "w", newline="") as fout:
            r = csv.reader(fin)
            w = csv.writer(fout)

            header = next(r, None)
            if header:
                w.writerow(header)

            for row in r:
                if not row or row[0] == image_name:
                    logger.info(f"Removed record for {image_name}")
                    continue
                w.writerow(row)

        os.replace(tmp, csv_file)
    except Exception as e:
        logger.error(f"CSV update failed: {e}")
        try:
            os.remove(tmp)
        except Exception:
            pass

def process_and_upload_images():
    """Match images to CSV detections, upload, and delete files and rows on success."""
    logger.info("Starting to process and upload images.")
    if not os.path.exists(csv_file):
        logger.warning(f"CSV file {csv_file} not found.")
        return
    try:
        detection_records = {}
        for row in safe_csv_rows(csv_file):
            image_name = row["Image Name"]
            detection_records.setdefault(image_name, []).append(row)
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return

    if not os.path.exists(image_output_dir):
        logger.warning(f"Output directory {image_output_dir} not found.")
        return

    for image_name in os.listdir(image_output_dir):
        image_path = os.path.join(image_output_dir, image_name)
        if os.path.isfile(image_path):
            logger.info(f"Processing image {image_name}.")
            detection_data_list = detection_records.get(image_name, [])
            if detection_data_list:
                success = upload_image_and_data(image_path, detection_data_list)
                if success:
                    remove_records_from_csv(image_name)
                    try:
                        os.remove(image_path)
                        logger.info(f"Deleted local image file: {image_path}")
                    except Exception as e:
                        logger.error(f"Failed to delete image file {image_path}: {e}")
            else:
                logger.warning(f"No corresponding detection records found for {image_name}.")

def process_and_upload_audio():
    """
    Upload audio files in small batches with limited concurrency.
    Uses a single ThreadPoolExecutor across all chunks to avoid
    repeatedly creating/destroying worker threads.
    """
    logger.info("Starting batch processing of audio files.")

    if not os.path.exists(audio_output_dir):
        logger.warning(f"Audio output directory {audio_output_dir} not found.")
        return

    audio_files = [a for a in os.listdir(audio_output_dir) if a.lower().endswith(".wav")]
    total_files = len(audio_files)
    if total_files == 0:
        logger.info("No audio files found for processing.")
        return

    BATCH_LIMIT = 500
    CHUNK_SIZE = 5
    MAX_WORKERS = 3
    INTER_CHUNK_SLEEP = 1.0

    batch_files = audio_files[:BATCH_LIMIT]
    logger.info(
        f"Preparing to process {len(batch_files)} (of {total_files}) "
        f"audio files with max_workers={MAX_WORKERS}, chunk_size={CHUNK_SIZE}."
    )

    success_count = 0
    failure_count = 0

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i in range(0, len(batch_files), CHUNK_SIZE):
            slice_files = batch_files[i:i + CHUNK_SIZE]
            logger.info(f"Uploading chunk {i//CHUNK_SIZE + 1} with {len(slice_files)} files.")

            future_to_path = {}
            for audio_name in slice_files:
                audio_path = os.path.join(audio_output_dir, audio_name)
                if os.path.isfile(audio_path) and is_file_accessible(audio_path, mode="rb"):
                    future = executor.submit(upload_audio_file, audio_path)
                    future_to_path[future] = audio_path
                else:
                    logger.warning(f"Audio file {audio_name} is not accessible. Skipping.")

            for future in as_completed(future_to_path):
                audio_path = future_to_path[future]
                try:
                    ok = future.result()
                except Exception as e:
                    logger.error(f"Exception during audio upload for {audio_path}: {e}")
                    ok = False

                if ok:
                    try:
                        os.remove(audio_path)
                        success_count += 1
                    except Exception as e:
                        logger.error(f"Failed to delete audio file {audio_path}: {e}")
                        failure_count += 1
                else:
                    failure_count += 1

            if INTER_CHUNK_SLEEP > 0:
                time.sleep(INTER_CHUNK_SLEEP)

    remaining = total_files - len(batch_files)
    if failure_count == 0:
        logger.info(
            f"Audio upload complete: {success_count} succeeded, 0 failed. "
            f"{remaining} files remain for subsequent runs."
        )
    else:
        logger.error(
            f"Audio upload partial: {success_count} succeeded, {failure_count} failed. "
            f"{remaining} files remain for subsequent runs."
        )

# Metrics Functions
def gather_system_metrics():
    """
    Gathers system metrics including CPU, memory, disk, temperature/humidity, solar, and pressure.
    Always returns a dict; sensor failures become None values.
    """
    metrics = {
        "auth_key": auth_key,
        "unique_id": unique_id,
        "timestamp": datetime.now().isoformat(),
        "cpu_usage": psutil.cpu_percent(interval=1),
        "memory_usage": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "network_sent": psutil.net_io_counters().bytes_sent,
        "network_received": psutil.net_io_counters().bytes_recv,
        "uptime_seconds": int(time.time() - psutil.boot_time()),
    }
    
    try:
        env = read_env(bus, SENSOR_STATE)
        metrics["temperature_celsius"]        = env.get("t_c")
        metrics["humidity_percent"]           = env.get("rh_pct")
        metrics["bme688_pressure_pa"]         = env.get("p_pa")
        metrics["bme688_temperature_celsius"] = env.get("t_bme_c")
        metrics["bme688_humidity_percent"]    = env.get("rh_bme_pct")
    except Exception as e:
        logger.error(f"Sensor read failed inside gather_system_metrics(): {e}")
        metrics.update({
            "temperature_celsius": None,
            "humidity_percent": None,
            "bme688_pressure_pa": None,
            "bme688_temperature_celsius": None,
            "bme688_humidity_percent": None,
        })

    try:
        ppv, yields, ved_v, ved_load_p = read_solar_generation()
    except Exception as e:
        logger.error(f"VE.Direct read failed: {e}")
        ppv, yields, ved_v, ved_load_p = None, {}, None, None

    metrics["solar_generation_watts"]     = round(ppv, 2) if ppv is not None else None
    metrics["yield_today_wh"]             = yields.get(0) if isinstance(yields, dict) else None
    metrics["yield_yesterday_wh"]         = yields.get(1) if isinstance(yields, dict) else None
    metrics["vedirect_battery_voltage"]   = round(ved_v, 2) if ved_v is not None else None
    metrics["vedirect_load_power_watts"]  = round(ved_load_p, 2) if ved_load_p is not None else None

    logger.info(f"System metrics gathered: {metrics}")
    return metrics

def send_system_metrics():
    """Send the latest system metrics to the server, appending to backlog on failure."""
    logger.info("Metrics job: starting send_system_metrics()")
    try:
        send_backlog_metrics()  # this already checks connectivity for backlog replay
    except Exception as e:
        logger.error(f"Backlog replay failed: {e}")

    try:
        metrics = gather_system_metrics()
        logger.info(f"Sending system metrics to {system_metrics_url}")
        response = requests.post(system_metrics_url, json=metrics, timeout=10)
        response.raise_for_status()
        logger.info(f"Successfully sent system metrics: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to POST system metrics: {e}")
        append_metric_to_backlog(metrics)
    except Exception as e:
        logger.critical(f"send_system_metrics() unexpected error before POST: {e}", exc_info=True)


# Scheduling
executor = ThreadPoolExecutor(max_workers=5)

image_task_flag = threading.Event()
audio_task_flag = threading.Event()
metrics_task_flag = threading.Event()

def safe_run(job_func):
    """Submit a job function to the thread pool when the scheduler triggers it."""
    def wrapper():
        executor.submit(job_func)
    return wrapper

def process_and_upload_images_safe():
    """Run image processing safely, skipping if a previous run is still active."""
    if image_task_flag.is_set():
        logger.info("Image upload task already running. Skipping this cycle.")
        return
    try:
        image_task_flag.set()
        process_and_upload_images()
    finally:
        image_task_flag.clear()

def process_and_upload_audio_safe():
    """Run audio processing safely, skipping if a previous run is still active."""
    if audio_task_flag.is_set():
        logger.info("Audio upload task already running. Skipping this cycle.")
        return
    try:
        audio_task_flag.set()
        process_and_upload_audio()
    finally:
        audio_task_flag.clear()

def send_system_metrics_safe():
    """Send metrics safely, skipping if a previous run is still active."""
    if metrics_task_flag.is_set():
        logger.info("System metrics task already running. Skipping this cycle.")
        return
    try:
        metrics_task_flag.set()
        send_system_metrics()
    finally:
        metrics_task_flag.clear()

def schedule_uploads():
    """Start initial tasks and schedule recurring runs every minute."""
    logger.info("Starting the scheduled upload process.")
    executor.submit(process_and_upload_images_safe)
    executor.submit(process_and_upload_audio_safe)
    executor.submit(send_system_metrics_safe)
    schedule.every().minute.do(safe_run(process_and_upload_images_safe))
    schedule.every().minute.do(safe_run(process_and_upload_audio_safe))
    schedule.every().minute.do(safe_run(send_system_metrics_safe))
    while True:
        try:
            schedule.run_pending()
        except Exception as e:
            logger.error(f"Unexpected error in scheduler loop: {e}")
        time.sleep(1)

# Main Execution
if __name__ == "__main__":
    # Logging Configuration (attach handlers to ROOT so sensors.py logs go here)
    os.makedirs(os.path.dirname(logs_dir), exist_ok=True)
    file_handler = TimedRotatingFileHandler(logs_dir, when='D', interval=2, backupCount=2)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    root_logger.setLevel(logging.INFO)

    # Keep module logger clean (avoid duplicate logs)
    logger.propagate = True

    # Detect sensors once
    detect_sensors(bus, SENSOR_STATE)

    # --- Explicit startup logs for both sensors (now including BME688 T/RH) ---
    try:
        env = read_env(bus, SENSOR_STATE)
    except Exception as e:
        logger.error(f"Initial sensor read failed: {e}")
        env = {}

    # SHTC3 (temperature/humidity)
    if sensors.temp_humidity_sensor == "SHTC3":
        t_sht = env.get("t_c")
        h_sht = env.get("rh_pct")
        if t_sht is not None and h_sht is not None:
            logger.info(f"SHTC3 detected at 0x70 and ready (T={t_sht:.2f} °C, RH={h_sht:.2f}%).")
        else:
            logger.info("SHTC3 detected but returned no data yet (T/RH None).")
    else:
        logger.warning("SHTC3 not detected; SHTC3 temperature/humidity will be None.")

    # BME688 (temperature, humidity, pressure)
    bme_addr = SENSOR_STATE.get("bme_addr", 0x77)
    t_bme   = env.get("t_bme_c")
    h_bme   = env.get("rh_bme_pct")
    p_bme   = env.get("p_pa")

    if any(v is not None for v in (t_bme, h_bme, p_bme)):
        parts = []
        parts.append(f"T={t_bme:.2f} °C" if t_bme is not None else "T=None")
        parts.append(f"RH={h_bme:.2f}%"   if h_bme is not None else "RH=None")
        parts.append(f"P={p_bme:.2f} Pa"  if p_bme is not None else "P=None")
        logger.info(f"BME688 detected at 0x{bme_addr:02X} and ready ({', '.join(parts)}).")
    else:
        logger.warning(f"BME688 not detected or unreadable at 0x{bme_addr:02X}; temperature/humidity/pressure will be None.")

    # Kick off the scheduler loop
    schedule_uploads()
