#!/usr/bin/env python3
"""
Apply VE.Direct controller settings with retries; auto-create defaults on first run.
Watches /app/config/controller_settings.json and re-applies only when JSON changes.
Checks every 30 seconds.
"""

import os, sys, time, json, hashlib, logging
from filelock import FileLock, Timeout
import serial

# Setup Logging & Folders
LOG_DIR  = "/app/logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "controller_settings.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Configuration Paths
CONFIG_DIR  = "/app/config"
CONFIG_FILE = os.path.join(CONFIG_DIR, "controller_settings.json")
STATE_FILE  = os.path.join(CONFIG_DIR, "ve_settings.json")
RUN_LOCK    = "/tmp/veinit.lock"
SERIAL_LOCK = "/tmp/vedirect.lock"

os.makedirs(CONFIG_DIR, exist_ok=True)

# Default Configuration
DEFAULTS = {
    "PORT": "/dev/ttyUSB0",
    "BAUDRATE": 19200,
    "FLOAT_VOLTAGE": 26.8,
    "ABSORPTION_VOLTAGE": 27.2,
    "LOAD_HIGH_VOLTAGE": 24.08,
    "LOAD_LOW_VOLTAGE": 22.08,
    "LOAD_CONTROL_MODE": 5  # Victron USER1
}

# Config Helpers
def load_config():
    if not os.path.exists(CONFIG_FILE):
        save_config(DEFAULTS)
        return DEFAULTS.copy()
    try:
        with FileLock(f"{CONFIG_FILE}.lock", timeout=5):
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
        merged = DEFAULTS.copy()
        merged.update({k: data.get(k, DEFAULTS[k]) for k in DEFAULTS})
        return merged
    except Timeout:
        logger.warning("Timeout acquiring config lock; using defaults.")
        return DEFAULTS.copy()
    except Exception as e:
        logger.error(f"Failed to load config; using defaults. Error: {e}")
        return DEFAULTS.copy()

def save_config(cfg: dict):
    try:
        with FileLock(f"{CONFIG_FILE}.lock", timeout=5):
            tmp = CONFIG_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump(cfg, f, indent=2)
            os.replace(tmp, CONFIG_FILE)
        logger.info(f"Config saved to {CONFIG_FILE}")
    except Exception as e:
        logger.error(f"Failed to save config: {e}")

if not os.path.exists(CONFIG_FILE):
    save_config(DEFAULTS)

# VE.Direct Helpers
def build_frame(cmd_nibble, register_id, flags, payload_bytes):
    lo = register_id & 0xFF
    hi = (register_id >> 8) & 0xFF
    raw = [cmd_nibble, lo, hi, flags] + payload_bytes
    cs  = (0x55 - sum(raw)) & 0xFF
    body = f"{cmd_nibble:X}" + "".join(f"{b:02X}" for b in raw[1:]) + f"{cs:02X}"
    return ":" + body + "\r\n"

def validate_checksum(reply: str) -> bool:
    if not reply or not reply.startswith(":") or len(reply) < 5:
        return False
    body, cs_hex = reply[1:-2], reply[-2:]
    try:
        cs_recv = int(cs_hex, 16)
    except ValueError:
        return False
    raw = [int(body[0], 16)]
    i = 1
    while i < len(body):
        raw.append(int(body[i:i+2], 16))
        i += 2
    return ((0x55 - sum(raw)) & 0xFF) == cs_recv

def send_and_validate(ser, frame, timeout=1.0) -> bool:
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    ser.write(frame.encode("ascii"))
    deadline = time.time() + timeout
    target = frame.strip()
    while time.time() < deadline:
        line = ser.readline()
        if not line:
            continue
        text = line.decode("ascii", errors="ignore").strip()
        if text == target:
            ok = validate_checksum(text)
            if not ok:
                logger.warning("Checksum invalid for echo: %s", text)
            return ok
    return False

def volts_to_bytes(v: float):
    centi = int(round(float(v) / 0.01))
    return [centi & 0xFF, (centi >> 8) & 0xFF]

# Desired State & Signatures
def desired_config_signature(cfg: dict):
    payload = {
        "PORT": cfg["PORT"],
        "BAUDRATE": int(cfg["BAUDRATE"]),
        "FLOAT_VOLTAGE": float(cfg["FLOAT_VOLTAGE"]),
        "ABSORPTION_VOLTAGE": float(cfg["ABSORPTION_VOLTAGE"]),
        "LOAD_HIGH_VOLTAGE": float(cfg["LOAD_HIGH_VOLTAGE"]),
        "LOAD_LOW_VOLTAGE": float(cfg["LOAD_LOW_VOLTAGE"]),
        "LOAD_CONTROL_MODE": int(cfg["LOAD_CONTROL_MODE"]),
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest(), payload

def read_last_sig():
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f).get("hash")
    except Exception:
        return None

def write_last_sig(sig, payload):
    try:
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"hash": sig, "payload": payload, "ts": int(time.time())}, f, indent=2)
        os.replace(tmp, STATE_FILE)
        logger.info("State updated at %s", STATE_FILE)
    except Exception as e:
        logger.error(f"Failed to write state: {e}")

# Apply Settings (One Pass)
def apply_settings_once(ser, cfg: dict):
    stages = [
        ("Unlock User-Defined mode",   0xEDF1, [0xFF]),
        ("Set float voltage",          0xEDF6, volts_to_bytes(cfg["FLOAT_VOLTAGE"])),
        ("Set absorption voltage",     0xEDF7, volts_to_bytes(cfg["ABSORPTION_VOLTAGE"])),
        ("Set load control mode",      0xEDAB, [int(cfg["LOAD_CONTROL_MODE"])]),
        ("Set load switch high level", 0xED9D, volts_to_bytes(cfg["LOAD_HIGH_VOLTAGE"])),
        ("Set load switch low level", 0xED9C, volts_to_bytes(cfg["LOAD_LOW_VOLTAGE"])),
    ]
    for desc, reg, payload in stages:
        frame = build_frame(0x8, reg, 0x00, payload)
        if not send_and_validate(ser, frame, timeout=1.0):
            raise RuntimeError(f"Stage failed: {desc}")
        logger.info("Applied stage: %s", desc)

# Serial Open
def open_serial(cfg: dict):
    return serial.Serial(
        cfg["PORT"], int(cfg["BAUDRATE"]),
        bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        xonxoff=False, rtscts=False, dsrdtr=False,
        timeout=0.2, write_timeout=0.2
    )

# Ensure Applied (Controller)
def ensure_applied_now():
    cfg = load_config()
    desired_sig, payload = desired_config_signature(cfg)

    if read_last_sig() == desired_sig:
        logger.info("No change detected.")
        return

    logger.info("Change detected (or first run). Applying settings...")

    try:
        with FileLock(RUN_LOCK, timeout=30):
            if read_last_sig() == desired_sig:
                logger.info("Already applied (post-lock).")
                return

            backoff, attempts = 1.0, 0
            while True:
                attempts += 1
                try:
                    with FileLock(SERIAL_LOCK, timeout=10):
                        with open_serial(cfg) as ser:
                            time.sleep(0.2)
                            apply_settings_once(ser, cfg)
                    write_last_sig(desired_sig, payload)
                    logger.info("Applied successfully in %d attempt(s).", attempts)
                    return
                except Timeout:
                    logger.warning("Serial lock busy; retrying...")
                except serial.SerialException as e:
                    logger.warning("Serial error: %s; retrying...", e)
                except Exception as e:
                    logger.warning("Apply failed: %s; retrying...", e)

                time.sleep(min(backoff, 15))
                backoff = min(backoff * 1.7, 30)
    except Timeout:
        logger.warning("Run lock busy.")

# Main
def main():
    logger.info("Controller settings watcher starting.")
    while True:
        try:
            ensure_applied_now()
        except Exception as e:
            logger.error("Unexpected error in watch loop: %s", e, exc_info=True)
        time.sleep(30)

if __name__ == "__main__":
    sys.exit(main())
