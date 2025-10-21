#!/usr/bin/env python3
"""
Manages Starlink power on a Jetson using a GPIO relay: it fetches a sleep/wake
window from the server, reads battery voltage via VE.Direct (/dev/ttyUSB0),
and toggles power accordingly. It avoids cutting power during Starlink updates
(using gRPC status + optional reboot to install).
"""

import os
import json
import time
import logging
import hashlib
import requests
from datetime import datetime, timezone
import threading
import Jetson.GPIO as GPIO
import serial
import sys
from filelock import FileLock, Timeout
from starlink_grpc import ChannelContext, status_data

# VE.Direct / Paths
VE_DIRECT_PORT = "/dev/ttyUSB0"

CONFIG_PATH = "/app/config/schedule_config.json"
LOG_DIR = "/app/logs"
LOG_FILE = os.path.join(LOG_DIR, "starlink_schedule.log")

# Setup Logging & Folders
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# REST API Endpoint & Auth
SERVER_BASE_URL = os.getenv("SERVER_BASE_URL", "https://server.sparrow-earth.com").rstrip("/")
REST_API_URL    = f"{SERVER_BASE_URL}/get_schedule"
logger.info(f"Schedule endpoint: {REST_API_URL}")

try:
    with open("/app/config/access_key.txt", "r") as f:
        AUTH_KEY = f.read().strip()
except Exception as e:
    logging.error(f"Failed to read access key from /app/config/access_key.txt: {e}")
    exit(1)

# GPIO Relay Setup
PIN_RELAY = 8   # BCM 8 > mikroBUS CS > Relay 2 coil & LED (socket 1)
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN_RELAY, GPIO.OUT, initial=GPIO.LOW)

# Hardware ID Retrieval
def get_hardware_id():
    """
    Retrieve a unique hardware ID based on a stored UUID.
    This ID uniquely identifies the device when communicating with the server.
    Returns:
        str: A 12-character hexadecimal string representing the hardware ID.
    Raises:
        Exception: If the UUID file is not found or an error occurs during retrieval.
    """
    try:
        uuid_path = os.getenv("UNIQUE_ID_PATH", "/host/etc/unique_id")
        logger.debug(f"UUID Path: {uuid_path}")
        if not os.path.exists(uuid_path):
            raise FileNotFoundError(f"UUID file not found at {uuid_path}")
        with open(uuid_path, "r") as f:
            uuid_str = f.read().strip()
            logger.debug(f"Retrieved UUID: {uuid_str}")
        if not uuid_str:
            raise ValueError("UUID is empty.")
        unique_id = hashlib.sha256(uuid_str.encode()).hexdigest()[:12]
        logger.info(f"Generated Hardware ID: {unique_id}")
        return unique_id
    except Exception as e:
        logger.error(f"Failed to retrieve hardware ID: {e}")
        raise

# VE.Direct Helpers
def read_vedirect_battery_voltage(port=VE_DIRECT_PORT):
    """
    Open VE.Direct at 19 200 baud, look for the 'V' (battery voltage in mV) register,
    scale it to volts, and return. Returns None on any failure.
    """
    try:
        with serial.Serial(port, 19200, timeout=1) as ser:
            logger.info(f"Opened VE.Direct on {port} for battery voltage fallback")

            start = time.time()
            while True:
                raw = ser.readline()
                if not raw or time.time() - start > 2:
                    break
                line = raw.decode("ascii", "ignore").strip()
                if line.startswith("V\t") or line.startswith("V :") or line.startswith("V "):
                    parts = line.replace(":", "\t").split("\t", 1)
                    try:
                        mv = float(parts[1])
                        volts = mv / 1000.0
                        logger.info(f"VE.Direct battery voltage read: {volts:.2f} V")
                        return volts
                    except Exception as e:
                        logger.error(f"Failed parsing VE.Direct V value: {e}")
                        return None

        logger.warning("VE.Direct battery voltage not found in frame")
        return None
    except Exception as e:
        logger.error(f"VE.Direct port open error: {e}")
        return None

# Schedule Load/Save
def load_local_schedule():
    """Load the local sleep schedule from the configuration file."""
    try:
        with open(CONFIG_PATH, "r") as f:
            logger.info("Loading local sleep schedule configuration.")
            return json.load(f)
    except FileNotFoundError:
        logger.warning("Local schedule_config.json not found. Creating a new one.")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in local schedule_config.json: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load local schedule config: {e}")
        return None

def save_local_schedule(schedule):
    """Save the sleep schedule to the local configuration file."""
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(schedule, f, indent=4)
        logger.info("Local schedule_config.json updated successfully.")
    except Exception as e:
        logger.error(f"Failed to save local schedule config: {e}")

def convert_to_utc_minutes(local_time_str):
    """Convert local time (HH:MM) to UTC minutes past midnight."""
    try:
        local_time = datetime.strptime(local_time_str, "%H:%M")
        local_offset = datetime.now().astimezone().utcoffset()
        utc_time = local_time - local_offset
        utc_time = utc_time.replace(year=1900, month=1, day=1)
        utc_minutes = utc_time.hour * 60 + utc_time.minute
        return utc_minutes % 1440
    except ValueError as e:
        logger.error(f"Invalid time format '{local_time_str}': {e}")
        return None

# Relay Control (GPIO only)
def turn_on_starlink():
    """Power ON Starlink via GPIO relay."""
    GPIO.output(PIN_RELAY, GPIO.LOW)
    logger.info("Starlink turned ON (GPIO)")

def turn_off_starlink():
    """Power OFF Starlink via GPIO relay."""
    GPIO.output(PIN_RELAY, GPIO.HIGH)
    logger.info("Starlink turned OFF (GPIO)")

# Update Status / Control
def is_update_in_progress():
    """
    Check if a Starlink software update is in progress using the starlink_grpc package.
    Returns:
        bool: True if an update is in progress, False otherwise.
    """
    try:
        channel = ChannelContext()
        status = status_data(channel)
        logger.info(f"Starlink status: {status}")
        if isinstance(status, tuple) and len(status) >= 3:
            update_flag = status[2].get("alert_install_pending", False)
        else:
            update_flag = False
        logger.info(f"Starlink update status (alert_install_pending): {update_flag}")
        return update_flag
    except Exception as e:
        logger.error(f"Error checking update status: {e}")
        return False

def trigger_update_installation():
    """
    Trigger the Starlink update installation via GRPC.
    Returns:
        bool: True if the update installation was successfully triggered, False otherwise.
    """
    try:
        channel = ChannelContext()
        from starlink_grpc import reboot
        reboot(channel)
        logger.info("Triggered update installation via reboot command.")
        return True
    except Exception as e:
        logger.error(f"Failed to trigger update installation: {e}")
        return False

def wait_for_update_to_clear(poll_interval=30, max_retries=10):
    """
    Poll every poll_interval seconds until no update is pending.
    """
    retries = 0
    while retries < max_retries:
        if not is_update_in_progress():
            logger.info("Update flag cleared.")
            return True
        logger.info("Update still pending. Waiting...")
        time.sleep(poll_interval)
        retries += 1
    logger.warning("Max retries reached; proceeding anyway.")
    return True

def wait_for_update_to_complete(poll_interval=300):
    """
    Poll every poll_interval seconds until no update is in progress.
    """
    while is_update_in_progress():
        logger.info("Update in progress; waiting 5 minutes.")
        time.sleep(poll_interval)
    logger.info("No update in progress.")

def log_update_status_periodically(poll_interval=300):
    """
    Periodically check and log the update status every poll_interval seconds.
    Runs in a separate thread.
    """
    while True:
        status = is_update_in_progress()
        logger.info(f"(Periodic) Starlink update status: {status}")
        time.sleep(poll_interval)

# Schedule Application
def apply_schedule(schedule):
    """
    Apply the sleep/wake schedule using GPIO relay control,
    reading battery voltage from the solar controller via VE.Direct.
    """
    try:
        start_time = schedule.get("start_time")
        end_time = schedule.get("end_time")
        if start_time and end_time:
            start_minutes = convert_to_utc_minutes(start_time)
            end_minutes = convert_to_utc_minutes(end_time)
            now = datetime.now(timezone.utc)
            current_minutes = now.hour * 60 + now.minute
            if start_minutes < end_minutes:
                in_sleep = (current_minutes >= start_minutes and current_minutes < end_minutes)
            else:
                in_sleep = (current_minutes >= start_minutes or current_minutes < end_minutes)

            if in_sleep:
                logger.info("Within sleep window: preparing to cut power.")
                if is_update_in_progress():
                    logger.info("Update pending; triggering installation.")
                    if trigger_update_installation():
                        logger.info("Waiting 15 minutes for update to initiate.")
                        time.sleep(900)
                    wait_for_update_to_clear()
                logger.info("Cutting power to Starlink.")
                turn_off_starlink()
            else:
                logger.info("Outside sleep window: powering ON Starlink if voltage OK.")

                battery_voltage = read_vedirect_battery_voltage()
                if battery_voltage is not None:
                    logger.info(f"Using VE.Direct battery voltage: {battery_voltage:.2f} V")
                else:
                    logger.error("No battery voltage available from VE.Direct")

                if battery_voltage is None or battery_voltage >= 20.0:
                    logger.info(
                        f"Battery {battery_voltage if battery_voltage is not None else 'unknown'} V → powering ON."
                    )
                    turn_on_starlink()
                else:
                    logger.warning(f"Battery {battery_voltage:.2f} V < 20.0 V; keeping OFF.")
        else:
            logger.warning("Start or end time missing in schedule.")
    except Exception as e:
        logger.error(f"Error applying schedule: {e}")

# Remote Schedule Fetch
def fetch_remote_schedule(unique_id):
    """Fetch the schedule from the REST API."""
    try:
        payload = {"unique_id": unique_id, "auth_key": AUTH_KEY}
        headers = {"Content-Type": "application/json"}
        logger.info(f"Fetching remote schedule for {unique_id}")
        response = requests.post(REST_API_URL, json=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            logger.error("Unauthorized access.")
            return None
        elif response.status_code == 404:
            logger.error("Schedule not found.")
            return None
        else:
            logger.error(f"Error {response.status_code}: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"REST API error: {e}")
        return None

def schedules_are_different(local_schedule, remote_schedule):
    """Compare local and remote schedules."""
    if not local_schedule and remote_schedule:
        return True
    if not remote_schedule:
        return False
    return (local_schedule.get("start_time") != remote_schedule.get("start_time") or
            local_schedule.get("end_time")   != remote_schedule.get("end_time"))

# Main
def main():
    logger.info("Starting Starlink sleep schedule manager...")
    unique_id = get_hardware_id()
    last_applied_schedule = load_local_schedule()
    update_logger_thread = threading.Thread(target=log_update_status_periodically, args=(300,), daemon=True)
    update_logger_thread.start()
    try:
        while True:
            remote_schedule = fetch_remote_schedule(unique_id)
            if remote_schedule and schedules_are_different(last_applied_schedule, remote_schedule):
                save_local_schedule(remote_schedule)
                last_applied_schedule = remote_schedule
            if remote_schedule:
                apply_schedule(remote_schedule)
            elif last_applied_schedule:
                apply_schedule(last_applied_schedule)
            else:
                logger.warning("No schedule available.")
            time.sleep(120)
    finally:
        GPIO.cleanup()
        logger.info("GPIO cleaned up; exiting.")

# Entry Point (Single-Instance)
if __name__ == "__main__":
    LOCK_PATH = "/tmp/starlink_schedule.lock"
    try:
        with FileLock(LOCK_PATH, timeout=10):
            main()
    except Timeout:
        logger.warning("Another starlink_schedule instance is already running; exiting.")
        sys.exit(0)
