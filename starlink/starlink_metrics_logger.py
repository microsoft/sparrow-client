"""
Polls a Starlink terminal over gRPC, gathers status + history + GPS location,
and posts them (along with the local sleep schedule) to the server on a loop.
"""

import os
import json
import time
import logging
from datetime import datetime, timezone
import requests
import hashlib
import grpc
import signal
from starlink_grpc import ChannelContext, status_data, history_bulk_data, get_location

# Configuration
CONFIG_PATH = "/app/config/schedule_config.json"
LOG_DIR = "/app/logs"
SERVER_BASE_URL = os.getenv("SERVER_BASE_URL", "https://server.sparrow-earth.com").rstrip("/")
API_ENDPOINT    = f"{SERVER_BASE_URL}/metrics"
with open("/app/config/access_key.txt", "r") as f:
    AUTH_KEY = f.read().strip()
LOG_FILE = os.path.join(LOG_DIR, "starlink_metrics.log")

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

# Global Context
# make context visible to the signal handler
context = None

# Hardware ID Retrieval
def get_hardware_id():
    """
    Retrieve a unique hardware ID based on a stored UUID.
    This ID uniquely identifies the device when communicating with the server.
    """
    try:
        # Retrieve the UUID path from environment variables
        uuid_path = os.getenv("UNIQUE_ID_PATH", "/host/etc/unique_id")
        logger.debug(f"UUID Path: {uuid_path}")

        # Check if the UUID file exists
        if not os.path.exists(uuid_path):
            raise FileNotFoundError(f"UUID file not found at {uuid_path}")

        # Read the UUID from the file
        with open(uuid_path, "r") as f:
            uuid_str = f.read().strip()
            logger.debug(f"Retrieved UUID: {uuid_str}")

        # Ensure the UUID is not empty
        if not uuid_str:
            raise ValueError("UUID is empty.")

        # Generate hardware ID by hashing the UUID using SHA-256 and truncating
        unique_id = hashlib.sha256(uuid_str.encode()).hexdigest()[:12]
        logger.info(f"Generated Hardware ID: {unique_id}")

        return unique_id

    except Exception as e:
        logger.error(f"Failed to retrieve hardware ID: {e}")
        raise

# Schedule Helpers
def load_schedule():
    """Load the sleep schedule from the configuration file."""
    try:
        with open(CONFIG_PATH, "r") as f:
            schedule = json.load(f)
            return schedule
    except Exception as e:
        logging.error(f"Failed to load schedule config: {e}")
        return None

def is_starlink_awake(schedule):
    """Determine if Starlink is awake based on the current time and schedule."""
    try:
        now = datetime.now(timezone.utc)
        start_time_str = schedule.get("start_time")
        end_time_str = schedule.get("end_time")
        if not start_time_str or not end_time_str:
            return True
        start_time = datetime.strptime(start_time_str, "%H:%M").time()
        end_time = datetime.strptime(end_time_str, "%H:%M").time()
        return not (start_time <= now.time() < end_time)
    except Exception as e:
        logging.error(f"Failed to determine Starlink's sleep status: {e}")
        return False

# Starlink gRPC Fetch
def fetch_starlink_metrics_and_location(context, max_retries=3, retry_delay=5):
    """
    Fetch metrics and location directly using the starlink_grpc module with retry logic.
    """
    for attempt in range(1, max_retries + 1):
        try:
            logging.info(f"Attempt {attempt}: Fetching Starlink metrics and location...")

            # Fetch metrics
            status, _, _ = status_data(context=context)
            general, bulk = history_bulk_data(context=context, parse_samples=True)

            # Fetch location
            location_data = get_location(context)
            latitude = location_data.lla.lat
            longitude = location_data.lla.lon

            location = {"latitude": latitude, "longitude": longitude}

            logging.info("Successfully fetched metrics and location.")
            return general, bulk, status, location

        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                logging.error(f"Attempt {attempt}: Deadline exceeded while fetching metrics or location.")
            else:
                logging.error(f"Attempt {attempt}: gRPC error occurred: {e}")
        except Exception as e:
            logging.error(f"Attempt {attempt}: Unexpected error while fetching metrics or location: {e}")

        if attempt < max_retries:
            logging.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

    logging.error(f"All {max_retries} attempts to fetch metrics and location failed.")
    return None, None, None, None

# Server Submission
def send_metrics_to_server(metrics, general, bulk, schedule, location, max_retries=3, retry_delay=5):
    """
    Send the metrics, sleep schedule, location, and unique ID to the REST API server with retry logic.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    unique_id = get_hardware_id()
    payload = {
        "unique_id": unique_id,
        "timestamp": timestamp,
        "general": general,
        "status": metrics,
        "history": bulk,
        "schedule": schedule,
        "location": location,
        "auth_key": AUTH_KEY
    }

    headers = {"Content-Type": "application/json"}

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(API_ENDPOINT, json=payload, headers=headers, timeout=10)
            if response.status_code == 200:
                logging.info("Metrics, schedule, location, and unique ID successfully sent to the server.")
                return
            else:
                logging.warning(f"Attempt {attempt}: Failed to send metrics. Status code {response.status_code}. Response: {response.text}")
        except requests.RequestException as e:
            logging.error(f"Attempt {attempt}: Error sending metrics to server: {e}")

        if attempt < max_retries:
            logging.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

    logging.error(f"All {max_retries} attempts to send metrics failed. Skipping to next cycle.")

# Signal Handling
def _shutdown(*_):
    logging.info("Received stop signal; closing gRPC context.")
    try:
        if context is not None:
            context.close()
    finally:
        raise SystemExit(0)

signal.signal(signal.SIGTERM, _shutdown)

# Main
def main():
    """Main loop for fetching and sending Starlink metrics, schedule, and location."""
    logging.info("Starting Starlink metrics logger...")

    global context
    context = ChannelContext()

    try:
        while True:
            try:
                schedule = load_schedule()
                if schedule and is_starlink_awake(schedule):
                    general, bulk, metrics, location = fetch_starlink_metrics_and_location(context)
                    if metrics and general and location:
                        send_metrics_to_server(metrics, general, bulk, schedule, location)
                    else:
                        logging.warning("No metrics, location, or schedule fetched; skipping server update.")
                else:
                    logging.info("Starlink is in sleep mode; skipping metrics collection.")
            except Exception as inner_e:
                logging.error(f"An error occurred during metrics collection or submission: {inner_e}", exc_info=True)

            time.sleep(20)
    except KeyboardInterrupt:
        logging.info("Shutting down metrics logger...")
    finally:
        context.close()
        logging.info("gRPC channel context closed.")

if __name__ == "__main__":
    main()
