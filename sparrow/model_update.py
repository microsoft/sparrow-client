
"""
Triton model sync service: periodically contacts SERVER_BASE_URL for a model manifest, mirrors the
declared repository into LOCAL_MODELS_DIR (atomic downloads, versioned folders), and reconciles the
running Triton server via the Repository API-unloading models that will change, loading/reloading
updated or missing models, and optionally unloading models no longer listed. Autoloads any on-disk models at startup.
"""

import os
import time
import requests
from requests.utils import requote_uri
import shutil
import logging

# Setup Logging & Folders
LOG_DIR = "/app/logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "model_update.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# Configuration
SERVER_BASE_URL = os.getenv("SERVER_BASE_URL", "https://server.sparrow-earth.com").rstrip("/")
SERVER_URL      = f"{SERVER_BASE_URL}/model_update"   # returns {"models":[...], "model_details":{...}}
BASE_FILE_URL   = SERVER_BASE_URL                      # to make relative file URLs absolute

TRITON_URL = (os.getenv("TRITON_SERVER_URL") or os.getenv("TRITON_URL", "http://triton:8000")).rstrip("/")
UNLOAD_REMOVED   = os.getenv("UNLOAD_REMOVED",   "true").strip().lower() in ("1", "true", "yes")
STARTUP_AUTOLOAD = os.getenv("STARTUP_AUTOLOAD", "true").strip().lower() in ("1", "true", "yes")

HTTP_TIMEOUT = 20
RETRY_SLEEP = 2.0

AUTH_KEY_PATH = "/app/config/access_key.txt"
try:
    with open(AUTH_KEY_PATH, "r") as f:
        AUTH_KEY = f.read().strip()
        log.info("Loaded AUTH_KEY")
except Exception as e:
    log.error(f"Failed to read auth key from {AUTH_KEY_PATH}: {e}")
    raise SystemExit(1)

LOCAL_MODELS_DIR = os.environ.get(
    "LOCAL_MODELS_DIR",
    os.path.join(os.path.expanduser("~"), "Desktop", "system", "Models", "tritonserver", "model_repository")
)
log.info(f"Local models directory: {LOCAL_MODELS_DIR}")
os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)

# Triton Helpers (Repository API)
def triton_ready(timeout_sec: int = 120) -> bool:
    """Wait until Triton is up. Use live + ready (strict=false) endpoints."""
    live_url  = f"{TRITON_URL}/v2/health/live"
    ready_url = f"{TRITON_URL}/v2/health/ready?strict=false"
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            rl = requests.get(live_url, timeout=HTTP_TIMEOUT)
            rr = requests.get(ready_url, timeout=HTTP_TIMEOUT)
            if rl.status_code == 200 and rr.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(1.0)
    return False

def repo_index() -> list:
    """Return Triton repository index: list of {name, version, state, ready}."""
    url = f"{TRITON_URL}/v2/repository/index"
    try:
        r = requests.post(url, json={}, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        log.warning(f"repository/index failed: {e}")
        return []

def loaded_model_names() -> set:
    """Set of models currently loaded (ready=True)."""
    idx = repo_index()
    names = {item.get("name") for item in idx if item.get("ready") is True}
    return {n for n in names if n}

def triton_load_model(name: str, retries: int = 3) -> bool:
    url = f"{TRITON_URL}/v2/repository/models/{name}/load"
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(url, timeout=HTTP_TIMEOUT)
            if r.status_code == 200:
                log.info(f"Loaded model: {name}")
                return True
            log.warning(f"Load {name} attempt {attempt} failed: {r.status_code} {r.text}")
        except requests.RequestException as e:
            log.warning(f"Load {name} attempt {attempt} exception: {e}")
        time.sleep(RETRY_SLEEP * attempt)
    return False

def triton_unload_model(name: str, retries: int = 3) -> bool:
    url = f"{TRITON_URL}/v2/repository/models/{name}/unload"
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(url, timeout=HTTP_TIMEOUT)
            if r.status_code == 200:
                log.info(f"Unloaded model: {name}")
                return True
            log.warning(f"Unload {name} attempt {attempt} failed: {r.status_code} {r.text}")
        except requests.RequestException as e:
            log.warning(f"Unload {name} attempt {attempt} exception: {e}")
        time.sleep(RETRY_SLEEP * attempt)
    return False

def autoload_existing_models():
    """Load any models present on disk (with config.pbtxt) that are not yet loaded."""
    if not triton_ready():
        log.error("Triton not ready; skipping autoload.")
        return
    current_loaded = loaded_model_names()
    for name in os.listdir(LOCAL_MODELS_DIR):
        mdir = os.path.join(LOCAL_MODELS_DIR, name)
        if not os.path.isdir(mdir):
            continue
        if not os.path.exists(os.path.join(mdir, "config.pbtxt")):
            continue
        if name in current_loaded:
            log.info(f"Model already loaded: {name}")
            continue
        triton_load_model(name)

# Download Helper
def robust_download_file(file_url: str, local_file_path: str):
    """Download to temp then atomic replace; cleanup on failure."""
    temp_file_path = local_file_path + ".tmp"
    try:
        with requests.get(file_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            with open(temp_file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        os.replace(temp_file_path, local_file_path)
        log.info(f"Downloaded {file_url} -> {local_file_path}")
    except Exception as e:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass
        raise e

# Server model_update Fetch
def get_model_update():
    payload = {"auth_key": AUTH_KEY}
    try:
        r = requests.post(SERVER_URL, json=payload, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        log.info("Model update data received.")
        return data
    except requests.RequestException as e:
        log.error(f"model_update request failed: {e}")
        return None

# Sync Logic (Unload/Reload)
def sync_models(model_data: dict):
    """
    Sync local repo with server spec and reconcile Triton load state (explicit mode).
    - Unloads a model if its files/versions will change, then reloads after.
    - Optionally unloads models that are no longer listed by the server.
    """
    if not triton_ready():
        log.error("Triton not ready; aborting sync.")
        return

    server_models = set(model_data.get("models", []))
    server_model_details = model_data.get("model_details", {})

    # Current loaded before changes
    currently_loaded = loaded_model_names()

    # Helper: plan detection so we unload only if needed
    def plan_changes_for_model(model: str):
        """
        Return a tuple (will_change: bool, plan: dict) describing what will be created/removed/downloaded.
        """
        local_model_dir = os.path.join(LOCAL_MODELS_DIR, model)
        plan = {
            "create_model_dir": not os.path.exists(local_model_dir),
            "remove_versions": set(),
            "create_versions": set(),
            "per_version": {}  # version -> dict(remove_files, download_files)
        }

        # Numeric subfolders only
        server_subs = {
            sub: files
            for sub, files in server_model_details.get(model, {}).items()
            if sub.isdigit()
        }
        local_subs = set()
        if os.path.isdir(local_model_dir):
            local_subs = {
                d for d in os.listdir(local_model_dir)
                if os.path.isdir(os.path.join(local_model_dir, d)) and d.isdigit()
            }

        # Identify creates/removals of version folders
        server_sub_names = set(server_subs.keys())
        plan["create_versions"] = server_sub_names - local_subs
        plan["remove_versions"] = local_subs - server_sub_names

        # For shared versions, compute file diffs
        for ver in (server_sub_names & local_subs):
            local_vdir = os.path.join(local_model_dir, ver)
            local_files = set(os.listdir(local_vdir)) if os.path.isdir(local_vdir) else set()
            server_file_names = {
                f.get("file_name", "").replace("@SynoEAStream", "")
                for f in server_subs.get(ver, [])
                if f.get("file_name") and "Zone.Identifier" not in f.get("file_name", "")
            }
            remove_files = local_files - server_file_names
            download_files = server_file_names - local_files
            plan["per_version"][ver] = {
                "remove_files": remove_files,
                "download_files": download_files,
                "server_files_info": server_subs.get(ver, []),
            }

        will_change = (
            plan["create_model_dir"]
            or bool(plan["create_versions"])
            or bool(plan["remove_versions"])
            or any(d["remove_files"] or d["download_files"] for d in plan["per_version"].values())
        )
        return will_change, plan

    # Apply plan per model
    for model in server_models:
        local_model_dir = os.path.join(LOCAL_MODELS_DIR, model)
        will_change, plan = plan_changes_for_model(model)

        # Unload if loaded AND we need to change files/versions
        was_loaded = model in currently_loaded
        if will_change and was_loaded:
            log.info(f"Model '{model}' will change; unloading before update.")
            triton_unload_model(model)

        # Ensure model dir
        if plan["create_model_dir"]:
            os.makedirs(local_model_dir, exist_ok=True)
            log.info(f"Created model dir: {local_model_dir}")

        # Remove extra local versions
        for ver in plan["remove_versions"]:
            path = os.path.join(local_model_dir, ver)
            log.info(f"Removing extra local version: {path}")
            shutil.rmtree(path, ignore_errors=True)

        # Ensure server versions exist
        for ver in plan["create_versions"]:
            vdir = os.path.join(local_model_dir, ver)
            os.makedirs(vdir, exist_ok=True)
            log.info(f"Created version dir: {vdir}")

        # For shared versions, remove extra files + download missing
        for ver, vplan in plan["per_version"].items():
            vdir = os.path.join(local_model_dir, ver)

            # Remove extra files
            for fname in vplan["remove_files"]:
                fpath = os.path.join(vdir, fname)
                try:
                    os.remove(fpath)
                    log.info(f"Removed extra file: {fpath}")
                except FileNotFoundError:
                    pass
                except Exception as e:
                    log.warning(f"Failed to remove {fpath}: {e}")

            # Download missing files
            for file_info in vplan["server_files_info"]:
                orig = file_info.get("file_name")
                if not orig or "Zone.Identifier" in orig:
                    continue
                fname = orig.replace("@SynoEAStream", "")
                if fname not in vplan["download_files"]:
                    continue
                furl = file_info.get("url", "")
                if not furl.startswith("http"):
                    furl = BASE_FILE_URL + furl
                furl = requote_uri(furl)
                fpath = os.path.join(vdir, fname)
                try:
                    log.info(f"Downloading {fname} -> {fpath}")
                    robust_download_file(furl, fpath)
                except Exception as e:
                    log.error(f"Failed to download {fname}: {e}")

        # Ensure root config.pbtxt present
        cfg_path = os.path.join(local_model_dir, "config.pbtxt")
        if not os.path.exists(cfg_path):
            cfg_url = requote_uri(f"{BASE_FILE_URL}/models/{model}/config.pbtxt")
            try:
                log.info(f"Downloading config.pbtxt -> {cfg_path}")
                robust_download_file(cfg_url, cfg_path)
            except Exception as e:
                log.error(f"Failed to download config.pbtxt for {model}: {e}")

        # Reload if it was loaded before changes, or simply ensure it's loaded
        # (explicit mode requires an explicit load call)
        if will_change or model not in currently_loaded:
            triton_load_model(model)

    # Optionally unload models no longer on the server list
    if UNLOAD_REMOVED:
        still_loaded = loaded_model_names()
        to_unload = still_loaded - server_models
        for model in to_unload:
            log.info(f"Unloading model no longer listed by server: {model}")
            triton_unload_model(model)

# Main Loop
def main_loop():
    # Wait for Triton at startup
    if not triton_ready():
        log.error("Triton did not become ready in time.")
    else:
        log.info("Triton is ready.")

    # Autoload any on-disk models once (explicit mode)
    if STARTUP_AUTOLOAD:
        autoload_existing_models()

    while True:
        log.info("Checking server for model updates...")
        data = get_model_update()
        if data:
            try:
                sync_models(data)
            except Exception as e:
                log.error(f"Error during sync: {e}", exc_info=True)
        else:
            log.info("Server unreachable or empty response.")
        log.info("Sleeping 60s...")
        time.sleep(60)

if __name__ == "__main__":
    main_loop()
