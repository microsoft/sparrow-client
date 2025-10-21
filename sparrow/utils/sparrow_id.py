"""
Shared hardware ID helper. Reads a UUID string from a file (default: /host/etc/unique_id),
hashes it with SHA-256, and returns the first 12 hex chars.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

# Logger
logger = logging.getLogger(__name__)

# Configuration
UUID_DEFAULT_PATH = "/host/etc/unique_id"

# Hardware Key
def get_hardware_id(uuid_path: Optional[str] = None, env_var: str = "UNIQUE_ID_PATH") -> str:
    """
    Generate a 12-character hardware ID by hashing a stored UUID file.

    Args:
        uuid_path: Explicit path to the UUID file. If None, check env_var, then default.
        env_var:   Environment variable name to read the UUID path from.

    Returns:
        str: 12-character hex hardware id.
    """
    uuid_file = uuid_path or os.getenv(env_var, UUID_DEFAULT_PATH)
    p = Path(uuid_file)
    if not p.exists():
        raise FileNotFoundError(f"UUID file not found at {uuid_file}")

    uuid_str = p.read_text(encoding="utf-8").strip()
    if not uuid_str:
        raise ValueError("UUID is empty")

    # Do not log the raw UUID (sensitive)
    logger.debug("UUID read (len=%d) from %s", len(uuid_str), uuid_file)

    hardware_id = hashlib.sha256(uuid_str.encode("utf-8")).hexdigest()[:12]
    logger.info("Generated Hardware ID: %s", hardware_id)
    return hardware_id

# Main
if __name__ == "__main__":
    try:
        print(get_hardware_id())
    except Exception as e:
        print(f"[ERROR] {e}")
