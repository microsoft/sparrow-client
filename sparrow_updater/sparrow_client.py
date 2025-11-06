#!/usr/bin/env python3

"""
Sparrow Updater: staged, checksum-verified updates for a Docker Compose app.

- Polls a remote HTTPS manifest, downloads only changed files, and verifies SHA256.
- Stages updates to sparrow_releases/<release_id>, backs up changed files, and swaps atomically.
- Restarts docker compose (v1 or v2 auto-detected). No container health checks.
- Automatic rollback if the restart itself fails. No deletion of files not in the manifest.
- Config is hot-reloaded from sparrow_updater.env and/or environment variables.
"""

import os
import sys
import hashlib
import shutil
import subprocess
import time
import json
import logging
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from github import Github
from github.GithubException import GithubException
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Paths
SYSTEM_ROOT = Path(os.path.expanduser(os.getenv("SYSTEM_ROOT", "~/Desktop/system")))
LIVE_ROOT = SYSTEM_ROOT

DOCKER_COMPOSE_YML = str(LIVE_ROOT / "docker-compose.yml")

# Access key
ACCESS_KEY_PATH = str(LIVE_ROOT / "sparrow" / "config" / "access_key.txt")

RELEASES_DIR = LIVE_ROOT / "sparrow_releases"  # staging
BACKUPS_DIR  = LIVE_ROOT / "sparrow_backups"   # rollback
RELEASES_DIR.mkdir(parents=True, exist_ok=True)
BACKUPS_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = Path(os.path.expanduser("~/.cache/sparrow_updater"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CURRENT_VERSION_PATH = CACHE_DIR / "current_version.txt"
BUNDLE_CACHE_DIR = CACHE_DIR / "bundles"
BUNDLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Optional config file that we hot-reload
ENV_FILE_PATH = LIVE_ROOT / "sparrow_updater.env"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.expanduser("~/sparrow_update_agent.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("update_agent")

# DEFAULT ENV (auto-create if missing)
DEFAULT_ENV_CONTENT = """# Sparrow Updater defaults (auto-generated)
# You can edit this file; the updater hot-reloads it each cycle.

# Where the client reports update results
CLIENT_UPDATE_URL=https://server.sparrow-earth.com/client_update

# Updater behavior
DISABLE_UPDATES=false
POLL_INTERVAL_SECONDS=600

# Project root (set to /system for your layout)
SYSTEM_ROOT=/system

# GitHub release configuration
GITHUB_REPOSITORY=microsoft/sparrow-client
RELEASE_BUNDLE_NAME=sparrow-client-update.tar.gz
# Optional PAT for private repos or higher rate limits
GITHUB_TOKEN=
MAX_BUNDLE_SIZE_BYTES=209715200
"""

def ensure_default_env():
    try:
        if not ENV_FILE_PATH.exists():
            ENV_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            ENV_FILE_PATH.write_text(DEFAULT_ENV_CONTENT, encoding="utf-8")
            os.chmod(ENV_FILE_PATH, 0o644)
            logger.info("Created default env at %s", ENV_FILE_PATH)
    except Exception as e:
        logger.warning("Could not create default env at %s: %s", ENV_FILE_PATH, e)

# ---- tiny helper: strip only a *leading* "./" literally (do NOT drop leading dots)
def _strip_dot_slash(s: str) -> str:
    return s[2:] if s.startswith("./") else s

# HTTP SESSION (retries/backoff)
def build_session() -> requests.Session:
    common = dict(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        respect_retry_after_header=True,
    )
    try:
        retry = Retry(allowed_methods=frozenset({"GET", "POST"}), raise_on_status=False, **common)
    except TypeError:
        common.pop("respect_retry_after_header", None)
        retry = Retry(method_whitelist=frozenset({"GET", "POST"}), **common)

    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=20)
    sess = requests.Session()
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    sess.headers.update({"User-Agent": "sparrow-updater/2.0"})
    return sess

SESSION = build_session()

# CONFIG (hot-reloaded)
class Config:
    CLIENT_UPDATE_URL: str = "https://server.sparrow-earth.com/client_update"

    DISABLE_UPDATES: bool = False
    POLL_INTERVAL_SECONDS: int = 600
    CONNECT_TIMEOUT_SECONDS: int = 10
    READ_TIMEOUT_SECONDS: int = 60
    MAX_BUNDLE_SIZE_BYTES: int = 200 * 1024 * 1024
    GITHUB_REPOSITORY: str = "microsoft/sparrow-client"
    RELEASE_BUNDLE_NAME: str = "sparrow-client-update.tar.gz"
    GITHUB_TOKEN: Optional[str] = None

    _env_mtime: Optional[float] = None

    def _parse_bool(self, v: Optional[str], default=False) -> bool:
        if v is None:
            return default
        return v.strip().lower() in ("1", "true", "yes", "on")

    def _from_env_vars(self):
        self.CLIENT_UPDATE_URL = os.getenv("CLIENT_UPDATE_URL", self.CLIENT_UPDATE_URL)

        self.DISABLE_UPDATES = self._parse_bool(os.getenv("DISABLE_UPDATES"), self.DISABLE_UPDATES)
        self.POLL_INTERVAL_SECONDS   = int(os.getenv("POLL_INTERVAL_SECONDS", self.POLL_INTERVAL_SECONDS))
        self.CONNECT_TIMEOUT_SECONDS = int(os.getenv("CONNECT_TIMEOUT_SECONDS", self.CONNECT_TIMEOUT_SECONDS))
        self.READ_TIMEOUT_SECONDS    = int(os.getenv("READ_TIMEOUT_SECONDS", self.READ_TIMEOUT_SECONDS))
        self.MAX_BUNDLE_SIZE_BYTES   = int(os.getenv("MAX_BUNDLE_SIZE_BYTES", self.MAX_BUNDLE_SIZE_BYTES))
        self.GITHUB_REPOSITORY       = os.getenv("GITHUB_REPOSITORY", self.GITHUB_REPOSITORY)
        self.RELEASE_BUNDLE_NAME     = os.getenv("RELEASE_BUNDLE_NAME", self.RELEASE_BUNDLE_NAME)
        token_override = os.getenv("GITHUB_TOKEN")
        if token_override is not None:
            self.GITHUB_TOKEN = token_override

    def _from_env_file(self):
        if not ENV_FILE_PATH.exists():
            return
        try:
            text = ENV_FILE_PATH.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to read %s: %s", ENV_FILE_PATH, e)
            return

        kv: Dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()

        self.CLIENT_UPDATE_URL = kv.get("CLIENT_UPDATE_URL", self.CLIENT_UPDATE_URL)

        if "DISABLE_UPDATES" in kv:
            self.DISABLE_UPDATES = self._parse_bool(kv["DISABLE_UPDATES"], self.DISABLE_UPDATES)

        self.POLL_INTERVAL_SECONDS   = int(kv.get("POLL_INTERVAL_SECONDS", self.POLL_INTERVAL_SECONDS))
        self.CONNECT_TIMEOUT_SECONDS = int(kv.get("CONNECT_TIMEOUT_SECONDS", self.CONNECT_TIMEOUT_SECONDS))
        self.READ_TIMEOUT_SECONDS    = int(kv.get("READ_TIMEOUT_SECONDS", self.READ_TIMEOUT_SECONDS))
        self.MAX_BUNDLE_SIZE_BYTES   = int(kv.get("MAX_BUNDLE_SIZE_BYTES", self.MAX_BUNDLE_SIZE_BYTES))
        self.GITHUB_REPOSITORY       = kv.get("GITHUB_REPOSITORY", self.GITHUB_REPOSITORY)
        self.RELEASE_BUNDLE_NAME     = kv.get("RELEASE_BUNDLE_NAME", self.RELEASE_BUNDLE_NAME)
        if "GITHUB_TOKEN" in kv:
            self.GITHUB_TOKEN = kv["GITHUB_TOKEN"]

    def _require_https(self, url: str, name: str):
        if not url.lower().startswith("https://"):
            raise RuntimeError(f"{name} must be HTTPS: {url}")

    def validate(self):
        self._require_https(self.CLIENT_UPDATE_URL, "CLIENT_UPDATE_URL")
        if "/" not in self.GITHUB_REPOSITORY:
            raise RuntimeError("GITHUB_REPOSITORY must be in 'owner/name' format")

    def load(self, first_load=False) -> bool:
        before = self.to_dict()
        try:
            mtime = ENV_FILE_PATH.stat().st_mtime if ENV_FILE_PATH.exists() else None
        except Exception:
            mtime = None

        if first_load or (mtime != self._env_mtime):
            if not first_load:
                logger.info("Config file change detected; reloading settings from %s", ENV_FILE_PATH)
            self._from_env_file()
            self._env_mtime = mtime

        self._from_env_vars()
        self.TOTAL_TIMEOUT = (self.CONNECT_TIMEOUT_SECONDS, self.READ_TIMEOUT_SECONDS)
        self.validate()

        after = self.to_dict()
        changed = (before != after)
        if changed and not first_load:
            logger.info("Active settings updated: %s", json.dumps(after))
        return changed

    def to_dict(self) -> dict:
        return {
            "CLIENT_UPDATE_URL": self.CLIENT_UPDATE_URL,
            "DISABLE_UPDATES": self.DISABLE_UPDATES,
            "POLL_INTERVAL_SECONDS": self.POLL_INTERVAL_SECONDS,
            "CONNECT_TIMEOUT_SECONDS": self.CONNECT_TIMEOUT_SECONDS,
            "READ_TIMEOUT_SECONDS": self.READ_TIMEOUT_SECONDS,
            "MAX_BUNDLE_SIZE_BYTES": self.MAX_BUNDLE_SIZE_BYTES,
            "GITHUB_REPOSITORY": self.GITHUB_REPOSITORY,
            "RELEASE_BUNDLE_NAME": self.RELEASE_BUNDLE_NAME,
            "GITHUB_TOKEN_SET": bool(self.GITHUB_TOKEN),
        }

ensure_default_env()
CFG = Config()
CFG.load(first_load=True)

# UTILITIES
def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def _safe_join(base: Path, rel: str) -> Path:
    rel = _strip_dot_slash(rel)
    p = (base / rel).resolve()
    if not str(p).startswith(str(base.resolve())):
        raise ValueError(f"Unsafe path: {rel}")
    return p

def bytes_free(path: Path) -> int:
    s = os.statvfs(str(path))
    return s.f_bavail * s.f_frsize

def ensure_space(required_bytes: int, base: Path, buffer_bytes: int = 200*1024*1024):
    free = bytes_free(base)
    if required_bytes and free < required_bytes + buffer_bytes:
        raise RuntimeError(f"Not enough disk space: need {required_bytes + buffer_bytes}B, have {free}B")

# FETCH & DOWNLOAD ---------------------------------------------------------
def _bounded_stream_to_file(resp: requests.Response, dest_path: Path, max_bytes: Optional[int]) -> None:
    content_length = resp.headers.get("Content-Length")
    if max_bytes is not None and content_length is not None:
        try:
            clen = int(content_length)
            if clen > max_bytes:
                raise ValueError(f"Remote file too large: {clen} > {max_bytes}")
        except ValueError:
            pass

    bytes_written = 0
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if not chunk:
                continue
            bytes_written += len(chunk)
            if max_bytes is not None and bytes_written > max_bytes:
                raise ValueError(f"Download exceeded cap: {bytes_written} > {max_bytes}")
            f.write(chunk)
        f.flush()
        os.fsync(f.fileno())

def _copy_bundle_file(bundle_root: Path, relpath: str, target_path: Path) -> None:
    source = _safe_join(bundle_root, relpath)
    if not source.exists():
        raise FileNotFoundError(f"Bundle missing file {relpath}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target_path)
    with open(target_path, "rb+") as handle:
        handle.flush()
        os.fsync(handle.fileno())

# Release discovery helpers -------------------------------------------------
def normalize_version_tag(tag: Optional[str]) -> str:
    if not tag:
        return "0.0.0"
    value = str(tag).strip()
    if value.lower().startswith("refs/tags/"):
        value = value[10:]
    if value.lower().startswith("release/"):
        value = value.split("/", 1)[1]
    if value.startswith("v") or value.startswith("V"):
        value = value[1:]
    return value or "0.0.0"

def _version_key(value: str) -> Tuple[Tuple[int, ...], int, str]:
    normalized = normalize_version_tag(value)
    base, _, suffix = normalized.partition("-")
    parts: List[int] = []
    for part in base.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    suffix = suffix.lower()
    suffix_flag = 1 if not suffix else 0
    return tuple(parts), suffix_flag, suffix

def is_remote_version_newer(remote: str, local: str) -> bool:
    return _version_key(remote) > _version_key(local)

def read_current_version() -> str:
    try:
        value = CURRENT_VERSION_PATH.read_text(encoding="utf-8").strip()
        return value or "0.0.0"
    except FileNotFoundError:
        return "0.0.0"
    except Exception as exc:
        logger.warning("Failed to read %s: %s", CURRENT_VERSION_PATH, exc)
        return "0.0.0"

def write_current_version(version: str) -> None:
    try:
        CURRENT_VERSION_PATH.write_text(version.strip(), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to persist current version to %s: %s", CURRENT_VERSION_PATH, exc)

def _build_github_client() -> Github:
    token = (CFG.GITHUB_TOKEN or "").strip()
    if token:
        return Github(token, per_page=100)
    return Github(per_page=100)

def _latest_published_release():
    try:
        client = _build_github_client()
        repo = client.get_repo(CFG.GITHUB_REPOSITORY)
        for release in repo.get_releases():
            if not release.draft:
                return release
        logger.info("No published releases found for %s", CFG.GITHUB_REPOSITORY)
    except GithubException as exc:
        logger.error("GitHub API error while fetching releases: %s", exc)
    except Exception:
        logger.exception("Unexpected error while retrieving GitHub release metadata.")
    return None

def _find_bundle_asset(release) -> Optional[object]:
    try:
        for asset in release.get_assets():
            if asset.name == CFG.RELEASE_BUNDLE_NAME:
                return asset
    except GithubException as exc:
        logger.error("GitHub API error while listing assets: %s", exc)
    return None

def _safe_extract_tar(tar_obj: tarfile.TarFile, target_dir: Path) -> None:
    base = target_dir.resolve()
    for member in tar_obj.getmembers():
        member_path = (base / member.name).resolve()
        if not str(member_path).startswith(str(base)):
            raise RuntimeError(f"Unsafe path in archive: {member.name}")
    tar_obj.extractall(path=str(base))

def _download_release_bundle(asset, version: str) -> Path:
    sanitized_version = normalize_version_tag(version).replace("/", "-")
    bundle_path = BUNDLE_CACHE_DIR / f"{sanitized_version}-{asset.name}"
    headers = {"Accept": "application/octet-stream"}
    token = (CFG.GITHUB_TOKEN or "").strip()
    if token:
        headers["Authorization"] = f"token {token}"

    if bundle_path.exists():
        bundle_path.unlink()

    logger.info("Downloading release bundle %s for version %s", asset.name, version)
    with SESSION.get(asset.browser_download_url, stream=True, headers=headers, timeout=CFG.TOTAL_TIMEOUT) as resp:
        resp.raise_for_status()
        _bounded_stream_to_file(resp, bundle_path, CFG.MAX_BUNDLE_SIZE_BYTES)
    return bundle_path

def _load_manifest_from_release(release, version: str) -> Tuple[dict, Path]:
    asset = _find_bundle_asset(release)
    if asset is None:
        raise RuntimeError(
            f"Release {release.tag_name} is missing required asset {CFG.RELEASE_BUNDLE_NAME}"
        )

    sanitized_version = normalize_version_tag(version).replace("/", "-") or "0.0.0"
    extract_dir = BUNDLE_CACHE_DIR / sanitized_version
    manifest_path = extract_dir / "manifest.json"

    if not manifest_path.exists():
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = _download_release_bundle(asset, version)
        with tarfile.open(bundle_path, "r:gz") as tar_obj:
            _safe_extract_tar(tar_obj, extract_dir)

    if not manifest_path.exists():
        raise RuntimeError("Manifest not found in release bundle after extraction.")

    try:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse manifest from release bundle: {exc}") from exc
    return manifest_data, extract_dir

def fetch_release_manifest(local_version: str) -> Optional[Dict[str, Any]]:
    release = _latest_published_release()
    if release is None:
        return None

    remote_version = normalize_version_tag(release.tag_name or release.name or release.title)
    if not remote_version:
        remote_version = "0.0.0"

    if not is_remote_version_newer(remote_version, local_version):
        logger.info(
            "Client already at latest release (local=%s, remote=%s)",
            local_version,
            remote_version,
        )
        return {"version": remote_version, "manifest": None}

    try:
        manifest, bundle_root = _load_manifest_from_release(release, remote_version)
    except Exception as exc:
        logger.error("Failed to prepare manifest from release %s: %s", remote_version, exc)
        return None

    logger.info(
        "Latest release %s requires update (local=%s)",
        remote_version,
        local_version,
    )
    return {
        "version": remote_version,
        "manifest": manifest,
        "bundle_root": bundle_root,
    }

# DOCKER COMPOSE detection (v2 "docker compose" vs v1 "docker-compose")
def _compose_cmd() -> List[str]:
    candidates = [
        ["docker", "compose"],     # v2
        ["docker-compose"],        # v1
    ]
    for cmd in candidates:
        try:
            subprocess.run(cmd + ["version"], check=True, capture_output=True)
            return cmd
        except Exception:
            continue
    raise RuntimeError("Docker Compose CLI not found (tried: 'docker compose' and 'docker-compose').")

COMPOSE = _compose_cmd()

# STAGING -> BACKUP -> APPLY -> RESTART -> ROLLBACK
def make_release_id(version: str) -> str:
    return f"{version}-{time.strftime('%Y-%m-%dT%H-%M-%SZ', time.gmtime())}"

def stage_files(
    release_id: str,
    files_to_update: List[Dict[str, Any]],
    bundle_root: Path,
) -> Path:
    stage_dir = RELEASES_DIR / release_id
    stage_dir.mkdir(parents=True, exist_ok=False)
    logger.info("Staging into %s", stage_dir)

    for item in files_to_update:
        relpath = str(item["rel"])
        expected_sha = item["sha256"]
        dst = _safe_join(stage_dir, relpath)
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        _copy_bundle_file(bundle_root, relpath, tmp)
        actual = sha256_of_file(str(tmp))
        if actual.lower() != str(expected_sha).lower():
            raise RuntimeError(f"Checksum mismatch for {relpath}: expected {expected_sha}, got {actual}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "rb+") as f:
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dst)
    return stage_dir

def backup_current_files(
    release_id: str,
    files_to_update: List[Dict[str, Any]],
) -> Path:
    backup_dir = BACKUPS_DIR / release_id
    backup_dir.mkdir(parents=True, exist_ok=False)
    logger.info("Backing up current files to %s", backup_dir)

    for item in files_to_update:
        relpath = str(item["rel"])
        src = _safe_join(LIVE_ROOT, relpath)
        dst = _safe_join(backup_dir, relpath)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            (dst.parent / (Path(relpath).name + ".__MISSING__")).touch()

    return backup_dir

def apply_stage_to_live(stage_dir: Path, files_to_update: List[Dict[str, Any]]) -> None:
    for item in files_to_update:
        relpath = str(item["rel"])
        mode = item.get("mode")
        src = _safe_join(stage_dir, relpath)
        dest = _safe_join(LIVE_ROOT, relpath)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".swaptmp")
        shutil.copy2(src, tmp)
        with open(tmp, "rb+") as f:
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dest)
        if mode:
            try:
                m = int(str(mode), 8)
                if m == 0o777:
                    logger.warning("Applying very permissive mode 0777 to %s", dest)
                os.chmod(dest, m)
            except Exception:
                logger.warning("Failed to chmod %s to %s", dest, mode)

def restore_backup(backup_dir: Path, files_to_update: List[Dict[str, Any]]) -> None:
    for item in files_to_update:
        relpath = str(item["rel"])
        bsrc = _safe_join(backup_dir, relpath)
        missing_marker = bsrc.parent / (Path(relpath).name + ".__MISSING__")
        dest = _safe_join(LIVE_ROOT, relpath)
        if missing_marker.exists():
            try:
                if os.path.exists(dest):
                    os.remove(dest)
            except Exception:
                logger.exception("Failed to remove newly added file %s during rollback", dest)
        else:
            if os.path.exists(bsrc):
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(dest.suffix + ".rollbacktmp")
                shutil.copy2(bsrc, tmp)
                with open(tmp, "rb+") as f:
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, dest)

def compose_down_up_build_detached() -> None:
    cmd_down = COMPOSE + ["-f", DOCKER_COMPOSE_YML, "down"]
    cmd_up   = COMPOSE + ["-f", DOCKER_COMPOSE_YML, "up", "--build", "-d"]
    logger.info("Bringing down Docker Compose stack…")
    subprocess.run(cmd_down, check=True, cwd=str(LIVE_ROOT))
    logger.info("Starting Docker Compose (rebuild)…")
    subprocess.run(cmd_up, check=True, cwd=str(LIVE_ROOT))

def prune_dirs(base: Path, keep: int = 5):
    try:
        items = [d for d in base.iterdir() if d.is_dir()]
        items.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for d in items[keep:]:
            try:
                shutil.rmtree(d)
            except Exception:
                logger.warning("Failed to prune %s", d)
    except Exception:
        logger.warning("Prune failed for %s", base)

# IDENTITY & REPORTING
def get_access_key() -> str:
    try:
        with open(ACCESS_KEY_PATH, "r") as f:
            v = f.read().strip()
            if v:
                return v
    except Exception as e:
        logger.error("Failed to read access_key from %s: %s", ACCESS_KEY_PATH, e)
    return ""

def get_hardware_id() -> str:
    try:
        uuid_path = os.getenv("UNIQUE_ID_PATH", "/etc/unique_id")
        if not os.path.exists(uuid_path):
            raise FileNotFoundError(f"UUID file not found at {uuid_path}")
        with open(uuid_path, "r") as f:
            uuid_str = f.read().strip()
        if not uuid_str:
            raise ValueError("UUID is empty.")
        return hashlib.sha256(uuid_str.encode()).hexdigest()[:12]
    except Exception as e:
        logger.error(f"Failed to retrieve hardware ID: {e}")
        return ""

def report_update_to_server(updated_files: List[str], version: str, changelog: List[str],
                            success: bool, message: str = ""):
    payload = {
        "access_key": get_access_key(),
        "hardware_id": get_hardware_id(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "updated_files": updated_files,
        "version": version,
        "changelog": changelog,
        "success": success,
        "detail": message,
    }
    try:
        resp = SESSION.post(CFG.CLIENT_UPDATE_URL, json=payload, timeout=CFG.TOTAL_TIMEOUT)
        resp.raise_for_status()
        logger.info("Reported update to server (success=%s).", success)
    except Exception as e:
        logger.error(f"Failed to report update to server: {e}")

# MAIN LOOP
def main_loop():
    while True:
        try:
            CFG.load()  # hot-reload

            if CFG.DISABLE_UPDATES:
                logger.info("Updates are DISABLED. Sleeping %ds.", CFG.POLL_INTERVAL_SECONDS)
                time.sleep(CFG.POLL_INTERVAL_SECONDS)
                continue

            logger.info("Checking for updates...")

            local_version = read_current_version()
            release_payload = fetch_release_manifest(local_version)
            if release_payload is None:
                logger.warning(
                    "Unable to retrieve release information; retrying in %ds.",
                    CFG.POLL_INTERVAL_SECONDS,
                )
                time.sleep(CFG.POLL_INTERVAL_SECONDS)
                continue

            remote_version = release_payload["version"]
            manifest = release_payload.get("manifest")
            bundle_root_raw = release_payload.get("bundle_root")
            if manifest is None:
                logger.info(
                    "Client already at latest release (local=%s, remote=%s). "
                    "Sleeping %ds.",
                    local_version,
                    remote_version,
                    CFG.POLL_INTERVAL_SECONDS,
                )
                time.sleep(CFG.POLL_INTERVAL_SECONDS)
                continue

            if bundle_root_raw is None:
                logger.error(
                    "Release payload missing bundle root; retrying in %ds.",
                    CFG.POLL_INTERVAL_SECONDS,
                )
                time.sleep(CFG.POLL_INTERVAL_SECONDS)
                continue

            bundle_root = Path(bundle_root_raw)

            if not isinstance(manifest, dict):
                logger.error(
                    "Release manifest payload is invalid; retrying in %ds.",
                    CFG.POLL_INTERVAL_SECONDS,
                )
                time.sleep(CFG.POLL_INTERVAL_SECONDS)
                continue

            full_manifest = manifest
            manifest_version_raw = str(
                full_manifest.get("version", remote_version)
            )
            normalized_version = normalize_version_tag(
                manifest_version_raw or remote_version
            )
            version_for_reporting = manifest_version_raw or remote_version
            changelog = full_manifest.get("changelog", [])
            raw_files = full_manifest.get("files", {})

            logger.info(
                "Preparing update for release %s (local=%s)",
                normalized_version,
                local_version,
            )

            # Normalize to { rel: {sha256,size,mode} }
            normalized: Dict[str, Dict[str, Optional[str]]] = {}
            total_bytes = 0
            for k, v in raw_files.items():
                rel = _strip_dot_slash(str(k))
                if isinstance(v, str):  # support old format {rel: sha256}
                    meta = {"sha256": v, "size": None, "mode": None}
                else:
                    meta = {
                        "sha256": v.get("sha256"),
                        "size": v.get("size"),
                        "mode": v.get("mode"),
                    }
                if not meta["sha256"]:
                    continue
                normalized[rel] = meta
                size_value = meta.get("size")
                if isinstance(size_value, int):
                    total_bytes += size_value
                else:
                    try:
                        total_bytes += _safe_join(bundle_root, rel).stat().st_size
                    except FileNotFoundError:
                        logger.warning("Bundle missing size info for %s", rel)

            remote_files = normalized

            # Preflight disk space (staging)
            try:
                ensure_space(total_bytes, RELEASES_DIR)
            except Exception as e:
                logger.error("Space preflight failed: %s", e)
                time.sleep(CFG.POLL_INTERVAL_SECONDS)
                continue

            # Build update list by hashing local files on demand (covers entire LIVE_ROOT)
            to_update: List[Dict[str, Optional[str]]] = []
            for relpath, meta in remote_files.items():
                local_path = _safe_join(LIVE_ROOT, relpath)
                local_sha = (
                    sha256_of_file(str(local_path))
                    if os.path.exists(local_path)
                    else None
                )
                remote_sha = meta["sha256"]
                if local_sha is None or local_sha.lower() != str(remote_sha).lower():
                    to_update.append({"rel": relpath, **meta})

            # Nothing to do?
            if not to_update:
                logger.info(
                    "No file differences detected for release %s. "
                    "Recording version and sleeping %ds.",
                    normalized_version,
                    CFG.POLL_INTERVAL_SECONDS,
                )
                write_current_version(normalized_version)
                time.sleep(CFG.POLL_INTERVAL_SECONDS)
                continue

            logger.info("Found %d file(s) to update.", len(to_update))
            for item in to_update:
                logger.info("  → %s", item["rel"])

            release_id = make_release_id(normalized_version)

            # Stage changed files
            try:
                stage_dir = stage_files(release_id, to_update, bundle_root)
            except Exception as e:
                logger.exception("Staging failed: %s", e)
                time.sleep(CFG.POLL_INTERVAL_SECONDS)
                continue

            # Backup changed files
            backup_dir = backup_current_files(release_id, to_update)

            # Apply to live
            try:
                apply_stage_to_live(stage_dir, to_update)
                try:
                    os.sync()
                except Exception:
                    pass
            except Exception as e:
                logger.exception("Apply failed; restoring backup. %s", e)
                restore_backup(backup_dir, to_update)
                time.sleep(CFG.POLL_INTERVAL_SECONDS)
                continue

            # Restart; consider success if compose commands succeed
            try:
                compose_down_up_build_detached()

                logger.info("Update applied (compose restart succeeded).")
                report_update_to_server(
                    [str(i["rel"]) for i in to_update],
                    version_for_reporting,
                    changelog,
                    success=True
                )
                write_current_version(normalized_version)
                # Prune old releases/backups (keep last 5)
                prune_dirs(RELEASES_DIR, keep=5)
                prune_dirs(BACKUPS_DIR, keep=5)

            except Exception as e:
                logger.exception("Restart failed; rolling back. %s", e)
                try:
                    restore_backup(backup_dir, to_update)
                    compose_down_up_build_detached()
                    logger.info("Rollback performed.")
                finally:
                    report_update_to_server(
                        [str(i["rel"]) for i in to_update],
                        version_for_reporting,
                        changelog,
                        success=False, message=str(e)
                    )

            time.sleep(CFG.POLL_INTERVAL_SECONDS)

        except Exception:
            logger.exception("Unhandled exception in main loop; sleeping briefly.")
            time.sleep(60)

if __name__ == "__main__":
    try:
        if not LIVE_ROOT.is_dir():
            logger.error("Live root %s does not exist. Exiting.", LIVE_ROOT)
            sys.exit(1)
        if not os.path.isfile(DOCKER_COMPOSE_YML):
            logger.error("Compose file %s not found. Exiting.", DOCKER_COMPOSE_YML)
            sys.exit(1)
        main_loop()
    except RuntimeError as e:
        logger.error("Fatal configuration error: %s", e)
        sys.exit(1)
