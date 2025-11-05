"""Utilities for generating dashboard update manifests."""
import hashlib
import json
import os
import stat

from pathlib import Path
from typing import Iterable, Optional


def changelog_json(path: Optional[str]) -> str:
    """Return the changelog file as a JSON array string."""
    if not path:
        return "[]"
    changelog_path = Path(path)
    if not changelog_path.is_file():
        return "[]"

    lines: list[str] = []
    with changelog_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            stripped = raw_line.rstrip("\n")
            if stripped.strip():
                lines.append(stripped)
    return json.dumps(lines, ensure_ascii=False)


def _load_file_list(filelist_path: str) -> list[str]:
    file_list = Path(filelist_path).read_bytes().split(b"\x00")
    return [
        entry.decode("utf-8", "surrogateescape")
        for entry in file_list
        if entry
    ]


def _sha256_of_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files_map(file_paths: Iterable[str]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for absolute_path in file_paths:
        rel_path = os.path.relpath(absolute_path, ".").replace("\\", "/")
        stats = os.stat(absolute_path, follow_symlinks=True)
        mode = format(stat.S_IMODE(stats.st_mode), "04o")
        result[rel_path] = {
            "sha256": _sha256_of_file(absolute_path),
            "size": stats.st_size,
            "mode": mode,
        }
    return result


def write_files_manifest(output_path: str, filelist_path: str) -> None:
    """Create the temporary manifest that only contains the files map."""
    files = _load_file_list(filelist_path)
    payload = {"files": _files_map(files)}
    Path(output_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_files_section(source_manifest: str, destination_path: str) -> None:
    """Write only the files section of a manifest to a destination path."""
    with open(source_manifest, "r", encoding="utf-8") as handle:
        content = json.load(handle)
    files_section = content.get("files", {})
    Path(destination_path).write_text(
        json.dumps(files_section, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def write_final_manifest(
    output_path: str,
    tmp_manifest_path: str,
    version: str,
    changelog_json_blob: str,
) -> None:
    """Combine version, changelog, and files into the final manifest."""
    with open(tmp_manifest_path, "r", encoding="utf-8") as handle:
        files_part = json.load(handle).get("files", {})

    try:
        changelog = (
            json.loads(changelog_json_blob) if changelog_json_blob else []
        )
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid changelog JSON") from exc

    payload = {
        "version": version,
        "changelog": changelog,
        "files": files_part,
    }

    Path(output_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
