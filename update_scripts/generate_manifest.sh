#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
else
  export PYTHONPATH="${SCRIPT_DIR}"
fi

if [ -z ${SPARROW_DIR+x} ]; then
  echo "You need to provide the Sparrow client directory path in SPARROW_DIR environment variable."
  exit 1
fi


CHANGELOG_FILE="${SPARROW_DIR}/CHANGELOG.md"

EXCLUDES=(
  "./manifest.json"
  "./manifest_tmp.json"
  "./manifest_tmp.*.json"
  "./manifest_build.*.json"
  "./old_files.*.json"
  "./new_files.*.json"
  "./files_list.*"           # safety: exclude any old filelist left in dir
  "./config/*"
  "./logs/*"
  "./starlink/config/*"
  "./starlink/logs/*"
)

# cleanup temp files (both in the updates dir and /tmp)
FILELIST=""
cleanup(){
  rm -f manifest_tmp.*.json manifest_build.*.json old_files.*.json new_files.*.json || true
  [[ -n "${FILELIST}" ]] && rm -f "${FILELIST}" || true
}
trap cleanup EXIT

echo "Generating manifest.json in ${SPARROW_DIR}…"
cd "${SPARROW_DIR}"

# Build find args and create null-delimited file list
FIND_ARGS=(. -type f)
for pat in "${EXCLUDES[@]}"; do
  FIND_ARGS+=(! -path "$pat")
done
FIND_ARGS+=(! -iname "*.csv")

# ➜ Create file list OUTSIDE the updates dir so it can't be picked up by find
FILELIST="$(mktemp)"              # e.g., /tmp/tmp.ABC123
find "${FIND_ARGS[@]}" -print0 > "${FILELIST}"

# Count for sanity
COUNT=$(tr -cd '\0' < "${FILELIST}" | wc -c | awk '{print $1}')
echo "Found ${COUNT} files to hash."

# Read changelog (optional)
CHANGELOG_JSON="[]"
if [[ -f "${CHANGELOG_FILE}" ]]; then
  CHANGELOG_JSON=$(python3 -c 'import sys, manifest_utils; print(manifest_utils.changelog_json(sys.argv[1]))' "${CHANGELOG_FILE}")
fi

# Build files map with sha256, size, mode
TMP_MANIFEST="$(mktemp -p . manifest_tmp.XXXXXX.json)"
python3 -c 'import sys, manifest_utils; manifest_utils.write_files_manifest(sys.argv[1], sys.argv[2])' "${TMP_MANIFEST}" "${FILELIST}"

# Version bump only if files changed
OLD_VERSION="1.0.0"
if [[ -f manifest.json ]]; then
  OLD_VERSION=$(grep -oE '"version"\s*:\s*"[^"]+"' manifest.json | sed -E 's/.*"version"\s*:\s*"([^"]+)".*/\1/') || true
fi

FILES_CHANGED="yes"
if [[ -f manifest.json ]]; then
  OLD_TMP=$(mktemp -p . old_files.XXXXXX.json)
  NEW_TMP=$(mktemp -p . new_files.XXXXXX.json)
  python3 -c 'import sys, manifest_utils; manifest_utils.write_files_section(sys.argv[1], sys.argv[2])' "manifest.json" "${OLD_TMP}"
  python3 -c 'import sys, manifest_utils; manifest_utils.write_files_section(sys.argv[1], sys.argv[2])' "${TMP_MANIFEST}" "${NEW_TMP}"
  if cmp -s "$OLD_TMP" "$NEW_TMP"; then FILES_CHANGED="no"; fi
  rm -f "$OLD_TMP" "$NEW_TMP"
fi

NEW_VERSION="${OLD_VERSION}"
if [[ "$FILES_CHANGED" == "yes" ]]; then
  IFS='.' read -r major minor patch <<< "${OLD_VERSION}"
  if [[ "$major" =~ ^[0-9]+$ && "$minor" =~ ^[0-9]+$ && "$patch" =~ ^[0-9]+$ ]]; then
    patch=$((patch+1)); NEW_VERSION="${major}.${minor}.${patch}"
  else
    NEW_VERSION="1.0.0"
  fi
fi
echo "→ Version: ${NEW_VERSION} (files changed: ${FILES_CHANGED})"

# Final assembly
FINAL_TMP="$(mktemp -p . manifest_build.XXXXXX.json)"
env TMP_MANIFEST="${TMP_MANIFEST}" NEW_VERSION="${NEW_VERSION}" CHANGELOG_JSON="${CHANGELOG_JSON}" FINAL_MANIFEST_PATH="${FINAL_TMP}" \
python3 -c 'import os, manifest_utils; manifest_utils.write_final_manifest(os.environ["FINAL_MANIFEST_PATH"], os.environ["TMP_MANIFEST"], os.environ["NEW_VERSION"], os.environ["CHANGELOG_JSON"])'

mv -f "${FINAL_TMP}" manifest.json
echo "→ manifest.json written."
