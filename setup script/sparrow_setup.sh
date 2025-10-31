#!/usr/bin/env bash
###############################################################################
#  SPARROW Setup Script           31-October-2025
###############################################################################
set -euo pipefail

###############################################################################
# 0.  -- GUI HELPERS --
###############################################################################
GUI=true; for a in "$@"; do [[ "$a" == "--no-gui" ]] && GUI=false; done

_install_zenity() {
    $GUI || return
    if ! command -v zenity >/dev/null 2>&1; then
        echo "Installing Zenity..." >&2
        apt-get update -qq
        DEBIAN_FRONTEND=noninteractive apt-get install -y zenity >/dev/null
    fi
}

_yesno() {               # returns 0 (true) for Yes / OK
    if $GUI && command -v zenity >/dev/null 2>&1; then
        zenity --question --no-wrap --text="$1" --width=440
    else
        read -rp "$1 (y/n): " _ans; [[ "${_ans,,}" =~ ^y(es)?$ ]]
    fi
}

_input() {               # $1 prompt , $2 = "hide" to mask
    if $GUI && command -v zenity >/dev/null 2>&1; then
        local opts=(--entry --text="$1" --width=460)
        [[ "${2:-}" == "hide" ]] && opts+=(--hide-text)
        zenity "${opts[@]}"
    else
        if [[ "${2:-}" == "hide" ]]; then
            read -rsp "$1: " _txt; echo; echo "$_txt"
        else
            read -rp "$1: " _txt; echo "$_txt"
        fi
    fi
}

_info()  { $GUI && command -v zenity >/dev/null 2>&1 && zenity --info  --no-wrap --text="$*" --width=480 || echo -e "$*"; }
_error() { $GUI && command -v zenity >/dev/null 2>&1 && zenity --error --no-wrap --text="$*" --width=480 || { echo -e "ERROR: $*" >&2; } }

_progress() {            # $1 = message, $2… = command
    if $GUI && command -v zenity >/dev/null 2>&1; then
        ( "${@:2}" ) 2>&1 | zenity --progress --pulsate --no-cancel \
                                    --auto-close --text="$1" --width=480
    else
        "${@:2}"
    fi
}

###############################################################################
# 1.  -- CONFIGURATION --
###############################################################################
DOCKER_COMPOSE_VERSION="2.21.0"
DOCKER_COMPOSE_URL="https://github.com/docker/compose/releases/download/v${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)"
DOCKER_COMPOSE_PATH="/usr/local/bin/docker-compose"
SYMLINK_PATH="/usr/bin/docker-compose"

LOG_FILE="/var/log/sparrow_setup.log"
UUID_FILE="/etc/unique_id"

HOTSPOT_SSID="CameraTraps"
HOTSPOT_PASSWORD=""        # set interactively via prompt_hotspot_password
WIFI_INTERFACE="wlan0"

MODEL_DOWNLOAD_URL="https://zenodo.org/record/14661733/files/MDV6b-yolov9c.onnx?download=1"
MODEL_DIR_NAME="1"
MODEL_FILENAME_TEMP="MDV6b-yolov9c.onnx"
MODEL_FILENAME_FINAL="model.onnx"

AI4G_MODEL_DOWNLOAD_URL="https://zenodo.org/records/15041754/files/AI4GAmazonClassificationV2.onnx?download=1"
AI4G_MODEL_DIR_NAME="1"
AI4G_MODEL_FILENAME_TEMP="AI4GAmazonClassificationV2.onnx"
AI4G_MODEL_FILENAME_FINAL="model.onnx"

AUDIO_BIRDS_MODEL_DOWNLOAD_URL="https://zenodo.org/records/17256803/files/MD_AudioBirds_V1.onnx?download=1"
AUDIO_BIRDS_MODEL_DIR_NAME="1"
AUDIO_BIRDS_MODEL_FILENAME_TEMP="MD_AudioBirds_V1.onnx"
AUDIO_BIRDS_MODEL_FILENAME_FINAL="model.onnx"

REPO_URL="https://github.com/zhmiao/sparrow.git"
CLONE_DIR=""

###############################################################################
# 2.  -- UTILITY FUNCTIONS --
###############################################################################
log() { echo -e "$(date +"%Y-%m-%d %H:%M:%S") : $*" | tee -a "$LOG_FILE"; }
command_exists() { command -v "$1" >/dev/null 2>&1; }

install_uuidgen() {
    if command_exists uuidgen; then
        log "uuidgen already installed."
    else
        mkdir -p /run/uuidd && chmod 755 /run/uuidd
        apt-get update -y && apt-get install -y uuid-runtime
    fi
}

generate_unique_id() {
    [[ -s "$UUID_FILE" ]] && { log "UUID exists: $(cat "$UUID_FILE")"; return; }
    NEW_UUID=$(uuidgen)
    echo "$NEW_UUID" | tee "$UUID_FILE" >/dev/null
    chmod 644 "$UUID_FILE"; chown root:root "$UUID_FILE"
    log "Generated UUID: $NEW_UUID"
}

install_curl()  { command_exists curl  || { apt-get update -y; apt-get install -y curl;  } }
install_wget()  { command_exists wget  || { apt-get update -y; apt-get install -y wget;  } }
install_git()   { command_exists git   || { apt-get update -y; apt-get install -y git;   } }

install_docker() {
    apt-get update -y
    apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] \
    https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
        | tee /etc/apt/sources.list.d/docker.list >/dev/null
    apt-get update -y
    apt-get install -y docker-ce docker-ce-cli containerd.io
}

install_docker_compose() {
    rm -f "$DOCKER_COMPOSE_PATH" "$SYMLINK_PATH" 2>/dev/null || true
    curl -L "$DOCKER_COMPOSE_URL" -o "$DOCKER_COMPOSE_PATH"
    chmod +x "$DOCKER_COMPOSE_PATH"
    ln -sf "$DOCKER_COMPOSE_PATH" "$SYMLINK_PATH"
}

# Minimal prompt: ask twice, ensure match (no strength rules)
prompt_hotspot_password() {
    local p1 p2
    while true; do
        p1=$(_input "Enter Wi-Fi Hotspot password for SSID \"$HOTSPOT_SSID\"" hide)
        p2=$(_input "Re-enter the Wi-Fi Hotspot password" hide)

        if [[ -z "${p1:-}" || -z "${p2:-}" ]]; then
            _error "Password cannot be empty."
            continue
        fi
        if [[ "$p1" != "$p2" ]]; then
            _error "Passwords do not match. Please try again."
            continue
        fi
        # Basic WPA length guard so nmcli doesn't fail immediately
        if (( ${#p1} < 8 || ${#p1} > 63 )); then
            _error "WPA2-PSK password must be between 8 and 63 characters."
            continue
        fi
        HOTSPOT_PASSWORD="$p1"
        break
    done
}

setup_persistent_wifi_hotspot() {
    local ssid="$HOTSPOT_SSID" pw="$HOTSPOT_PASSWORD" iface="$WIFI_INTERFACE"

    if [[ -z "$pw" ]]; then
        _error "Hotspot password is empty. This should have been set earlier."
        return 1
    fi

    if nmcli connection show "$ssid" >/dev/null 2>&1; then
        # Update PSK in case user changed it on a re-run
        nmcli connection modify "$ssid" wifi-sec.psk "$pw"
        nmcli -t -f NAME connection show --active | grep -qx "$ssid" || nmcli connection up "$ssid"
    else
        nmcli connection add type wifi ifname "$iface" con-name "$ssid" autoconnect yes ssid "$ssid"
        nmcli connection modify "$ssid" 802-11-wireless.mode ap 802-11-wireless.band bg ipv4.method shared
        nmcli connection modify "$ssid" wifi-sec.key-mgmt wpa-psk
        nmcli connection modify "$ssid" wifi-sec.psk "$pw"
        nmcli connection up "$ssid"
    fi
}

download_model() {
    local dir="$SYSTEM_FOLDER/Models/tritonserver/model_repository/megadetectorv6/$MODEL_DIR_NAME"
    local tmp="$dir/$MODEL_FILENAME_TEMP" fin="$dir/$MODEL_FILENAME_FINAL"
    mkdir -p "$dir"
    while true; do
        if _progress "Downloading Megadetector v6..." wget -q -O "$tmp" "$MODEL_DOWNLOAD_URL"; then
            mv "$tmp" "$fin"; break
        fi
        _yesno "Megadetector download failed. Retry?" || { _error "Aborted."; exit 1; }
    done
}

download_model_ai4g() {
    local dir="$SYSTEM_FOLDER/Models/tritonserver/model_repository/AI4GAmazonClassification/$AI4G_MODEL_DIR_NAME"
    local tmp="$dir/$AI4G_MODEL_FILENAME_TEMP" fin="$dir/$AI4G_MODEL_FILENAME_FINAL"
    mkdir -p "$dir"
    while true; do
        if _progress "Downloading AI4G model..." wget -q -O "$tmp" "$AI4G_MODEL_DOWNLOAD_URL"; then
            mv "$tmp" "$fin"; break
        fi
        _yesno "AI4G model download failed. Retry?" || { _error "Aborted."; exit 1; }
    done
}

download_model_audio_birds() {
    local dir="$SYSTEM_FOLDER/Models/tritonserver/model_repository/megadetector_birds_v1/$AUDIO_BIRDS_MODEL_DIR_NAME"
    local tmp="$dir/$AUDIO_BIRDS_MODEL_FILENAME_TEMP" fin="$dir/$AUDIO_BIRDS_MODEL_FILENAME_FINAL"
    mkdir -p "$dir"
    while true; do
        if _progress "Downloading MD Audio Birds v1..." wget -q -O "$tmp" "$AUDIO_BIRDS_MODEL_DOWNLOAD_URL"; then
            mv "$tmp" "$fin"; break
        fi
        _yesno "Audio Birds model download failed. Retry?" || { _error "Aborted."; exit 1; }
    done
}

clone_public_repo() {
    CLONE_DIR="$SYSTEM_FOLDER"  # project root = system/
    local tmpdir
    tmpdir="$(mktemp -d)"
    _progress "Cloning repo..." git clone "$REPO_URL" "$tmpdir"
    (
      shopt -s dotglob
      cp -a "$tmpdir"/* "$CLONE_DIR"/
    )
    rm -rf "$tmpdir"
    create_additional_directories
}

create_additional_directories() {
    # ensure runtime dirs under sparrow/ and starlink/
    mkdir -p "$SYSTEM_FOLDER/sparrow"/{logs,images,recordings,static/data,static/gallery,config}
    mkdir -p "$SYSTEM_FOLDER/starlink"/{logs,config}
}

create_folders() {
    local uh; uh=$(eval echo ~"${SUDO_USER:-$USER}")
    SYSTEM_FOLDER="$uh/Desktop/system"
    mkdir -p "$SYSTEM_FOLDER/Models/tritonserver/model_repository/megadetectorv6/$MODEL_DIR_NAME"
    mkdir -p "$SYSTEM_FOLDER/Models/tritonserver/model_repository/AI4GAmazonClassification/$AI4G_MODEL_DIR_NAME"
    mkdir -p "$SYSTEM_FOLDER/Models/tritonserver/model_repository/megadetector_birds_v1/$AUDIO_BIRDS_MODEL_DIR_NAME"
    CLONE_DIR="$SYSTEM_FOLDER"  # compose/project root

# config.pbtxt - Megadetector
cat >"$SYSTEM_FOLDER/Models/tritonserver/model_repository/megadetectorv6/config.pbtxt" <<'EOF'
name: "megadetectorv6"
platform: "onnxruntime_onnx"
max_batch_size: 0

input [
  {
    name: "images"
    data_type: TYPE_FP32
    dims: [1, 3, 640, 640]
  }
]

output [
  {
    name: "output0"
    data_type: TYPE_FP32
    dims: [-1, -1, -1]
  }
]

parameters {
  key: "intra_op_num_threads"
  value: { string_value: "2" }
}
parameters {
  key: "inter_op_num_threads"
  value: { string_value: "2" }
}
parameters {
  key: "execution_mode"
  value: { string_value: "1" }   # 1=parallel, 0=sequential
}
parameters {
  key: "enable_cpu_mem_arena"
  value: { string_value: "1" }
}
parameters {
  key: "enable_mem_pattern"
  value: { string_value: "1" }
}

instance_group [
  {
    kind: KIND_GPU
    gpus: [0]
    count: 1
  }
]
EOF

# config.pbtxt - AI4G Amazon Classificaion Model
cat >"$SYSTEM_FOLDER/Models/tritonserver/model_repository/AI4GAmazonClassification/config.pbtxt" <<'EOF'
name: "AI4GAmazonClassification"
platform: "onnxruntime_onnx"
max_batch_size: 0

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [-1, 3, 224, 224]
  }
]

output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [-1, 36]
  }
]

parameters {
  key: "intra_op_num_threads"
  value: { string_value: "2" }
}
parameters {
  key: "inter_op_num_threads"
  value: { string_value: "2" }
}
parameters {
  key: "execution_mode"
  value: { string_value: "1" }   # 1=parallel, 0=sequential
}
parameters {
  key: "enable_cpu_mem_arena"
  value: { string_value: "1" }
}
parameters {
  key: "enable_mem_pattern"
  value: { string_value: "1" }
}

instance_group [
  {
    kind: KIND_GPU
    gpus: [0]
    count: 1
  }
]
EOF

# config.pbtxt - MD Audio Birds v1
cat >"$SYSTEM_FOLDER/Models/tritonserver/model_repository/megadetector_birds_v1/config.pbtxt" <<'EOF'
name: "megadetector_birds_v1"
backend: "onnxruntime"
max_batch_size: 32

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [1, 224, -1]
  }
]

output [
  {
    name: "logits"
    data_type: TYPE_FP32
    dims: [1]
  }
]

parameters {
  key: "intra_op_num_threads"
  value: { string_value: "2" }
}
parameters {
  key: "inter_op_num_threads"
  value: { string_value: "2" }
}
parameters {
  key: "execution_mode"
  value: { string_value: "1" }   # 1 = parallel, 0 = sequential
}
parameters {
  key: "enable_cpu_mem_arena"
  value: { string_value: "1" }
}
parameters {
  key: "enable_mem_pattern"
  value: { string_value: "1" }
}

instance_group [
  {
    kind: KIND_GPU
    gpus: [0]
    count: 1
  }
]
EOF
    download_model
    download_model_ai4g
    download_model_audio_birds
    clone_public_repo
}

install_smbus2() {
    python3 -m pip show smbus2 >/dev/null 2>&1 && return
    command_exists pip3 || { apt-get update -y; apt-get install -y python3-pip; }
    python3 -m pip install smbus2
}

###############################################################################
# 3.  -- DS3231 RTC SEED --
###############################################################################
seed_ds3231() {
    log "Seeding DS3231 RTC with current UTC time..."
    max_retries=5
    attempt=1
    current_time=""

    # 1) Try World Clock API
    while [ $attempt -le $max_retries ]; do
        log "Attempt $attempt/$max_retries: fetching UTC from World Clock API..."
        raw=$(curl -sL http://worldclockapi.com/api/json/utc/now)
        current_time=$(echo "$raw" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("currentDateTime",""))' 2>/dev/null \
        || true)

        if [ -n "$current_time" ]; then
            current_time=$(echo "$current_time" | sed 's/T/ /; s/Z//'):00
            log "Retrieved UTC time from API: $current_time"
            break
        fi

        log "World Clock API fetch failed on attempt $attempt."
        attempt=$((attempt+1))
        sleep 2
    done

    # 2) Fallback to system NTP if API never returned a time
    if [ -z "$current_time" ]; then
        log "World Clock API failed after $max_retries attempts. Falling back to system NTP sync."
        echo "World Clock API failed; using system NTP to obtain UTC time."

        if ! command_exists timedatectl; then
            log "timedatectl not found—installing systemd-timesyncd..."
            apt-get update -y || log "Warning: apt-get update failed, continuing"
            apt-get install -y systemd-timesyncd || log "Warning: systemd-timesyncd install failed, continuing"
        fi

        if ! timedatectl set-ntp true; then
            log "Warning: timedatectl set-ntp failed—continuing with local system time"
            current_time=$(date -u +"%Y-%m-%d %H:%M:%S")
            log "Using local UTC time: $current_time"
        else
            sync_max=5
            for i in $(seq 1 $sync_max); do
                if [ "$(timedatectl show -p NTPSynchronized --value)" = "yes" ]; then
                    current_time=$(date -u +"%Y-%m-%d %H:%M:%S")
                    log "Retrieved UTC time via NTP: $current_time"
                    break
                fi
                log "Waiting for NTP synchronization ($i/$sync_max)…"
                sleep 2
            done
            if [ -z "$current_time" ]; then
                log "Warning: NTP never synchronized—using local UTC time"
                current_time=$(date -u +"%Y-%m-%d %H:%M:%S")
                log "Local UTC time: $current_time"
            fi
        fi
    fi

    # 3) Write to the DS3231, with its own retry loop
    write_attempts=1
    while [ $write_attempts -le $max_retries ]; do
        if python3 - <<EOF
import smbus2
from datetime import datetime
rtc = datetime.strptime("$current_time", "%Y-%m-%d %H:%M:%S")
int_to_bcd = lambda v: ((v // 10) << 4) | (v % 10)
bus = smbus2.SMBus(7)
DS3231_ADDR = 0x68
data = [
    int_to_bcd(rtc.second),
    int_to_bcd(rtc.minute),
    int_to_bcd(rtc.hour),
    int_to_bcd((rtc.isoweekday() % 7) + 1),
    int_to_bcd(rtc.day),
    int_to_bcd(rtc.month),
    int_to_bcd(rtc.year - 2000)
]
bus.write_i2c_block_data(DS3231_ADDR, 0x00, data)
bus.close()
EOF
        then
            log "DS3231 time set to: $current_time"
            return
        else
            log "DS3231 write failed on attempt $write_attempts."
            if [ $write_attempts -lt $max_retries ]; then
                if _yesno "Seeding failed. Retry write?" ; then
                    :
                else
                    log "User aborted DS3231 seeding during write."
                    echo "Automatic DS3231 RTC Failed!! Please manually seed the DS3231 RTC before running Docker Compose!!"
                    return
                fi
            fi
            write_attempts=$((write_attempts+1))
            sleep 2
        fi
    done

    log "Exceeded maximum retries ($max_retries) for DS3231 seeding."
    echo "Automatic DS3231 RTC Failed!! Please manually seed the DS3231 RTC before running Docker Compose!!"
}

configure_access_key() {
    while true; do
        k1=$(_input "Enter Sparrow Access Key" hide)
        k2=$(_input "Re-enter Sparrow Access Key" hide)
        [[ "$k1" == "$k2" ]] && break
        _error "Keys do not match - try again."
    done
    mkdir -p "$SYSTEM_FOLDER/sparrow/config" "$SYSTEM_FOLDER/starlink/config"
    echo "$k1" >"$SYSTEM_FOLDER/sparrow/config/access_key.txt"
    echo "$k1" >"$SYSTEM_FOLDER/starlink/config/access_key.txt"
}

###############################################################################
# 4.  -- MAIN SCRIPT LOGIC --
###############################################################################
[ "$EUID" -eq 0 ] || { _error "Run as root (sudo)."; exit 1; }
_install_zenity
touch "$LOG_FILE"; chmod 644 "$LOG_FILE"

_info "Welcome to the SPARROW Setup Wizard!\nThis will install all prerequisites, download the required model and start SPARROW."

_yesno "Have you installed the OS directly onto the SSD using NVIDIA SDK Manager?" \
    || { _error "Please flash the OS first, then rerun."; exit 1; }

_info "Checking internet connectivity. Press ok to continue"
ping -c3 8.8.8.8 >/dev/null 2>&1 || { _error "No internet connection."; exit 1; }

install_wget; install_git; create_folders
install_uuidgen; generate_unique_id; install_curl

if command_exists docker; then log "Docker already installed."
else _progress "Installing Docker..." install_docker; fi

if command_exists docker-compose; then
    cur=$(docker-compose --version | awk '{print $4}' | tr -d ',v')
    [[ "$cur" == "$DOCKER_COMPOSE_VERSION" ]] || _progress "Updating Docker-Compose..." install_docker_compose
else _progress "Installing Docker-Compose..." install_docker_compose; fi

# prompt for hotspot password before configuring Wi-Fi AP
prompt_hotspot_password
setup_persistent_wifi_hotspot

install_smbus2
seed_ds3231
configure_access_key

if _yesno "Install TeamViewer Host (ARM64)?" ; then
    _progress "Installing TeamViewer Host..." bash -c '
        wget -qO /tmp/tv.deb https://download.teamviewer.com/download/linux/teamviewer-host_arm64.deb &&
        apt-get install -y /tmp/tv.deb &&
        systemctl enable teamviewerd.service &&
        teamviewer passwd Sparrow0102!'
fi

log "Building Sparrow containers..."
cd "$SYSTEM_FOLDER"

# Enable Docker BuildKit + plain CLI progress
export COMPOSE_DOCKER_CLI_BUILD=1
export DOCKER_BUILDKIT=1

if $GUI && command -v zenity >/dev/null; then
    log "docker-compose build (GUI+console) with BuildKit + no-cache started"
    docker-compose build --no-cache --progress=plain 2>&1 \
      | tee /dev/tty \
      | zenity --progress --pulsate --no-cancel --auto-close \
               --text="Building Docker images (no cache)…" --width=480
    log "docker-compose build (GUI+console) with BuildKit + no-cache completed"
else
    log "docker-compose build (CLI) with BuildKit + no-cache started"
    docker-compose build --no-cache --progress=plain
    log "docker-compose build (CLI) with BuildKit + no-cache completed"
fi

# 1) Start in detached mode, showing pull/start logs in a Zenity window (or CLI)
if $GUI && command -v zenity >/dev/null; then
    log "docker-compose up (GUI+console) started"
    docker-compose up -d 2>&1 \
      | tee /dev/tty \
      | zenity --text-info \
               --title="Starting Sparrow Containers" \
               --width=800 --height=600 \
               --font="Monospace 10"
    log "docker-compose up (GUI+console) completed"
else
    _progress "Starting Sparrow.." docker-compose up -d
    log "Sparrow containers started"
fi

# 2) Follow the logs in real time
log "Tailing Sparrow logs (Ctrl-C to exit)..."
if $GUI && command -v zenity >/dev/null; then
    docker-compose logs --no-color --follow \
      | tee /dev/tty \
      | zenity --text-info \
               --title="Sparrow Container Logs" \
               --width=800 --height=600 \
               --font="Monospace 10"
else
    docker-compose logs --tail=100 --follow
fi

_info "Setup completed - Sparrow is now running!\n\nTo follow logs:\n  cd $SYSTEM_FOLDER && docker-compose logs -f"
exit 0
