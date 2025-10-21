"""
Sensor detection and reading utilities for SHTC3 and BME688, using a simple functional registry.
Provides:
- SHTC3 temperature & humidity
- BME688 pressure + temperature & humidity
- One-time sensor detection and a unified environment read helper
- Add sensors via register_sensor(name, detect_fn, read_fn)
"""

import time
import logging
import threading
from smbus2 import i2c_msg
from typing import Dict, Any, Optional, Callable

# Logger
logger = logging.getLogger(__name__)

# Sensor Configuration
I2C_BUS = 7
SHTC3_ADDRESS = 0x70      # SHTC3
BME688_DEFAULT = 0x77     # default (0x76 alternate)

# Registry & Types
SensorDetect = Callable[[Any], Optional[Dict[str, Any]]]
SensorRead   = Callable[[Any, Dict[str, Any]], Dict[str, Any]]

SENSOR_REGISTRY: Dict[str, Dict[str, Callable]] = {}

def register_sensor(name: str, detect_fn: SensorDetect, read_fn: SensorRead) -> None:
    """Register a sensor by name with its detect/read functions."""
    SENSOR_REGISTRY[name] = {"detect": detect_fn, "read": read_fn}

# BME688 Calibration Cache
_BME688_CALIB = None
_BME688_CALIB_ADDR = None
_BME688_CALIB_LOCK = threading.Lock()

# Global State (Back-Compat)
temp_humidity_sensor = None  # "SHTC3" | None

SENSOR_STATE = {
    "detected": False,
    "temp_hum": None,             # "SHTC3" | None
    "bme_addr": BME688_DEFAULT,   # default 0x77; may become 0x76 if detected
    # New: per-sensor states by name
    "sensors": {}                 # e.g., {"SHTC3": {...}, "BME688": {...}}
}

# I2C Helpers (use provided bus)
def _i2c_write(bus, addr, cmd_bytes):
    write = i2c_msg.write(addr, cmd_bytes)
    bus.i2c_rdwr(write)

def _i2c_read(bus, addr, n):
    read = i2c_msg.read(addr, n)
    bus.i2c_rdwr(read)
    return list(read)

# SHTC3 Impl (detect/read)
def _crc8_sensirion(data):
    crc = 0xFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x31) & 0xFF if (crc & 0x80) else ((crc << 1) & 0xFF)
    return crc

def read_shtc3(bus):
    """Low-level SHTC3 read -> (temp_c, rh_pct) or (None, None)."""
    if bus is None:
        logger.error("I2C bus not initialized. Cannot read SHTC3 sensor.")
        return None, None
    try:
        _i2c_write(bus, SHTC3_ADDRESS, [0x35, 0x17]); time.sleep(0.001)          # wake
        _i2c_write(bus, SHTC3_ADDRESS, [0x78, 0x66]); time.sleep(0.020)          # measure
        data = _i2c_read(bus, SHTC3_ADDRESS, 6)
        _i2c_write(bus, SHTC3_ADDRESS, [0xB0, 0x98]); time.sleep(0.001)          # sleep

        if _crc8_sensirion(data[0:2]) != data[2] or _crc8_sensirion(data[3:5]) != data[5]:
            logger.warning("SHTC3 CRC mismatch."); return None, None

        t_raw = (data[0] << 8) | data[1]
        rh_raw = (data[3] << 8) | data[4]
        temp_c = -45.0 + 175.0 * (t_raw / 65535.0)
        rh_pct = 100.0 * (rh_raw / 65535.0)
        return round(temp_c, 2), round(rh_pct, 2)
    except OSError as e:
        logger.error(f"SHTC3 I2C error: {e}"); return None, None
    except Exception as e:
        logger.error(f"SHTC3 read error: {e}"); return None, None

def detect_shtc3(bus) -> Optional[Dict[str, Any]]:
    """Return {} if SHTC3 responds; else None."""
    try:
        t, h = read_shtc3(bus)
        if t is not None and h is not None:
            logger.info(f"Detected SHTC3 at 0x{SHTC3_ADDRESS:02X}")
            return {}  # no special state needed
    except Exception:
        logger.debug("SHTC3 detection error", exc_info=True)
    return None

def read_shtc3_env(bus, st) -> Dict[str, Any]:
    t, h = read_shtc3(bus)
    return {"t_sht_c": t, "rh_sht_pct": h}

# BME688 Impl (detect/read)
def _bme_u8(bus, addr, reg): return bus.read_byte_data(addr, reg)
def _bme_u16(bus, addr, reg): return (_bme_u8(bus, addr, reg + 1) << 8) | _bme_u8(bus, addr, reg)
def _bme_s16(bus, addr, reg):
    v = _bme_u16(bus, addr, reg)
    return v - 0x10000 if v & 0x8000 else v

def bme688_read_calibration_data(bus, addr):
    global _BME688_CALIB, _BME688_CALIB_ADDR
    if bus is None: raise RuntimeError("I2C bus not initialized")
    with _BME688_CALIB_LOCK:
        if _BME688_CALIB is not None and _BME688_CALIB_ADDR == addr:
            return _BME688_CALIB
        cal = {}
        cal['T1'] = _bme_u16(bus, addr, 0xE9)
        cal['T2'] = _bme_s16(bus, addr, 0x8A)
        t3 = _bme_u8(bus, addr, 0x8C); cal['T3'] = t3 - 256 if (t3 & 0x80) else t3
        cal['P1']  = _bme_u16(bus, addr, 0x8E)
        cal['P2']  = _bme_s16(bus, addr, 0x90)
        p3 = _bme_u8(bus, addr, 0x92);  cal['P3'] = p3 - 256 if (p3 & 0x80) else p3
        cal['P4']  = _bme_s16(bus, addr, 0x94)
        cal['P5']  = _bme_s16(bus, addr, 0x96)
        p6 = _bme_u8(bus, addr, 0x99);  cal['P6'] = p6 - 256 if (p6 & 0x80) else p6
        p7 = _bme_u8(bus, addr, 0x98);  cal['P7'] = p7 - 256 if (p7 & 0x80) else p7
        cal['P8']  = _bme_s16(bus, addr, 0x9C)
        cal['P9']  = _bme_s16(bus, addr, 0x9E)
        cal['P10'] = _bme_u8(bus, addr, 0xA0)
        e1 = _bme_u8(bus, addr, 0xE1); e2 = _bme_u8(bus, addr, 0xE2); e3 = _bme_u8(bus, addr, 0xE3)
        h1 = (e3 << 4) | (e2 & 0x0F); h2 = (e1 << 4) | (e2 >> 4)
        if h1 & 0x800: h1 -= 0x1000
        if h2 & 0x800: h2 -= 0x1000
        cal['H1'] = h1; cal['H2'] = h2
        def _s8(v): return v - 256 if (v & 0x80) else v
        cal['H3'] = _s8(_bme_u8(bus, addr, 0xE4)); cal['H4'] = _s8(_bme_u8(bus, addr, 0xE5))
        cal['H5'] = _s8(_bme_u8(bus, addr, 0xE6)); cal['H6'] = _bme_u8(bus, addr, 0xE7)
        cal['H7'] = _s8(_bme_u8(bus, addr, 0xE8))
        _BME688_CALIB = cal; _BME688_CALIB_ADDR = addr
        logger.info(f"Cached BME688 calibration (addr 0x{addr:02X})")
        return _BME688_CALIB

def _bme_trigger_forced_measurement(bus, addr, osrs_h=0x01, osrs_t=0x02, osrs_p=0x04):
    bus.write_byte_data(addr, 0x72, osrs_h & 0x07)
    bus.write_byte_data(addr, 0x75, 0x00)
    ctrl_meas = ((osrs_t & 0x07) << 5) | ((osrs_p & 0x07) << 2) | 0x01
    bus.write_byte_data(addr, 0x74, ctrl_meas)
    t0 = time.time()
    while True:
        s = bus.read_byte_data(addr, 0x1D)
        measuring = (s & 0x20) != 0; new_data  = (s & 0x80) != 0
        if new_data and not measuring: break
        if (time.time() - t0) > 0.5:   break
        time.sleep(0.003)

def _bme_read_raw_frame(bus, addr):
    p = bus.read_i2c_block_data(addr, 0x1F, 3)
    t = bus.read_i2c_block_data(addr, 0x22, 3)
    h = bus.read_i2c_block_data(addr, 0x25, 2)
    raw_p = (p[0] << 12) | (p[1] << 4) | (p[2] >> 4)
    raw_t = (t[0] << 12) | (t[1] << 4) | (t[2] >> 4)
    raw_h = (h[0] << 8)  |  h[1]
    return raw_t, raw_p, raw_h

def _bme_compensate_temperature(raw_t, cal):
    var1 = (raw_t / 16384.0 - cal['T1'] / 1024.0) * cal['T2']
    var2 = ((raw_t / 131072.0 - cal['T1'] / 8192.0) ** 2) * cal['T3']
    t_fine = var1 + var2
    temp_c = t_fine / 5120.0
    return temp_c, t_fine

def _bme_compensate_pressure(raw_p, t_fine, cal):
    var1 = (t_fine / 2.0) - 64000.0
    var2 = var1 * var1 * cal['P6'] / 131072.0
    var2 += var1 * cal['P5'] * 2.0
    var2 = (var2 / 4.0) + (cal['P4'] * 65536.0)
    var1 = (cal['P3'] * var1 * var1 / 16384.0 + cal['P2'] * var1) / 524288.0
    var1 = (1.0 + var1 / 32768.0) * cal['P1']
    if var1 == 0: return 0.0
    p = 1048576.0 - raw_p
    p = ((p - (var2 / 4096.0)) * 6250.0) / var1
    var1 = cal['P9'] * p * p / 2147483648.0
    var2 = p * cal['P8'] / 32768.0
    p = p + (var1 + var2 + cal['P7']) / 16.0
    return p

def _bme_compensate_humidity(raw_h, temp_c, cal):
    var1 = raw_h - (cal['H1'] * 16.0 + (cal['H3'] / 2.0) * temp_c)
    var2 = (cal['H2'] / 262144.0) * (1.0 +
           (cal['H4'] / 16384.0) * temp_c +
           (cal['H5'] / 1048576.0) * temp_c * temp_c)
    hum = var1 * var2
    hum += ((cal['H6'] / 16384.0) + (cal['H7'] / 2097152.0) * temp_c) * hum * hum
    return max(0.0, min(100.0, hum))

def read_bme688_all(bus, addr):
    cal = bme688_read_calibration_data(bus, addr)
    _bme_trigger_forced_measurement(bus, addr)
    raw_t, raw_p, raw_h = _bme_read_raw_frame(bus, addr)
    temp_c, t_fine = _bme_compensate_temperature(raw_t, cal)
    pressure_pa = _bme_compensate_pressure(raw_p, t_fine, cal)
    rh_pct = _bme_compensate_humidity(raw_h, temp_c, cal)
    return round(temp_c, 2), round(rh_pct, 2), round(pressure_pa, 2)

def detect_bme688(bus) -> Optional[Dict[str, Any]]:
    """Return {'addr': 0x76/0x77} if BME688 present; else None."""
    for addr in (0x77, 0x76):
        try:
            if bus.read_byte_data(addr, 0xD0) == 0x61:
                logger.info(f"Detected BME688 at 0x{addr:02X}")
                return {"addr": addr}
        except Exception:
            continue
    logger.warning("No BME688 detected.")
    return None

def read_bme688_env(bus, st) -> Dict[str, Any]:
    addr = st["addr"]
    t, h, p = read_bme688_all(bus, addr)
    return {"t_bme_c": t, "rh_bme_pct": h, "p_pa": p}

# Sensor Register
register_sensor("SHTC3",  detect_shtc3,  read_shtc3_env)
register_sensor("BME688", detect_bme688, read_bme688_env)

# API: detect & read
def detect_sensors(bus, state):
    """
    Detect attached sensors via registry and update state.
    """
    global temp_humidity_sensor

    if state.get("detected"):
        return state

    if bus is None:
        logger.warning("I2C bus not available for detection.")
        state["detected"] = True
        state["sensors"] = {}
        temp_humidity_sensor = None
        return state

    state["sensors"] = {}
    # registry loop
    for name, fns in SENSOR_REGISTRY.items():
        try:
            st = fns["detect"](bus)
            if st is not None:
                state["sensors"][name] = st
        except Exception:
            logger.error(f"{name} detection failed", exc_info=True)

    # Legacy convenience flags for other modules:
    if "SHTC3" in state["sensors"]:
        state["temp_hum"] = "SHTC3"; temp_humidity_sensor = "SHTC3"
    else:
        state["temp_hum"] = None; temp_humidity_sensor = None

    if "BME688" in state["sensors"]:
        state["bme_addr"] = state["sensors"]["BME688"]["addr"]

    state["detected"] = True
    return state

def read_env(bus, state):
    """
    Unified read: merge all detected sensors' readings.
    Back-compat keys:
      - t_c, rh_pct (prefer SHTC3)
      - p_pa        (from BME688 if present)
    """
    from time import time as now

    if not state.get("detected"):
        detect_sensors(bus, state)

    readings: Dict[str, Any] = {"ts": now()}

    # read each detected sensor
    for name, st in state.get("sensors", {}).items():
        fn = SENSOR_REGISTRY.get(name, {}).get("read")
        if not fn:
            continue
        try:
            r = fn(bus, st) or {}
            readings.update({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
        except Exception as e:
            logger.error(f"{name} read failed: {e}")

    t_sht = readings.get("t_sht_c")
    h_sht = readings.get("rh_sht_pct")
    readings["t_c"]    = t_sht if t_sht is not None else None
    readings["rh_pct"] = h_sht if h_sht is not None else None

    return readings
