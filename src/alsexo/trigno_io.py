# alsexo/trigno_io.py
# Robust Trigno Discover CSV reader with support for per-sensor blocks:
#   - full:     EMG(2) + IMU(12) = 14 cols
#   - imu_only: IMU(12)          = 12 cols
#   - emg_only: EMG(2)           = 2 cols
#
# Key idea: parse columns left-to-right using a header-driven state machine,
# NOT by assuming a uniform block size across the whole file.

from __future__ import annotations

from pathlib import Path
import re
from itertools import islice
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np

FULL_BLOCK_COLS = 14   # EMG(2) + IMU(12)
IMU_BLOCK_COLS  = 12   # ACC(6) + GYRO(6)
EMG_BLOCK_COLS  = 2    # EMG only

DEFAULT_MAX_COLS = 112

CANONICAL_8 = [
    "TrapDesc", "DeltMed", "BicBrachii", "TricBrachii",
    "ExtCarpRad", "FlexCarpRad", "IntDors", "AbdV"
]

ITALIAN_TO_CANONICAL = {
    "InterDors": "IntDors",
    "Trapezio discendente": "TrapDesc",
    "discendente": "TrapDesc",
    "Deltoide Medio": "DeltMed",
    "Medio": "DeltMed",
    "Bicipite": "BicBrachii",
    "Tricipite": "TricBrachii",
    "Estensore": "ExtCarpRad",
    "Flessore": "FlexCarpRad",
    "Abduttore": "AbdV",
    "Abdutore": "AbdV",
}


# -------------------------
# Basic Helpers
# -------------------------
def _read_lines(filepath: Path, max_lines: int = 2000) -> List[str]:
    with filepath.open("r", encoding="utf-8", errors="ignore") as f:
        return [line.rstrip("\n") for line in islice(f, max_lines)]


def find_header_row(lines: List[str]) -> int:
    """
    Find the CSV column header row (Trigno exports have metadata above).
    """
    for i, line in enumerate(lines):
        if ("Time Series" in line) and (
            ("EMG 1 Time Series" in line) or ("ACC X Time Series" in line) or ("GYRO X Time Series" in line)
        ):
            return i
    raise ValueError("Header row not found (no '... Time Series' patterns found).")


def find_sensor_names_row(lines: List[str], header_row: int) -> Optional[int]:
    """
    Scan backwards for a line containing patterns like 'TrapDesc (76652)'.
    """
    pat = re.compile(r"\((\d+)\)")
    for i in range(header_row - 1, max(-1, header_row - 50), -1):
        if pat.search(lines[i]):
            return i
    return None


def parse_sensor_names(line: str) -> List[str]:
    """
    Extract names from patterns like 'TrapDesc (76652)'.
    Handles accented letters and spaces.
    """
    pat = re.compile(r"([A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ0-9 _\-]*)\s*\(\d+\)")
    names = [re.sub(r"\s+", " ", n).strip() for n in pat.findall(line)]
    out: List[str] = []
    for n in names:
        if n not in out:
            out.append(n)
    return out


def _normalize_sensor_names(raw_names: Optional[List[str]], nsensors: int) -> Tuple[Optional[List[str]], str]:
    """
    Returns (names or None, source_str)
    """
    if not raw_names:
        return None, "missing_sensor_row"

    mapped = [ITALIAN_TO_CANONICAL.get(n, n) for n in raw_names]
    mapped = mapped[:nsensors]

    if len(mapped) != nsensors:
        return None, "sensor_count_mismatch"
    if len(set(mapped)) != len(mapped):
        return None, "duplicate_sensor_names"

    return mapped, "normalized_from_header"


def _count_available_columns_from_header_line(header_line: str) -> int:
    return header_line.count(",") + 1


def _safe_read_csv_numeric(filepath: Path, header_row: int, use_ncols: int, on_bad_lines: str) -> pd.DataFrame:
    df = pd.read_csv(
        filepath,
        header=header_row,
        engine="python",
        usecols=range(use_ncols),
        on_bad_lines=on_bad_lines,
    )
    # numeric-coerce values
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# -------------------------
# Header-driven block parsing (state machine)
# -------------------------
def _parse_blocks_by_headers(df: pd.DataFrame) -> List[Dict]:
    """
    Parse sensor blocks left-to-right using a state machine.

    Supported block kinds:
      - full:     EMG(2) + IMU(12)  => 14 cols
      - imu_only: IMU(12)           => 12 cols
      - emg_only: EMG(2)            => 2 cols

    Returns list of dicts:
      {"start": i, "end": j, "kind": "full"|"imu_only"|"emg_only"}
    """
    cols = [str(c).strip() for c in df.columns]
    i = 0
    blocks: List[Dict] = []

    while i < len(cols):
        c = cols[i]

        # --- EMG-start: either full or emg_only ---
        if c.startswith("EMG 1 Time Series"):
            # need EMG value col next
            if i + 1 >= len(cols) or not cols[i + 1].startswith("EMG 1 ("):
                # malformed; stop to avoid cascading mis-parse
                break

            # if ACC follows immediately after EMG pair => full
            if i + 2 < len(cols) and cols[i + 2].startswith("ACC X Time Series"):
                if i + FULL_BLOCK_COLS <= len(cols):
                    blocks.append({"start": i, "end": i + FULL_BLOCK_COLS, "kind": "full"})
                    i += FULL_BLOCK_COLS
                    continue
                break
            else:
                # EMG-only
                blocks.append({"start": i, "end": i + EMG_BLOCK_COLS, "kind": "emg_only"})
                i += EMG_BLOCK_COLS
                continue

        # --- IMU-only ---
        if c.startswith("ACC X Time Series"):
            if i + 1 >= len(cols) or not cols[i + 1].startswith("ACC X ("):
                break
            if i + IMU_BLOCK_COLS <= len(cols):
                blocks.append({"start": i, "end": i + IMU_BLOCK_COLS, "kind": "imu_only"})
                i += IMU_BLOCK_COLS
                continue
            break

        # otherwise skip forward
        i += 1

    return blocks


def _build_emg_from_two_col_block(block: pd.DataFrame) -> Optional[pd.DataFrame]:
    # columns: [t_emg, emg_mv]
    emg = pd.DataFrame({"t_emg": block.iloc[:, 0], "emg_mv": block.iloc[:, 1]}).dropna(subset=["t_emg", "emg_mv"])
    return emg if len(emg) else None


def _build_imu_from_full_block(block: pd.DataFrame) -> Optional[pd.DataFrame]:
    # columns: [EMG t, EMG x, ACC X t, ACC X, ACC Y t, ACC Y, ACC Z t, ACC Z, GYRO X t, GYRO X, GYRO Y t, GYRO Y, GYRO Z t, GYRO Z]
    imu = pd.DataFrame(
        {
            "t_imu": block.iloc[:, 2],
            "acc_x_g": block.iloc[:, 3],
            "acc_y_g": block.iloc[:, 5],
            "acc_z_g": block.iloc[:, 7],
            "gyro_x_dps": block.iloc[:, 9],
            "gyro_y_dps": block.iloc[:, 11],
            "gyro_z_dps": block.iloc[:, 13],
        }
    ).dropna(subset=["t_imu"])
    return imu if len(imu) else None


def _build_imu_from_imu_only_block(block: pd.DataFrame) -> Optional[pd.DataFrame]:
    # 12-col order:
    # ACC X time,val | ACC Y time,val | ACC Z time,val | GYRO X time,val | GYRO Y time,val | GYRO Z time,val
    imu = pd.DataFrame(
        {
            "t_imu": block.iloc[:, 0],
            "acc_x_g": block.iloc[:, 1],
            "acc_y_g": block.iloc[:, 3],
            "acc_z_g": block.iloc[:, 5],
            "gyro_x_dps": block.iloc[:, 7],
            "gyro_y_dps": block.iloc[:, 9],
            "gyro_z_dps": block.iloc[:, 11],
        }
    ).dropna(subset=["t_imu"])
    return imu if len(imu) else None


# -------------------------
# Main reader
# -------------------------
def read_trigno_csv(
    filepath: Path,
    max_cols: int = DEFAULT_MAX_COLS,
    prefer_canonical_if_8: bool = True,
    on_bad_lines: str = "skip",
) -> Tuple[List[str], Dict[str, Dict[str, Optional[pd.DataFrame]]], Dict]:
    """
    Robust Trigno reader supporting mixed per-sensor presence of EMG/IMU:
      - full blocks:     14 cols
      - imu_only blocks: 12 cols
      - emg_only blocks: 2 cols

    Returns:
      sensors: list[str]
      per_sensor: dict[sensor] -> {"emg": df or None, "imu": df or None}
      meta: dict with parse/layout info
    """
    filepath = Path(filepath)

    # --- Read header lines to find header row and sensor row ---
    lines = _read_lines(filepath, max_lines=2000)
    header_row = find_header_row(lines)
    sensor_row = find_sensor_names_row(lines, header_row)

    raw_sensor_names: Optional[List[str]] = None
    if sensor_row is not None:
        raw_sensor_names = parse_sensor_names(lines[sensor_row])

    # --- Determine available columns from the header line text (fast) ---
    n_avail = _count_available_columns_from_header_line(lines[header_row])
    use_n = min(int(max_cols), int(n_avail))

    # --- Read numeric table (first use_n cols) ---
    df = _safe_read_csv_numeric(filepath, header_row=header_row, use_ncols=use_n, on_bad_lines=on_bad_lines)

    # Remove completely empty columns after numeric coercion
    df = df.dropna(axis=1, how="all")

    # --- Parse blocks by headers (state machine) ---
    blocks = _parse_blocks_by_headers(df)
    nsensors = len(blocks)

    # --- Decide sensor names ---
    sensors, sensor_name_source = _normalize_sensor_names(raw_sensor_names, nsensors)
    if sensors is None:
        if prefer_canonical_if_8 and nsensors == 8:
            sensors = CANONICAL_8.copy()
            sensor_name_source = "canonical_fallback_8"
        else:
            sensors = [f"Sensor_{i+1:02d}" for i in range(nsensors)]
            sensor_name_source = "placeholder_fallback"

    # --- Build per-sensor dfs ---
    per_sensor: Dict[str, Dict[str, Optional[pd.DataFrame]]] = {}
    kinds: List[str] = []

    for sname, binfo in zip(sensors, blocks):
        kind = binfo["kind"]
        kinds.append(kind)

        block_df = df.iloc[:, binfo["start"] : binfo["end"]]

        if kind == "full":
            emg = _build_emg_from_two_col_block(block_df.iloc[:, 0:2])
            imu = _build_imu_from_full_block(block_df)
        elif kind == "imu_only":
            emg = None
            imu = _build_imu_from_imu_only_block(block_df)
        else:  # emg_only
            emg = _build_emg_from_two_col_block(block_df.iloc[:, 0:2])
            imu = None

        per_sensor[sname] = {"emg": emg, "imu": imu}

    # --- Layout label ---
    if nsensors == 0:
        layout = "unknown"
    elif all(k == "full" for k in kinds):
        layout = "full_emg_imu"
    elif all(k == "imu_only" for k in kinds):
        layout = "imu_only"
    elif all(k == "emg_only" for k in kinds):
        layout = "emg_only"
    else:
        layout = "mixed"

    # --- Meta ---
    meta = {
        "file": filepath.name,
        "header_row": int(header_row),
        "sensor_names_row": int(sensor_row) if sensor_row is not None else None,

        "raw_sensor_names": raw_sensor_names,
        "final_sensor_names": sensors,
        "sensor_name_source": sensor_name_source,

        "layout": layout,
        "block_cols": np.nan,  # variable (2/12/14)
        "block_kinds": "|".join(kinds),

        "max_cols_requested": int(max_cols),
        "ncols_available_in_header": int(n_avail),
        "ncols_requested_usecols": int(use_n),

        "ncols_read_after_drop_empty": int(df.shape[1]),
        "ncols_used": int(df.shape[1]),
        "nsensors": int(nsensors),
        "nrows_loaded": int(df.shape[0]),

        "mapping_ok": bool(nsensors > 0),
        "mapping_issues": "" if nsensors > 0 else "no_blocks_parsed",
        "on_bad_lines": on_bad_lines,
    }

    return sensors, per_sensor, meta
