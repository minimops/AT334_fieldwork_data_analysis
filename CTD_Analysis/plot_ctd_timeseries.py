"""
Generate a polished, multi-panel time-series plot per track: conductivity,
temperature, and depth vs. time of day -- a cleaner version of Neptus's
default CTD time-series export (stacked panels, shared time axis).

DATA SOURCE
-----------
In Neptus MRA, for each mission: right-click -> "Export (Georeferenced)
CTD to CSV". Save one CSV per track into a folder.

USAGE
-----
    python plot_ctd_timeseries.py <tracks_dir>

OUTPUTS
-------
    ctd_timeseries/<track_name>_timeseries.png   one per track
"""

import sys
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRACKS_DIR = sys.argv[1] if len(sys.argv) > 1 else "tracks"
OUTPUT_DIR = "ctd_timeseries"

RAW_HEADER = [
    "# timestamp", "gmt time", "latitude", "longitude", "lat (corrected)",
    "lon (corrected)", "altitude", "depth", "medium", "conductivity",
    "temperature", "salinity",
]
CORRECTED_ORDER = [
    "gmt time", "latitude", "longitude", "lat (corrected)",
    "lon (corrected)", "altitude", "depth", "medium",
    "conductivity", "temperature", "salinity", "extra",
]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "font.size": 10,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_time_str(t):
    """'HH:MM:SS.sss' (with possible leading space) -> datetime.time-like
    seconds-since-midnight, wrapped in a dummy date so matplotlib can treat
    it as a proper time axis."""
    try:
        h, m, s = str(t).strip().split(":")
        total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
        return datetime(2000, 1, 1) + timedelta(seconds=total_seconds)
    except (ValueError, AttributeError):
        return pd.NaT


def load_track(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if list(df.columns) == RAW_HEADER and len(df.columns) == len(CORRECTED_ORDER):
        df.columns = CORRECTED_ORDER

    out = df[["gmt time", "conductivity", "temperature", "depth"]].copy()
    out = out.rename(columns={"gmt time": "time_str"})

    for col in ["conductivity", "temperature", "depth"]:
        if out[col].dtype == object:
            out[col] = out[col].astype(str).str.replace(",", ".", regex=False)
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["time"] = out["time_str"].apply(parse_time_str)
    out = out.dropna(subset=["conductivity", "temperature", "depth", "time"])
    out = out.sort_values("time")

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    csv_files = sorted(glob.glob(os.path.join(TRACKS_DIR, "*.csv")))
    if not csv_files:
        print(f"No CSV files found in '{TRACKS_DIR}'.")
        return

    print(f"Found {len(csv_files)} CTD track file(s) in '{TRACKS_DIR}'.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for path in csv_files:
        name = os.path.splitext(os.path.basename(path))[0]
        df = load_track(path)
        if len(df) < 5:
            print(f"  [skipped] {name}: too few valid rows ({len(df)})")
            continue
        print(f"  loaded {name}: {len(df)} rows")

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        axes[0].plot(df["time"], df["conductivity"], color="tab:red", linewidth=0.9)
        axes[0].set_ylabel("Conductivity\n(S/m)")

        axes[1].plot(df["time"], df["temperature"], color="tab:blue", linewidth=0.9)
        axes[1].set_ylabel("Temperature\n(°C)")

        # Depth plotted as the "pressure"-style panel, inverted so deeper
        # excursions point downward, matching physical intuition.
        axes[2].plot(df["time"], df["depth"], color="tab:green", linewidth=0.9)
        axes[2].set_ylabel("Depth (m)")
        axes[2].invert_yaxis()
        axes[2].set_xlabel("Time of day (GMT)")

        axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        fig.autofmt_xdate()

        fig.suptitle(name, fontsize=13, fontweight="bold")
        fig.tight_layout()

        out_path = os.path.join(OUTPUT_DIR, f"{name}_timeseries.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
