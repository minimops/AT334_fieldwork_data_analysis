"""
Generate polished CTD depth-profile plots from the same georeferenced
CTD CSV exports used for the 3D/curtain scripts.

For each track: a classic oceanographic profile plot (temperature and
salinity vs. depth, twin x-axes, depth inverted so the surface is at
the top).

Across all tracks: two overlay comparison plots (salinity vs. depth,
temperature vs. depth) with every track on the same axes, so you can
compare tracks/crossings at a glance.

DATA SOURCE
-----------
In Neptus MRA, for each mission: right-click -> "Export (Georeferenced)
CTD to CSV". Save one CSV per track into a folder.

USAGE
-----
    python plot_ctd_profiles.py <tracks_dir>

OUTPUTS
-------
    ctd_profiles/<track_name>_profile.png   one per track
    ctd_overlay_salinity.png                all tracks, salinity vs depth
    ctd_overlay_temperature.png             all tracks, temperature vs depth
"""

import sys
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRACKS_DIR = sys.argv[1] if len(sys.argv) > 1 else "tracks"
PROFILE_DIR = "ctd_profiles"

# Known column-shift quirk in the Neptus "Export (Georeferenced) CTD to
# CSV" exporter: 13 real data fields but only 12 header names, so every
# header is shifted one column relative to the actual data.
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

# Consistent, distinguishable colors across all plots for each track
COLOR_CYCLE = plt.get_cmap("tab10").colors

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "font.size": 11,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_time_to_seconds(time_series):
    """Convert 'HH:MM:SS.sss' strings (with possible leading space) into
    seconds-since-midnight floats."""
    def _parse(t):
        try:
            h, m, s = str(t).strip().split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
        except (ValueError, AttributeError):
            return np.nan
    return time_series.apply(_parse)


def load_track(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if list(df.columns) == RAW_HEADER and len(df.columns) == len(CORRECTED_ORDER):
        df.columns = CORRECTED_ORDER

    out = df[["gmt time", "depth", "temperature", "salinity"]].copy()
    out = out.rename(columns={"gmt time": "time_str"})

    for col in ["depth", "temperature", "salinity"]:
        if out[col].dtype == object:
            out[col] = out[col].astype(str).str.replace(",", ".", regex=False)
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["seconds"] = parse_time_to_seconds(out["time_str"])
    out = out.dropna(subset=["depth", "temperature", "salinity", "seconds"])
    out["elapsed_min"] = (out["seconds"] - out["seconds"].min()) / 60.0

    return out


def style_profile_axis(ax_depth):
    ax_depth.invert_yaxis()
    ax_depth.set_ylabel("Depth (m)")
    ax_depth.grid(True, alpha=0.3, linestyle="--")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    csv_files = sorted(glob.glob(os.path.join(TRACKS_DIR, "*.csv")))
    if not csv_files:
        print(f"No CSV files found in '{TRACKS_DIR}'.")
        return

    print(f"Found {len(csv_files)} CTD track file(s) in '{TRACKS_DIR}'.")

    tracks = {}
    for path in csv_files:
        name = os.path.splitext(os.path.basename(path))[0]
        df = load_track(path)
        if len(df) < 5:
            print(f"  [skipped] {name}: too few valid rows ({len(df)})")
            continue
        tracks[name] = df
        print(f"  loaded {name}: {len(df)} rows")

    if not tracks:
        print("No tracks loaded.")
        return

    os.makedirs(PROFILE_DIR, exist_ok=True)
    SALINITY_DIR = "ctd_salinity_depth"
    os.makedirs(SALINITY_DIR, exist_ok=True)

    # --- Per-track standalone depth vs. salinity plot, colored by time ---
    for name, df in tracks.items():
        fig, ax = plt.subplots(figsize=(6.5, 7))

        ax.plot(df["salinity"], df["depth"], color="0.75", linewidth=0.6, zorder=1)
        sc = ax.scatter(df["salinity"], df["depth"], c=df["elapsed_min"],
                         cmap="plasma", s=10, zorder=2)

        ax.set_xlabel("Salinity (PSU)")
        style_profile_axis(ax)
        ax.set_title(f"Salinity vs. depth -- {name}", fontsize=12, fontweight="bold")
        cbar = fig.colorbar(sc, ax=ax, shrink=0.9)
        cbar.set_label("Elapsed time (min)")

        fig.tight_layout()
        out_path = os.path.join(SALINITY_DIR, f"{name}_salinity_depth.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"  saved -> {out_path}")

    # --- Per-track classic CTD profile (temperature + salinity vs depth) ---
    for i, (name, df) in enumerate(tracks.items()):
        color_t = "tab:red"
        color_s = "tab:blue"

        fig, ax_t = plt.subplots(figsize=(5, 7))
        ax_s = ax_t.twiny()

        ax_t.plot(df["temperature"], df["depth"], color=color_t, linewidth=1.2)
        ax_t.set_xlabel("Temperature (°C)", color=color_t)
        ax_t.tick_params(axis="x", labelcolor=color_t)

        ax_s.plot(df["salinity"], df["depth"], color=color_s, linewidth=1.2)
        ax_s.set_xlabel("Salinity (PSU)", color=color_s)
        ax_s.tick_params(axis="x", labelcolor=color_s)

        style_profile_axis(ax_t)
        ax_t.set_title(name, fontsize=12, fontweight="bold")

        fig.tight_layout()
        out_path = os.path.join(PROFILE_DIR, f"{name}_profile.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"  saved -> {out_path}")

    # --- Overlay comparison: salinity vs depth, all tracks ---
    fig, ax = plt.subplots(figsize=(6, 8))
    for i, (name, df) in enumerate(tracks.items()):
        color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
        ax.plot(df["salinity"], df["depth"], color=color, linewidth=1.2, label=name)
    ax.set_xlabel("Salinity (PSU)")
    style_profile_axis(ax)
    ax.set_title("Salinity vs. depth -- all tracks")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig("ctd_overlay_salinity.png", dpi=200)
    plt.close(fig)
    print("\nSaved overlay -> ctd_overlay_salinity.png")

    # --- Overlay comparison: temperature vs depth, all tracks ---
    fig, ax = plt.subplots(figsize=(6, 8))
    for i, (name, df) in enumerate(tracks.items()):
        color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
        ax.plot(df["temperature"], df["depth"], color=color, linewidth=1.2, label=name)
    ax.set_xlabel("Temperature (°C)")
    style_profile_axis(ax)
    ax.set_title("Temperature vs. depth -- all tracks")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig("ctd_overlay_temperature.png", dpi=200)
    plt.close(fig)
    print("Saved overlay -> ctd_overlay_temperature.png")


if __name__ == "__main__":
    main()
