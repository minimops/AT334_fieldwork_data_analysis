"""
Compute sound speed from CTD data (temperature, salinity, depth) using the
Mackenzie (1981) empirical formula, and plot it as a depth profile per
track, colored by elapsed time, plus an overlay comparison across all
tracks.

SOUND SPEED FORMULA (Mackenzie, 1981)
--------------------------------------
c = 1448.96 + 4.591*T - 5.304e-2*T^2 + 2.374e-4*T^3
    + 1.340*(S - 35) + 1.630e-2*D + 1.675e-7*D^2
    - 1.025e-2*T*(S - 35) - 7.139e-13*T*D^3

where T = temperature (deg C), S = salinity (PSU), D = depth (m).
Valid range: T 2-30 C, S 25-40 PSU, D 0-8000 m -- comfortably covers
shallow fjord conditions.

DATA SOURCE
-----------
In Neptus MRA, for each mission: right-click -> "Export (Georeferenced)
CTD to CSV". Save one CSV per track into a folder.

USAGE
-----
    python plot_sound_speed_profile.py <tracks_dir>

OUTPUTS
-------
    sound_speed_profiles/<track_name>_sound_speed.png   one per track
    sound_speed_overlay.png                              all tracks, one plot
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
OUTPUT_DIR = "sound_speed_profiles"
OVERLAY_PNG = "sound_speed_overlay.png"

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
# Sound speed
# ---------------------------------------------------------------------------

def mackenzie_sound_speed(T, S, D):
    """Mackenzie (1981) sound speed in seawater. T in deg C, S in PSU,
    D (depth) in meters. Returns speed in m/s."""
    return (
        1448.96
        + 4.591 * T
        - 5.304e-2 * T**2
        + 2.374e-4 * T**3
        + 1.340 * (S - 35)
        + 1.630e-2 * D
        + 1.675e-7 * D**2
        - 1.025e-2 * T * (S - 35)
        - 7.139e-13 * T * D**3
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_time_to_seconds(time_series):
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

    out["sound_speed"] = mackenzie_sound_speed(
        out["temperature"].values, out["salinity"].values, out["depth"].values
    )

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

    tracks = {}
    for path in csv_files:
        name = os.path.splitext(os.path.basename(path))[0]
        df = load_track(path)
        if len(df) < 5:
            print(f"  [skipped] {name}: too few valid rows ({len(df)})")
            continue
        tracks[name] = df
        print(f"  loaded {name}: {len(df)} rows, "
              f"sound speed {df['sound_speed'].min():.1f}-{df['sound_speed'].max():.1f} m/s")

    if not tracks:
        print("No tracks loaded.")
        return

    # --- Per-track profile, colored by elapsed time ---
    for name, df in tracks.items():
        fig, ax = plt.subplots(figsize=(6.5, 7))

        ax.plot(df["sound_speed"], df["depth"], color="0.75", linewidth=0.6, zorder=1)
        sc = ax.scatter(df["sound_speed"], df["depth"], c=df["elapsed_min"],
                         cmap="plasma", s=10, zorder=2)

        ax.set_xlabel("Sound speed (m/s)")
        ax.set_ylabel("Depth (m)")
        ax.invert_yaxis()
        ax.set_title(f"Sound speed vs. depth -- {name}", fontsize=12, fontweight="bold")
        cbar = fig.colorbar(sc, ax=ax, shrink=0.9)
        cbar.set_label("Elapsed time (min)")

        fig.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"{name}_sound_speed.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"  saved -> {out_path}")

    # --- Overlay comparison across all tracks ---
    fig, ax = plt.subplots(figsize=(6, 8))
    for i, (name, df) in enumerate(tracks.items()):
        color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
        ax.plot(df["sound_speed"], df["depth"], color=color, linewidth=1.2, label=name)
    ax.set_xlabel("Sound speed (m/s)")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.set_title("Sound speed vs. depth -- all tracks")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(OVERLAY_PNG, dpi=200)
    plt.close(fig)
    print(f"\nSaved overlay -> {OVERLAY_PNG}")


if __name__ == "__main__":
    main()
