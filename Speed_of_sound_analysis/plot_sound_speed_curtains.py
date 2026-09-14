"""
Build interpolated 3D "curtain" (fence diagram) plots of sound speed
(computed from CTD data via the Mackenzie 1981 formula) across multiple
crossing AUV tracks -- same technique as the salinity curtain script,
applied to sound speed instead.

DATA SOURCE
-----------
In Neptus MRA, for each mission: right-click -> "Export (Georeferenced)
CTD to CSV". Save one CSV per track into a folder.

USAGE
-----
    python plot_sound_speed_curtains.py <tracks_dir>

OUTPUTS
-------
    sound_speed_curtains.png                        combined, all tracks
    sound_speed_curtains_per_track/<name>.png        one per track

Requires: pandas, numpy, matplotlib, scipy
"""

import sys
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRACKS_DIR = sys.argv[1] if len(sys.argv) > 1 else "tracks"

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

GRID_ALONG_TRACK = 200
GRID_DEPTH = 80

OUTPUT_PNG = "sound_speed_curtains.png"
PER_TRACK_DIR = "sound_speed_curtains_per_track"


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

def load_track(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if list(df.columns) == RAW_HEADER and len(df.columns) == len(CORRECTED_ORDER):
        df.columns = CORRECTED_ORDER

    lat_col = "lat (corrected)" if "lat (corrected)" in df.columns else "latitude"
    lon_col = "lon (corrected)" if "lon (corrected)" in df.columns else "longitude"

    out = df[[lat_col, lon_col, "depth", "temperature", "salinity"]].rename(
        columns={lat_col: "lat", lon_col: "lon"}
    )
    for col in ["lat", "lon", "depth", "temperature", "salinity"]:
        if out[col].dtype == object:
            out[col] = out[col].astype(str).str.replace(",", ".", regex=False)
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna()

    out["value"] = mackenzie_sound_speed(
        out["temperature"].values, out["salinity"].values, out["depth"].values
    )
    return out


def project_to_meters(df, lat0, lon0):
    df = df.copy()
    df["x"] = (df["lon"] - lon0) * np.cos(np.radians(lat0)) * 111_320
    df["y"] = (df["lat"] - lat0) * 110_540
    return df


def build_curtain(df):
    dx = np.diff(df["x"].values, prepend=df["x"].values[0])
    dy = np.diff(df["y"].values, prepend=df["y"].values[0])
    s = np.cumsum(np.sqrt(dx**2 + dy**2))

    s_grid = np.linspace(s.min(), s.max(), GRID_ALONG_TRACK)
    depth_grid = np.linspace(df["depth"].min(), df["depth"].max(), GRID_DEPTH)
    S, D = np.meshgrid(s_grid, depth_grid)

    V = griddata((s, df["depth"].values), df["value"].values, (S, D), method="linear")

    x_of_s = np.interp(s_grid, s, df["x"].values)
    y_of_s = np.interp(s_grid, s, df["y"].values)
    X = np.tile(x_of_s, (GRID_DEPTH, 1))
    Y = np.tile(y_of_s, (GRID_DEPTH, 1))
    Z = -D

    return X, Y, Z, V


def draw_curtains(curtains, vmin, vmax, title):
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    norm = plt.Normalize(vmin, vmax)
    cmap = cm.viridis

    for name, (X, Y, Z, V) in curtains.items():
        facecolors = cmap(norm(V))
        facecolors[np.isnan(V)] = (0, 0, 0, 0)
        ax.plot_surface(X, Y, Z, facecolors=facecolors, rstride=1, cstride=1,
                         linewidth=0, antialiased=False, shade=False)

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_zlabel("Depth (m)")
    ax.set_title(title, fontsize=12, fontweight="bold")

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.6, label="Sound speed (m/s)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    csv_files = sorted(glob.glob(os.path.join(TRACKS_DIR, "*.csv")))
    if not csv_files:
        print(f"No CSV files found in '{TRACKS_DIR}'.")
        return

    print(f"Found {len(csv_files)} CTD track file(s) in '{TRACKS_DIR}'.")

    raw_tracks = {}
    for path in csv_files:
        name = os.path.splitext(os.path.basename(path))[0]
        df = load_track(path)
        if len(df) < 10:
            print(f"  [skipped] {name}: too few valid rows ({len(df)})")
            continue
        raw_tracks[name] = df
        print(f"  loaded {name}: {len(df)} rows, "
              f"sound speed {df['value'].min():.1f}-{df['value'].max():.1f} m/s")

    if not raw_tracks:
        print("No tracks loaded.")
        return

    all_lat = pd.concat([t["lat"] for t in raw_tracks.values()])
    all_lon = pd.concat([t["lon"] for t in raw_tracks.values()])
    lat0, lon0 = all_lat.mean(), all_lon.mean()

    for name in raw_tracks:
        raw_tracks[name] = project_to_meters(raw_tracks[name], lat0, lon0)

    all_values = pd.concat([t["value"] for t in raw_tracks.values()])
    vmin, vmax = all_values.min(), all_values.max()

    curtains = {name: build_curtain(df) for name, df in raw_tracks.items()}

    # --- Combined plot, all tracks ---
    fig = draw_curtains(curtains, vmin, vmax, "Interpolated sound speed curtains across tracks")
    fig.savefig(OUTPUT_PNG, dpi=200)
    plt.close(fig)
    print(f"\nSaved combined curtain plot -> {OUTPUT_PNG}")

    # --- Per-track plots ---
    os.makedirs(PER_TRACK_DIR, exist_ok=True)
    for name, curtain in curtains.items():
        fig = draw_curtains({name: curtain}, vmin, vmax, f"Sound speed curtain -- {name}")
        out_path = os.path.join(PER_TRACK_DIR, f"sound_speed_curtain_{name}.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
