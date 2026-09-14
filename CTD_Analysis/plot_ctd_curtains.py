"""
Build interpolated 3D "curtain" (fence diagram) plots of CTD data across
multiple crossing AUV tracks: each track becomes a smooth vertical sheet
(distance-along-track vs. depth), colored by salinity or temperature,
positioned in true 3D space at its real geographic path.

DATA SOURCE
-----------
In Neptus MRA, for each mission: right-click -> "Export (Georeferenced)
CTD to CSV". Save one CSV per track into a folder.

USAGE
-----
    python plot_ctd_curtains.py <tracks_dir> [variable]

    <tracks_dir>  folder containing one CSV per track
    [variable]    "salinity" (default) or "temperature"

OUTPUTS
-------
    ctd_curtains.png    static 3D curtain plot (matplotlib)
    ctd_curtains.html   interactive 3D curtain plot (plotly) -- rotate,
                         zoom, and inspect where curtains cross

Requires: pandas, numpy, matplotlib, scipy, (optional) plotly
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
VARIABLE = sys.argv[2] if len(sys.argv) > 2 else "salinity"

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

GRID_ALONG_TRACK = 200   # resolution along the track direction
GRID_DEPTH = 80           # resolution in depth

OUTPUT_PNG = "ctd_curtains.png"
OUTPUT_HTML = "ctd_curtains.html"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_track(csv_path, var_name):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if list(df.columns) == RAW_HEADER and len(df.columns) == len(CORRECTED_ORDER):
        df.columns = CORRECTED_ORDER

    lat_col = "lat (corrected)" if "lat (corrected)" in df.columns else "latitude"
    lon_col = "lon (corrected)" if "lon (corrected)" in df.columns else "longitude"

    out = df[[lat_col, lon_col, "depth", var_name]].rename(
        columns={lat_col: "lat", lon_col: "lon", var_name: "value"}
    )
    for col in ["lat", "lon", "depth", "value"]:
        if out[col].dtype == object:
            out[col] = out[col].astype(str).str.replace(",", ".", regex=False)
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out.dropna()


def project_to_meters(df, lat0, lon0):
    df = df.copy()
    df["x"] = (df["lon"] - lon0) * np.cos(np.radians(lat0)) * 111_320
    df["y"] = (df["lat"] - lat0) * 110_540
    return df


def build_curtain(df, vmin, vmax):
    """
    Given one track's (x, y, depth, value) point cloud, build a smooth
    interpolated curtain: a grid over (along-track distance, depth),
    with each grid point's true (x, y, depth) position and interpolated
    value, ready to render as a 3D surface.
    """
    # Along-track distance: cumulative straight-line distance between
    # consecutive samples (data is already in time order in the CSV).
    dx = np.diff(df["x"].values, prepend=df["x"].values[0])
    dy = np.diff(df["y"].values, prepend=df["y"].values[0])
    s = np.cumsum(np.sqrt(dx**2 + dy**2))

    s_grid = np.linspace(s.min(), s.max(), GRID_ALONG_TRACK)
    depth_grid = np.linspace(df["depth"].min(), df["depth"].max(), GRID_DEPTH)
    S, D = np.meshgrid(s_grid, depth_grid)

    # Interpolate the scalar field onto the (s, depth) grid
    V = griddata((s, df["depth"].values), df["value"].values, (S, D), method="linear")

    # Interpolate true x, y position as a function of along-track distance s
    # (monotonic by construction), then broadcast across the depth axis.
    x_of_s = np.interp(s_grid, s, df["x"].values)
    y_of_s = np.interp(s_grid, s, df["y"].values)
    X = np.tile(x_of_s, (GRID_DEPTH, 1))
    Y = np.tile(y_of_s, (GRID_DEPTH, 1))
    Z = -D  # depth increases downward -> negative Z

    return X, Y, Z, V


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
        df = load_track(path, VARIABLE)
        if len(df) < 10:
            print(f"  [skipped] {name}: too few valid rows ({len(df)})")
            continue
        raw_tracks[name] = df
        print(f"  loaded {name}: {len(df)} rows")

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

    curtains = {}
    for name, df in raw_tracks.items():
        curtains[name] = build_curtain(df, vmin, vmax)

    # --- Static plot (matplotlib) ---
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    norm = plt.Normalize(vmin, vmax)
    cmap = cm.viridis

    for name, (X, Y, Z, V) in curtains.items():
        facecolors = cmap(norm(V))
        # Fully transparent where interpolation had no data (NaN)
        facecolors[np.isnan(V)] = (0, 0, 0, 0)
        ax.plot_surface(X, Y, Z, facecolors=facecolors, rstride=1, cstride=1,
                         linewidth=0, antialiased=False, shade=False)

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_zlabel("Depth (m)")
    ax.set_title(f"Interpolated {VARIABLE} curtains across tracks")

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.6, label=VARIABLE)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=200)
    print(f"\nSaved static curtain plot -> {OUTPUT_PNG}")

    # --- Interactive plot (plotly) ---
    try:
        import plotly.graph_objects as go

        fig3 = go.Figure()
        for i, (name, (X, Y, Z, V)) in enumerate(curtains.items()):
            fig3.add_trace(go.Surface(
                x=X, y=Y, z=Z, surfacecolor=V,
                cmin=vmin, cmax=vmax, colorscale="Viridis",
                showscale=(i == 0),
                colorbar=dict(title=VARIABLE) if i == 0 else None,
                name=name,
            ))

        fig3.update_layout(
            title=f"Interpolated {VARIABLE} curtains across tracks (rotate/zoom)",
            scene=dict(xaxis_title="Easting (m)", yaxis_title="Northing (m)",
                       zaxis_title="Depth (m)"),
        )
        fig3.write_html(OUTPUT_HTML)
        print(f"Saved interactive curtain plot -> {OUTPUT_HTML}")
    except ImportError:
        print("(Optional) install 'plotly' for an interactive view: "
              "pip install plotly --break-system-packages")


if __name__ == "__main__":
    main()
