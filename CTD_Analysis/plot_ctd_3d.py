"""
Build a 3D "curtain"/fence diagram of CTD data across multiple crossing
AUV tracks, using each track's georeferenced position + depth + a CTD
variable (salinity or temperature).

DATA SOURCE
-----------
In Neptus MRA, for each mission: right-click -> "Export (Georeferenced)
CTD to CSV". Save one CSV per track into a folder.

USAGE
-----
    python plot_ctd_3d.py <tracks_dir> [variable]

    <tracks_dir>  folder containing one CSV per track
    [variable]    which column to color by: "salinity" (default) or
                  "temperature" -- adjust VAR_CANDIDATES below if your
                  export uses different column names.

OUTPUTS
-------
    ctd_3d.png    static 3D scatter (matplotlib), colored by the chosen
                  variable, one color scale shared across all tracks
    ctd_3d.html   interactive 3D scatter (plotly) -- rotate/zoom/hover
                  to inspect crossing points directly
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
VARIABLE = sys.argv[2] if len(sys.argv) > 2 else "salinity"

LAT_CANDIDATES = ["lat (corrected)", "lat", "latitude", "Latitude", "Latitude (deg)"]
LON_CANDIDATES = ["lon (corrected)", "lon", "lng", "longitude", "Longitude", "Longitude (deg)"]
DEPTH_CANDIDATES = ["depth", "Depth", "depth (m)", "Depth (m)"]
VAR_CANDIDATES = {
    "salinity": ["salinity", "Salinity", "Salinity (PSU)", "psu"],
    "temperature": ["temperature", "Temperature", "Temperature (C)", "temp"],
}

OUTPUT_PNG = "ctd_3d.png"
OUTPUT_HTML = "ctd_3d.html"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    lowered = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lowered:
            return lowered[c.lower()]
    return None


def load_track(csv_path, var_name):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # strip leading/trailing spaces in headers

    # KNOWN QUIRK in this exporter: the CSV has one more real data field per
    # row than there are header names, so pandas silently uses the true
    # first field as a hidden index and shifts every header one column to
    # the right relative to the actual data. We correct that here by
    # relabeling columns positionally with the real field order.
    CORRECTED_ORDER = [
        "gmt time", "latitude", "longitude", "lat (corrected)",
        "lon (corrected)", "altitude", "depth", "medium",
        "conductivity", "temperature", "salinity", "extra",
    ]
    if list(df.columns) == [
        "# timestamp", "gmt time", "latitude", "longitude", "lat (corrected)",
        "lon (corrected)", "altitude", "depth", "medium", "conductivity",
        "temperature", "salinity",
    ] and len(df.columns) == len(CORRECTED_ORDER):
        df.columns = CORRECTED_ORDER
        print("    (applied known column-shift correction for this exporter)")

    # Diagnostic: show the actual column list and first raw row exactly as
    # pandas parsed them, so we can spot any misalignment before selecting.
    print(f"    columns ({len(df.columns)}): {df.columns.tolist()}")
    print(f"    first row: {df.iloc[0].tolist()}")

    lat_col = find_column(df, LAT_CANDIDATES)
    lon_col = find_column(df, LON_CANDIDATES)
    depth_col = find_column(df, DEPTH_CANDIDATES)
    var_col = find_column(df, VAR_CANDIDATES[var_name])

    missing = [name for name, col in
               [("lat", lat_col), ("lon", lon_col), ("depth", depth_col), (var_name, var_col)]
               if col is None]
    if missing:
        raise ValueError(
            f"Missing column(s) {missing} in {csv_path}. "
            f"Columns present: {list(df.columns)}. "
            f"Add the correct names to the *_CANDIDATES dicts above."
        )

    out = df[[lat_col, lon_col, depth_col, var_col]].rename(
        columns={lat_col: "lat", lon_col: "lon", depth_col: "depth", var_col: "value"}
    )

    # Diagnostic: show a couple of raw values before any conversion, so if
    # something still goes wrong we can see exactly what's in the file.
    print(f"    sample raw values -- depth: {out['depth'].head(3).tolist()}, "
          f"{var_name}: {out['value'].head(3).tolist()}")

    for col in ["lat", "lon", "depth", "value"]:
        if out[col].dtype == object:
            # Handle comma-as-decimal-separator (common in Norwegian/EU
            # locale exports), e.g. "10,5" instead of "10.5".
            out[col] = out[col].astype(str).str.replace(",", ".", regex=False)
        out[col] = pd.to_numeric(out[col], errors="coerce")

    n_before = len(out)
    out = out.dropna()
    n_after = len(out)
    print(f"    {n_after}/{n_before} rows kept after numeric conversion")

    return out


def project_to_meters(df, lat0, lon0):
    """Simple flat-earth projection -- fine at fjord/km scale."""
    df = df.copy()
    df["x"] = (df["lon"] - lon0) * np.cos(np.radians(lat0)) * 111_320
    df["y"] = (df["lat"] - lat0) * 110_540
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if VARIABLE not in VAR_CANDIDATES:
        print(f"Unknown variable '{VARIABLE}'. Choose from: {list(VAR_CANDIDATES)}")
        return

    csv_files = sorted(glob.glob(os.path.join(TRACKS_DIR, "*.csv")))
    if not csv_files:
        print(f"No CSV files found in '{TRACKS_DIR}'.")
        return

    print(f"Found {len(csv_files)} CTD track file(s) in '{TRACKS_DIR}'.")

    tracks = {}
    for path in csv_files:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            df = load_track(path, VARIABLE)
            if len(df) == 0:
                print(f"  [skipped] {name}: 0 valid rows after cleaning -- check sample values above")
                continue
            tracks[name] = df
        except ValueError as e:
            print(f"  [skipped] {name}: {e}")

    if not tracks:
        print("No tracks loaded -- check column names above.")
        return

    # Common local origin so all tracks share the same x/y coordinate system
    all_lat = pd.concat([t["lat"] for t in tracks.values()])
    all_lon = pd.concat([t["lon"] for t in tracks.values()])
    lat0, lon0 = all_lat.mean(), all_lon.mean()

    for name in tracks:
        tracks[name] = project_to_meters(tracks[name], lat0, lon0)

    all_values = pd.concat([t["value"] for t in tracks.values()])
    vmin, vmax = all_values.min(), all_values.max()

    # --- Static 3D scatter (matplotlib) ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    sc = None
    for name, df in tracks.items():
        sc = ax.scatter(df["x"], df["y"], -df["depth"], c=df["value"],
                         cmap="viridis", vmin=vmin, vmax=vmax, s=8, label=name)

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_zlabel("Depth (m)")
    ax.set_title(f"3D CTD {VARIABLE} across tracks")
    fig.colorbar(sc, ax=ax, shrink=0.6, label=VARIABLE)
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.15, 1.0))
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=200)
    print(f"Saved static 3D plot -> {OUTPUT_PNG}")

    # --- Interactive 3D scatter (plotly) ---
    try:
        import plotly.graph_objects as go

        fig3 = go.Figure()
        for name, df in tracks.items():
            fig3.add_trace(go.Scatter3d(
                x=df["x"], y=df["y"], z=-df["depth"],
                mode="markers",
                marker=dict(size=3, color=df["value"], colorscale="Viridis",
                            cmin=vmin, cmax=vmax,
                            colorbar=dict(title=VARIABLE) if name == list(tracks)[0] else None),
                name=name,
                text=name,
            ))

        fig3.update_layout(
            title=f"3D CTD {VARIABLE} across tracks (rotate/zoom, hover for track)",
            scene=dict(xaxis_title="Easting (m)", yaxis_title="Northing (m)",
                       zaxis_title="Depth (m)"),
        )
        fig3.write_html(OUTPUT_HTML)
        print(f"Saved interactive 3D plot -> {OUTPUT_HTML} (open in a browser)")
    except ImportError:
        print("(Optional) install 'plotly' for an interactive 3D view: "
              "pip install plotly --break-system-packages")


if __name__ == "__main__":
    main()
