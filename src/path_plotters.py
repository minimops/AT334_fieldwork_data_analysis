from pathlib import Path

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box


def load_neptus_paths(path):
    paths = gpd.read_file(Path(path))

    if paths.crs is None:
        paths = paths.set_crs("EPSG:25833")

    return paths


def describe_path_features(paths):
    columns = [
        column
        for column in ["name", "description", "geometry"]
        if column in paths.columns
    ]

    return paths[columns].copy()


def select_path(paths, name):

    picked = paths["name"].str.contains(name)
    executed = paths.loc[picked].copy()

    if executed.empty:
        raise ValueError("no path could be identified")

    return executed


def plot_mission_path(paths, bounds=None, output_file=None):
    fig, ax = plt.subplots(figsize=(9, 9))
    colors = ["#ff3b30", "#00d4ff", "#ffd60a", "#bf5af2"]

    for index, (label, path) in enumerate(paths.items()):
        path.to_crs("EPSG:25833").plot(ax=ax, color=colors[index % len(colors)], linewidth=2.5, label=label, zorder=2)

    if bounds is not None:
        map_bounds = gpd.GeoSeries([box(*bounds)], crs="EPSG:4326").to_crs("EPSG:25833").total_bounds

        ax.set_xlim(map_bounds[0], map_bounds[2])
        ax.set_ylim(map_bounds[1], map_bounds[3])

    for label, path in paths.items():
        print(label, path.to_crs("EPSG:4326").total_bounds)

    cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, crs="EPSG:25833", zorder=1)

    ax.legend()
    ax.set_axis_off()

    if output_file is not None:
        fig.savefig(output_file, dpi=300, bbox_inches="tight")

    return fig, ax
