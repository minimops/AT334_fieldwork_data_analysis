from pathlib import Path

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
from shapely.geometry import LineString



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


def plot_mission_path(paths=None, areas=None, coverage=None, bounds=None, output_file=None):
    fig, ax = plt.subplots(figsize=(9, 9))
    colors = ["#ff3b30", "#00d4ff", "#ffd60a", "#bf5af2"]
    accepted_color = "#58ff0a"

    paths = paths or {}
    areas = areas or {}
    coverage = coverage or {}

    for area in coverage.values():
        area.to_crs("EPSG:25833").plot( ax=ax, facecolor=accepted_color, edgecolor="none", alpha=0.4, zorder=2)

    for index, (label, path) in enumerate(paths.items()):
        path.to_crs("EPSG:25833").plot(ax=ax, color=colors[index % len(colors)], linewidth=2.5, label=label, zorder=3)# , alpha=0.5)

    for index, (label, area) in enumerate(areas.items()):
        area.to_crs("EPSG:25833").plot(ax=ax, facecolor=accepted_color, edgecolor=accepted_color, alpha=0.7, linewidth=1, zorder=4)

    if bounds is not None:
        map_bounds = gpd.GeoSeries([box(*bounds)], crs="EPSG:4326").to_crs("EPSG:25833").total_bounds

        ax.set_xlim(map_bounds[0], map_bounds[2])
        ax.set_ylim(map_bounds[1], map_bounds[3])

    for label, path in paths.items():
        print(label, path.to_crs("EPSG:4326").total_bounds)

    cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, crs="EPSG:25833", zorder=1)

    ax.legend()
    ax.set_axis_off()
    
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    if output_file is not None:
        fig.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0)

    return fig, ax


def create_usable_area(path, sections, buffer_metres=5):
    coordinates = list(path.geometry.iloc[0].coords)

    lines = [
        LineString([coordinates[start], coordinates[end]])
        for start, end in sections
    ]
    lines = gpd.GeoSeries(lines, crs=path.crs).to_crs("EPSG:25833")
    area = lines.buffer(buffer_metres, cap_style="flat").union_all()

    return gpd.GeoDataFrame(geometry=[area], crs="EPSG:25833").to_crs("EPSG:4326")


def create_path_sections(path, sections):
    coordinates = list(path.geometry.iloc[0].coords)

    lines = [LineString(coordinates[start:end + 1]) for start, end in sections]

    return gpd.GeoDataFrame(geometry=lines, crs=path.crs)



def create_path_coverage(path_sections, distance=30):
    projected = path_sections.to_crs("EPSG:25833")

    coverage = projected.buffer(distance, cap_style="flat").union_all()

    return gpd.GeoDataFrame(geometry=[coverage], crs="EPSG:25833").to_crs("EPSG:4326")
