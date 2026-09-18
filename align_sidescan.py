import re
import struct
from collections import Counter
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt
from shapely import contains_xy
from shapely.geometry import LineString


HEADER = struct.Struct("<II4s3sB")
CHANNEL = struct.Struct("<HHfffHHH")


def nmea_coordinate(value, hemisphere):
    value = float(value)
    degrees = int(value // 100)
    minutes = value - degrees * 100
    coordinate = degrees + minutes / 60
    return -coordinate if hemisphere in ("S", "W") else coordinate


def read_sds_metadata(sds_file):
    navigation = []
    pings = []
    packet_counts = Counter()

    with open(sds_file, "rb") as file:
        while header := file.read(HEADER.size):
            size, timestamp, tag, _, _ = HEADER.unpack(header)
            payload = file.read(size)
            packet_counts[tag] += 1

            if tag == b"RNR2":
                pings.append({
                    "timestamp_ms": timestamp,
                    "ping_id": struct.unpack_from("<I", payload, 1)[0],
                })

            elif tag == b"AEMN":
                start = payload.find(b"$MSAUV")
                if start == -1:
                    continue

                sentence = payload[start:].split(b"\x00", 1)[0].decode()
                fields = sentence.split("*", 1)[0].split(",")
                navigation.append({
                    "timestamp_ms": timestamp,
                    "latitude": nmea_coordinate(fields[2], fields[3]),
                    "longitude": nmea_coordinate(fields[4], fields[5]),
                    "speed": float(fields[6]),
                    "course": float(fields[7]),
                    "heading": float(fields[8]),
                    "depth": float(fields[9]),
                    "altitude": float(fields[10]),
                    "roll": float(fields[11]),
                    "pitch": float(fields[12]),
                })

    navigation = pd.DataFrame(navigation).sort_values("timestamp_ms")
    pings = pd.DataFrame(pings).sort_values("timestamp_ms")

    matched = pd.merge_asof(
        pings,
        navigation,
        on="timestamp_ms",
        direction="nearest",
        tolerance=100,
    )

    return matched, packet_counts


def load_sidescan_images(image_directory):
    image_files = sorted(
        Path(image_directory).glob("sss_*.png"),
        key=lambda path: int(re.search(r"\d+", path.stem).group()),
    )
    images = [np.asarray(Image.open(path).convert("RGB")) for path in image_files]
    valid_rows = [np.any(image != 0, axis=(1, 2)) for image in images]

    row_lookup = []
    for image_index, rows in enumerate(valid_rows):
        row_lookup.extend((image_index, row) for row in np.flatnonzero(rows))

    return images, row_lookup


def make_accepted_sections(path_file, sections, buffer_metres=10):
    path = gpd.read_file(path_file).iloc[[0]].to_crs("EPSG:25833")
    coordinates = list(path.geometry.iloc[0].coords)
    lines = [LineString(coordinates[start:end + 1]) for start, end in sections]
    polygons = [line.buffer(buffer_metres, cap_style="flat") for line in lines]
    return path, lines, polygons


def line_heading(line):
    start = line.coords[0]
    end = line.coords[-1]
    return np.degrees(np.arctan2(end[0] - start[0], end[1] - start[1])) % 360


def angle_difference(first, second):
    return np.abs((first - second + 180) % 360 - 180)


def select_survey_pings(pings, polygons, lines, heading_tolerance=20):
    selected = np.zeros(len(pings), dtype=bool)
    section_number = np.zeros(len(pings), dtype=int)

    for index, (polygon, line) in enumerate(zip(polygons, lines), start=1):
        inside = contains_xy(polygon, pings["east"], pings["north"])
        aligned = angle_difference(pings["heading"], line_heading(line)) < heading_tolerance
        use = inside & aligned
        selected |= use
        section_number[use] = index

    result = pings.loc[selected].copy()
    result["section"] = section_number[selected]
    return result


def create_mosaic(
    pings,
    images,
    row_lookup,
    range_metres=40,
    min_ground_range=2,
    max_ground_range=30,
    resolution=0.15,
    column_step=1,
    extent=None,
    cross_track_sign=1,
):
    selected = pings[pings["image_row"] < len(row_lookup)].copy()
    if extent is None:
        padding = max_ground_range + 5
        min_east = selected["east"].min() - padding
        max_east = selected["east"].max() + padding
        min_north = selected["north"].min() - padding
        max_north = selected["north"].max() + padding
    else:
        min_east, max_east, min_north, max_north = extent

    width = int(np.ceil((max_east - min_east) / resolution))
    height = int(np.ceil((max_north - min_north) / resolution))
    max_east = min_east + width * resolution
    max_north = min_north + height * resolution
    sums = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.uint16)
    maxima = np.zeros((height, width), dtype=np.float32)

    for ping in selected.itertuples():
        image_index, image_row = row_lookup[ping.image_row]
        row = images[image_index][image_row]
        columns = np.arange(0, row.shape[0], column_step)
        rgb = row[columns].astype(np.float32) / 255
        intensity = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]

        slant = columns * (2 * range_metres / row.shape[0]) - range_metres
        valid = (np.abs(slant) > ping.altitude) & np.isfinite(ping.altitude)
        ground = np.sign(slant[valid]) * np.sqrt(slant[valid] ** 2 - ping.altitude ** 2)
        ground *= cross_track_sign
        intensity = intensity[valid]

        usable = (np.abs(ground) >= min_ground_range) & (np.abs(ground) <= max_ground_range)
        ground = ground[usable]
        intensity = intensity[usable]

        heading = np.radians(ping.heading)
        pixel_east = ping.east + ground * np.cos(heading)
        pixel_north = ping.north - ground * np.sin(heading)
        x = ((pixel_east - min_east) / resolution).astype(int)
        y = ((pixel_north - min_north) / resolution).astype(int)
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        x = x[inside]
        y = y[inside]
        values = intensity[inside]

        np.add.at(sums, (y, x), values)
        np.add.at(counts, (y, x), 1)
        np.maximum.at(maxima, (y, x), values)

    mean = np.full_like(sums, np.nan)
    mean[counts > 0] = sums[counts > 0] / counts[counts > 0]
    maxima[counts == 0] = np.nan
    extent = (min_east, max_east, min_north, max_north)
    return mean, maxima, counts, extent


def fill_small_gaps(mosaic, max_distance_pixels=2):
    valid = np.isfinite(mosaic)
    distance, nearest = distance_transform_edt(~valid, return_indices=True)
    fill = (~valid) & (distance <= max_distance_pixels)

    result = mosaic.copy()
    result[fill] = mosaic[nearest[0][fill], nearest[1][fill]]
    return result, fill


def write_geotiff(mosaic, extent, resolution, output_file):
    min_east, _, _, max_north = extent
    transform = from_origin(min_east, max_north, resolution, resolution)

    with rasterio.open(
        output_file,
        "w",
        driver="GTiff",
        height=mosaic.shape[0],
        width=mosaic.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:25833",
        transform=transform,
        nodata=np.nan,
        compress="deflate",
    ) as dataset:
        dataset.write(np.flipud(mosaic.astype(np.float32)), 1)


def plot_result(mosaic, extent, path, lines, output_file, title):
    values = mosaic[np.isfinite(mosaic)]
    fig, ax = plt.subplots(figsize=(14, 9), dpi=180)
    ax.set_facecolor("black")
    ax.imshow(
        mosaic,
        extent=extent,
        origin="lower",
        cmap="copper",
        vmin=np.percentile(values, 3),
        vmax=np.percentile(values, 99.5),
        interpolation="nearest",
        zorder=2,
    )
    path.plot(ax=ax, color="white", linewidth=0.6, alpha=0.35, zorder=3)
    gpd.GeoSeries(lines, crs="EPSG:25833").plot(
        ax=ax,
        color="cyan",
        linewidth=0.8,
        alpha=0.8,
        zorder=4,
    )
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_clean_mosaic(mosaic, extent, output_file):
    values = mosaic[np.isfinite(mosaic)]
    fig, ax = plt.subplots(figsize=(14, 9), dpi=180)
    image = ax.imshow(
        mosaic,
        extent=extent,
        origin="lower",
        cmap="copper",
        vmin=np.percentile(values, 3),
        vmax=np.percentile(values, 99.5),
        interpolation="nearest",
    )
    image.cmap.set_bad((0, 0, 0, 0))
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    fig.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    sds_file = root / "upload" / "Data_20260906101108(1).sds"
    image_directory = root / "work" / "sss_images" / "sss_images"
    path_file = root / "upload" / "marie_mission1_path.geojson"
    output_directory = root / "output"
    output_directory.mkdir(exist_ok=True)

    pings, packet_counts = read_sds_metadata(sds_file)
    images, row_lookup = load_sidescan_images(image_directory)
    pings = pings.iloc[:len(row_lookup)].copy()
    pings["image_row"] = np.arange(len(pings))

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:25833", always_xy=True)
    pings["east"], pings["north"] = transformer.transform(
        pings["longitude"].to_numpy(),
        pings["latitude"].to_numpy(),
    )

    sections = [(63, 142), (156, 236), (250, 328)]
    path, lines, polygons = make_accepted_sections(path_file, sections)
    selected = select_survey_pings(pings, polygons, lines)

    mean, maxima, counts, extent = create_mosaic(selected, images, row_lookup)
    corrected, filled = fill_small_gaps(maxima)
    write_geotiff(
        corrected,
        extent,
        0.15,
        output_directory / "mission1_sidescan_corrected_max.tif",
    )
    plot_clean_mosaic(
        corrected,
        extent,
        output_directory / "mission1_sidescan_corrected_max_clean.png",
    )
    plot_result(
        corrected,
        extent,
        path,
        lines,
        output_directory / "mission1_sidescan_corrected_max.png",
        "Mission 1 side-scan mosaic — corrected maximum overlap",
    )

    selected.to_csv(output_directory / "mission1_ping_navigation.csv", index=False)
    print("packets", packet_counts)
    print("image rows", len(row_lookup))
    print("matched pings", len(pings), "missing navigation", pings["latitude"].isna().sum())
    print("selected pings", len(selected), selected.groupby("section").size().to_dict())
    print("altitude", selected["altitude"].describe().to_dict())
    print("headings", selected.groupby("section")["heading"].agg(["mean", "std"]).to_dict("index"))
    print("grid", mean.shape, "covered cells", int(np.isfinite(mean).sum()))
    print("filled cells", int(filled.sum()))


if __name__ == "__main__":
    main()
