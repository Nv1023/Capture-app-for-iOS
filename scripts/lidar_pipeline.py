from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

import numpy as np

if __package__:
    from .train_depth_completion import decode_grayscale_png, paired_frames
else:
    from train_depth_completion import decode_grayscale_png, paired_frames


CAMERA_CONVENTION = "computer_vision_x_right_y_down_z_forward"
POSE_CONVENTION = "recorded_camera_to_world_quaternion"
DEPTH_SCALE_METERS_PER_UNIT = 0.001


def quaternion_rotate(points: np.ndarray, quaternions: np.ndarray) -> np.ndarray:
    vector = quaternions[:, :3]
    scalar = quaternions[:, 3:4]
    return points + 2 * np.cross(vector, np.cross(vector, points) + scalar * points)


def frame_pose_lookup(archive: ZipFile) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    odometry_name = next((name for name in archive.namelist() if name.lower().endswith("/odometry.csv")), None)
    if odometry_name is None:
        raise ValueError("LiDAR archive is missing odometry.csv")
    text = archive.read(odometry_name).decode("utf-8-sig")
    reader = csv.DictReader(text.splitlines(), skipinitialspace=True)
    required = {"frame", "x", "y", "z", "qx", "qy", "qz", "qw", "fx", "fy", "cx", "cy"}
    if not reader.fieldnames or not required.issubset({name.strip() for name in reader.fieldnames}):
        raise ValueError("Odometry CSV is missing pose or per-frame camera-intrinsic columns")
    lookup = {}
    for row in reader:
        row = {key.strip(): value.strip() for key, value in row.items() if key}
        quaternion = np.asarray([float(row[key]) for key in ("qx", "qy", "qz", "qw")], dtype=np.float64)
        norm = np.linalg.norm(quaternion)
        if norm <= 0:
            continue
        translation = np.asarray([float(row[key]) for key in ("x", "y", "z")], dtype=np.float64)
        intrinsics = np.asarray([float(row[key]) for key in ("fx", "fy", "cx", "cy")], dtype=np.float64)
        lookup[row["frame"]] = (translation, quaternion / norm, intrinsics)
    if not lookup:
        raise ValueError("Odometry CSV contains no usable frame poses")
    return lookup


def depth_to_camera_points(
    depth: np.ndarray,
    confidence: np.ndarray,
    intrinsics: np.ndarray,
    depth_scale: float = DEPTH_SCALE_METERS_PER_UNIT,
    pixel_stride: int = 8,
) -> np.ndarray:
    if depth.shape != confidence.shape:
        raise ValueError("Depth and confidence image dimensions do not match")
    if pixel_stride < 1:
        raise ValueError("Pixel stride must be positive")
    fx, fy, cx_rgb, cy_rgb = intrinsics
    height, width = depth.shape
    fx_depth = fx * width / (2 * cx_rgb)
    fy_depth = fy * height / (2 * cy_rgb)
    cx_depth = width / 2
    cy_depth = height / 2
    y, x = np.mgrid[0:height:pixel_stride, 0:width:pixel_stride]
    sampled_depth = depth[y, x] * depth_scale
    sampled_confidence = confidence[y, x]
    valid_confidence = confidence[confidence > 0]
    if not len(valid_confidence):
        return np.empty((0, 3), dtype=np.float64)
    confidence_floor = np.percentile(valid_confidence, 50)
    valid = np.isfinite(sampled_depth) & (sampled_depth > 0) & (sampled_depth <= 10)
    valid &= sampled_confidence >= confidence_floor
    sampled_depth = sampled_depth[valid]
    x = x[valid]
    y = y[valid]
    return np.column_stack((
        (x - cx_depth) * sampled_depth / fx_depth,
        (y - cy_depth) * sampled_depth / fy_depth,
        sampled_depth,
    ))


def reconstruct_point_cloud(
    archive_path: Path,
    frame_stride: int = 20,
    pixel_stride: int = 8,
    depth_scale: float = DEPTH_SCALE_METERS_PER_UNIT,
) -> tuple[np.ndarray, dict]:
    if frame_stride < 1 or pixel_stride < 1 or depth_scale <= 0:
        raise ValueError("Frame stride, pixel stride, and depth scale must be positive")
    try:
        archive = ZipFile(archive_path)
    except BadZipFile as error:
        raise ValueError("Input is not a valid LiDAR ZIP archive") from error
    with archive:
        poses = frame_pose_lookup(archive)
        matrix_name = next((name for name in archive.namelist() if name.lower().endswith("/camera_matrix.csv")), None)
        if matrix_name is None:
            raise ValueError("LiDAR archive is missing camera_matrix.csv")
        matrix = np.loadtxt(archive.open(matrix_name), delimiter=",")
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("Camera matrix must be a finite 3x3 CSV")
        pairs = paired_frames(archive)
        points_by_frame = []
        used_frames = 0
        for index, (depth_name, confidence_name) in enumerate(pairs):
            if index % frame_stride:
                continue
            frame_id = PurePosixPath(depth_name).stem
            if frame_id not in poses:
                continue
            depth = decode_grayscale_png(archive.read(depth_name))
            confidence = decode_grayscale_png(archive.read(confidence_name))
            translation, quaternion, frame_intrinsics = poses[frame_id]
            camera_points = depth_to_camera_points(depth, confidence, frame_intrinsics, depth_scale, pixel_stride)
            if not len(camera_points):
                continue
            world_points = quaternion_rotate(camera_points, np.broadcast_to(quaternion, (len(camera_points), 4)))
            points_by_frame.append(world_points + translation)
            used_frames += 1

    if not points_by_frame:
        raise ValueError("No matching depth frames, confidence maps, and poses produced valid points")
    points = np.concatenate(points_by_frame).astype(np.float32)
    bounds = np.percentile(points, [1, 50, 99], axis=0)
    metadata = {
        "source_archive": archive_path.name,
        "point_count": int(len(points)),
        "frames_used": used_frames,
        "frame_stride": frame_stride,
        "pixel_stride": pixel_stride,
        "depth_scale_meters_per_unit": depth_scale,
        "bounds_percentiles_xyz_m": bounds.round(4).tolist(),
        "coordinate_assumptions": {
            "depth_unit": "millimeters_inferred_from_value_range; not documented in archive",
            "camera_axes": CAMERA_CONVENTION,
            "pose_axes": POSE_CONVENTION,
            "confidence_filter": "per-frame nonzero confidence median",
            "intrinsics": "per-frame RGB intrinsics scaled to depth-image dimensions",
        },
        "accuracy_status": "unvalidated_without_external_ground_truth",
    }
    return points, metadata


def write_point_cloud(points: np.ndarray, metadata: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "world_points.npz", points_xyz_m=points)
    (output_dir / "geometry_ingest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def convex_hull(points: np.ndarray) -> np.ndarray:
    unique = sorted(set(map(tuple, np.asarray(points, dtype=np.float64))))
    if len(unique) < 3:
        return np.empty((0, 2), dtype=np.float64)

    def cross(origin, first, second):
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (second[0] - origin[0])

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def polygon_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    x = polygon[:, 0]
    z = polygon[:, 1]
    return float(abs(np.dot(x, np.roll(z, 1)) - np.dot(z, np.roll(x, 1))) / 2)


def connected_floor_regions(points_xz: np.ndarray, cell_size_m: float = 0.2, minimum_cells: int = 8) -> list[dict]:
    if not len(points_xz):
        return []
    origin = np.floor(np.min(points_xz, axis=0) / cell_size_m) * cell_size_m
    cells = np.floor((points_xz - origin) / cell_size_m).astype(np.int64)
    points_by_cell: dict[tuple[int, int], list[int]] = {}
    for index, cell in enumerate(cells):
        points_by_cell.setdefault((int(cell[0]), int(cell[1])), []).append(index)
    unseen = set(points_by_cell)
    components = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        queue = [start]
        component = []
        while queue:
            cell = queue.pop()
            component.append(cell)
            for delta_x in (-1, 0, 1):
                for delta_z in (-1, 0, 1):
                    neighbor = (cell[0] + delta_x, cell[1] + delta_z)
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        queue.append(neighbor)
        if len(component) < minimum_cells:
            continue
        point_indices = [index for cell in component for index in points_by_cell[cell]]
        region_points = points_xz[point_indices]
        hull = convex_hull(region_points)
        components.append({
            "cell_count": len(component),
            "point_count": len(point_indices),
            "hull_xz_m": hull.round(4).tolist(),
            "observed_hull_area_m2": round(polygon_area(hull), 4) if len(hull) >= 3 else None,
        })
    components.sort(key=lambda component: (-component["cell_count"], component["hull_xz_m"]))
    for index, component in enumerate(components, start=1):
        component["candidate_id"] = f"floor_region_{index:02d}"
    return components


def dominant_height_planes(points: np.ndarray, bin_width: float = 0.05) -> dict:
    height = points[:, 1]
    edges = np.arange(np.floor(height.min() / bin_width) * bin_width,
                      np.ceil(height.max() / bin_width) * bin_width + bin_width,
                      bin_width)
    counts, edges = np.histogram(height, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    median = float(np.median(height))
    lower_bins = np.flatnonzero(centers < median)
    upper_bins = np.flatnonzero(centers > median)
    floor_index = int(lower_bins[np.argmax(counts[lower_bins])]) if len(lower_bins) else None
    ceiling_index = int(upper_bins[np.argmax(counts[upper_bins])]) if len(upper_bins) else None
    floor_y = float(centers[floor_index]) if floor_index is not None else None
    ceiling_y = float(centers[ceiling_index]) if ceiling_index is not None else None
    return {
        "floor_y_candidate_m": floor_y,
        "floor_bin_support": int(counts[floor_index]) if floor_index is not None else 0,
        "ceiling_y_candidate_m": ceiling_y,
        "ceiling_bin_support": int(counts[ceiling_index]) if ceiling_index is not None else 0,
        "ceiling_height_candidate_m": round(ceiling_y - floor_y, 4) if floor_y is not None and ceiling_y is not None and ceiling_y > floor_y else None,
        "histogram_bin_width_m": bin_width,
    }


def analyze_floor_geometry(points: np.ndarray) -> dict:
    planes = dominant_height_planes(points)
    floor_y = planes["floor_y_candidate_m"]
    if floor_y is None:
        return {"surface_status": "not_found", "planes": planes, "observed_floor_hull_xz_m": [], "observed_floor_hull_area_m2": None}
    band = max(0.075, planes["histogram_bin_width_m"] * 1.5)
    floor_points = points[np.abs(points[:, 1] - floor_y) <= band][:, [0, 2]]
    if len(floor_points) >= 20:
        lower = np.percentile(floor_points, 1, axis=0)
        upper = np.percentile(floor_points, 99, axis=0)
        floor_points = floor_points[np.all((floor_points >= lower) & (floor_points <= upper), axis=1)]
    regions = connected_floor_regions(floor_points)
    if regions:
        minimum_region_cells = max(8, int(np.ceil(regions[0]["cell_count"] * 0.05)))
        regions = [region for region in regions if region["cell_count"] >= minimum_region_cells]
    hull = np.asarray(regions[0]["hull_xz_m"], dtype=np.float64) if regions else convex_hull(floor_points)
    return {
        "surface_status": "candidate_unvalidated" if len(hull) >= 3 else "insufficient_points",
        "floor_point_count": int(len(floor_points)),
        "planes": planes,
        "floor_region_candidates": regions,
        "observed_floor_hull_xz_m": hull.round(4).tolist(),
        "observed_floor_hull_area_m2": round(polygon_area(hull), 4) if len(hull) >= 3 else None,
        "area_semantics": "convex hull of observed near-floor returns; not a complete room-floor measurement",
        "confidence_interval_m2": None,
    }


def render_geometry_preview(points: np.ndarray, floor_geometry: dict, output_path: Path) -> None:
    horizontal = points[:, [0, 2]]
    minimum = np.percentile(horizontal, 1, axis=0)
    maximum = np.percentile(horizontal, 99, axis=0)
    span = np.maximum(maximum - minimum, 0.1)
    width, height, margin = 960, 700, 52
    scale = min((width - 2 * margin) / span[0], (height - 2 * margin) / span[1])

    def project(xz):
        screen_x = margin + (xz[0] - minimum[0]) * scale
        screen_y = height - margin - (xz[1] - minimum[1]) * scale
        return screen_x, screen_y

    step = max(1, len(horizontal) // 5000)
    circles = []
    for point in horizontal[::step]:
        x, y = project(point)
        circles.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.8"/>')
    hull = floor_geometry.get("observed_floor_hull_xz_m", [])
    polygon = ""
    if len(hull) >= 3:
        coordinates = " ".join(f"{project(point)[0]:.1f},{project(point)[1]:.1f}" for point in hull)
        polygon = f'<polygon points="{coordinates}" fill="#b9ed79" fill-opacity=".15" stroke="#db724d" stroke-width="2" stroke-dasharray="7 5"/>'
    title = "LiDAR point-cloud projection — unvalidated geometry preview"
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">{title}</title><desc id="desc">World-space sensor returns with a dashed candidate hull around observed near-floor points. This is not a measured room plan.</desc>
<rect width="100%" height="100%" fill="#f2f3ed"/><text x="{margin}" y="29" font-family="sans-serif" font-size="15" font-weight="700" fill="#243329">LIDAR GEOMETRY PREVIEW</text>
<text x="{margin}" y="47" font-family="monospace" font-size="10" fill="#8b6243">NOT A MEASURED FLOORPLAN · DEPTH SCALE / POSES UNVERIFIED</text>
<g fill="#315b42" fill-opacity=".53">{''.join(circles)}</g>{polygon}
<g font-family="monospace" font-size="10" fill="#627064"><text x="{width-margin-130}" y="{height-margin+28}">world X / Z projection</text></g>
<g transform="translate({width-margin-22} {margin+24})" stroke="#415f44" stroke-width="2"><path d="M0 25V0M0 0L-4 6M0 0L4 6M0 25H25M25 25L19 21M25 25L19 29" fill="none"/></g>
</svg>'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def build_partial_output(points: np.ndarray, metadata: dict, output_dir: Path) -> dict:
    floor_geometry = analyze_floor_geometry(points)
    render_path = output_dir / "geometry-preview.svg"
    render_geometry_preview(points, floor_geometry, render_path)
    return {
        "schema_version": "0.2.0-experimental",
        "input_tier": "lidar",
        "processing_status": "partial_geometry_unvalidated",
        "source_archive": metadata["source_archive"],
        "stages": {
            "lidar_reconstruction": {"status": "completed_sparse_pose_projection", "point_count": metadata["point_count"]},
            "common_geometry": {"status": "partial", "representation": "world_space_point_cloud", "artifact": "world_points.npz"},
            "room_detection": {
                "status": "floor_region_candidates_not_room_labels",
                "candidate_count": len(floor_geometry.get("floor_region_candidates", [])),
            },
            "room_connectivity": {
                "status": "single_scan_adjacency_unknown",
                "candidate_nodes": [region["candidate_id"] for region in floor_geometry.get("floor_region_candidates", [])],
                "edges": [],
            },
            "global_alignment": {"status": "odometry_only_drift_uncorrected", "drift_ablation": "not_run"},
            "floorplan_graph": {
                "status": "observed_floor_region_graph_only",
                "candidate_regions": floor_geometry.get("floor_region_candidates", []),
            },
            "damage_detection": {"status": "not_implemented", "labels_available": False},
            "concealed_damage": {"status": "not_implemented"},
            "confidence_calibration": {"status": "not_calibrated", "intervals": None},
            "render": {"status": "geometry_preview_only", "path": render_path.name},
        },
        "coordinate_assumptions": metadata["coordinate_assumptions"],
        "floor_geometry_candidate": floor_geometry,
        "dimensions": {
            "wall_lengths_m": [],
            "ceiling_height_m": floor_geometry["planes"]["ceiling_height_candidate_m"],
            "confidence_interval_m": None,
            "status": "candidate_unvalidated",
        },
        "damage_regions": [],
        "limitations": [
            "The recorded odometry is used without loop closure or drift correction.",
            "Depth units and camera-axis convention are inferred, not confirmed by capture metadata.",
            "Floor and ceiling candidates come from height histograms, not validated plane/room segmentation.",
            "This archive has no laser/tape ground truth, damage labels, or multi-room adjacency labels.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct a sparse LiDAR point cloud and geometry preview from a ZIP.")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/lidar"))
    parser.add_argument("--frame-stride", type=int, default=20)
    parser.add_argument("--pixel-stride", type=int, default=8)
    parser.add_argument("--depth-scale", type=float, default=DEPTH_SCALE_METERS_PER_UNIT)
    args = parser.parse_args()
    try:
        points, metadata = reconstruct_point_cloud(args.archive, args.frame_stride, args.pixel_stride, args.depth_scale)
    except (OSError, ValueError, BadZipFile) as error:
        parser.error(str(error))
    write_point_cloud(points, metadata, args.output_dir)
    output = build_partial_output(points, metadata, args.output_dir)
    (args.output_dir / "pipeline-output.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"ingest": metadata, "processing_status": output["processing_status"], "stages": output["stages"], "floor_geometry_candidate": output["floor_geometry_candidate"]}, indent=2))


if __name__ == "__main__":
    main()