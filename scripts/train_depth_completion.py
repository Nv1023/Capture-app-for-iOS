from __future__ import annotations

import argparse
import json
import struct
import zlib
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

import numpy as np


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
FEATURE_NAMES = [
    "depth_nw", "depth_n", "depth_ne", "depth_w", "depth_e", "depth_sw", "depth_s", "depth_se",
    "confidence_nw", "confidence_n", "confidence_ne", "confidence_w", "confidence_center", "confidence_e",
    "confidence_sw", "confidence_s", "confidence_se", "x_normalized", "y_normalized",
]


def decode_grayscale_png(payload: bytes) -> np.ndarray:
    if not payload.startswith(PNG_SIGNATURE):
        raise ValueError("Not a PNG image")
    offset = len(PNG_SIGNATURE)
    compressed = bytearray()
    width = height = bit_depth = color_type = interlace = None
    while offset < len(payload):
        length = struct.unpack_from(">I", payload, offset)[0]
        kind = payload[offset + 4:offset + 8]
        chunk = payload[offset + 8:offset + 8 + length]
        offset += 12 + length
        if kind == b"IHDR":
            width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(">IIBBBBB", chunk)
            if compression != 0 or filtering != 0:
                raise ValueError("Unsupported PNG compression or filter method")
        elif kind == b"IDAT":
            compressed.extend(chunk)
        elif kind == b"IEND":
            break
    if color_type != 0 or bit_depth not in {8, 16} or interlace != 0:
        raise ValueError("Expected a non-interlaced 8-bit or 16-bit grayscale PNG")

    bytes_per_pixel = bit_depth // 8
    row_bytes = width * bytes_per_pixel
    decoded = zlib.decompress(compressed)
    expected_size = height * (row_bytes + 1)
    if len(decoded) != expected_size:
        raise ValueError("PNG scanline data has an unexpected size")

    rows = np.empty((height, row_bytes), dtype=np.uint8)
    previous = np.zeros(row_bytes, dtype=np.uint8)
    cursor = 0
    for row_index in range(height):
        filter_type = decoded[cursor]
        row = np.frombuffer(decoded[cursor + 1:cursor + 1 + row_bytes], dtype=np.uint8).copy()
        cursor += row_bytes + 1
        if filter_type == 1:
            for index in range(bytes_per_pixel, row_bytes):
                row[index] = (int(row[index]) + int(row[index - bytes_per_pixel])) & 255
        elif filter_type == 2:
            row = (row.astype(np.uint16) + previous.astype(np.uint16)).astype(np.uint8)
        elif filter_type == 3:
            for index in range(row_bytes):
                left = int(row[index - bytes_per_pixel]) if index >= bytes_per_pixel else 0
                row[index] = (int(row[index]) + (left + int(previous[index])) // 2) & 255
        elif filter_type == 4:
            for index in range(row_bytes):
                left = int(row[index - bytes_per_pixel]) if index >= bytes_per_pixel else 0
                above = int(previous[index])
                upper_left = int(previous[index - bytes_per_pixel]) if index >= bytes_per_pixel else 0
                estimate = left + above - upper_left
                distances = (abs(estimate - left), abs(estimate - above), abs(estimate - upper_left))
                predictor = left if distances[0] <= distances[1] and distances[0] <= distances[2] else (
                    above if distances[1] <= distances[2] else upper_left
                )
                row[index] = (int(row[index]) + predictor) & 255
        elif filter_type != 0:
            raise ValueError(f"Unsupported PNG filter type: {filter_type}")
        rows[row_index] = row
        previous = row

    if bit_depth == 8:
        return rows.reshape(height, width).astype(np.float32)
    return rows.reshape(height, width, 2).view(">u2").reshape(height, width).astype(np.float32)


def paired_frames(archive: ZipFile) -> list[tuple[str, str]]:
    depth_paths = {}
    confidence_paths = {}
    for name in archive.namelist():
        normalized = name.lower()
        if not normalized.endswith(".png"):
            continue
        parts = PurePosixPath(normalized).parts
        if "depth" in parts:
            depth_paths[PurePosixPath(name).stem] = name
        elif "confidence" in parts:
            confidence_paths[PurePosixPath(name).stem] = name
    pairs = [(depth_paths[key], confidence_paths[key]) for key in sorted(depth_paths.keys() & confidence_paths.keys())]
    if not pairs:
        raise ValueError("Archive has no matching depth/confidence PNG pairs")
    return pairs


def sample_frame(depth: np.ndarray, confidence: np.ndarray, rng: np.random.Generator, sample_limit: int):
    if depth.shape != confidence.shape:
        raise ValueError("Depth and confidence frames must have matching dimensions")
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return None
    confidence_floor = np.percentile(confidence[valid], 50)
    valid &= confidence >= confidence_floor
    coordinates = np.flatnonzero(valid)
    if not len(coordinates):
        return None
    if len(coordinates) > sample_limit:
        coordinates = rng.choice(coordinates, size=sample_limit, replace=False)

    height, width = depth.shape
    y, x = np.divmod(coordinates, width)
    padded_depth = np.pad(depth, 1, mode="edge")
    padded_confidence = np.pad(confidence, 1, mode="edge")
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    depth_neighbors = np.column_stack([padded_depth[y + 1 + dy, x + 1 + dx] for dy, dx in offsets])
    confidence_neighbors = np.column_stack([
        padded_confidence[y + 1 + dy, x + 1 + dx]
        for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    ])
    has_neighbor = np.any(depth_neighbors > 0, axis=1)
    if not np.any(has_neighbor):
        return None

    features = np.column_stack((
        depth_neighbors / 65535.0,
        confidence_neighbors / 255.0,
        x / max(width - 1, 1),
        y / max(height - 1, 1),
    )).astype(np.float64)
    target = depth[y, x].astype(np.float64) / 65535.0
    median_prediction = np.array([
        np.median(neighbors[neighbors > 0]) if np.any(neighbors > 0) else np.nan
        for neighbors in depth_neighbors
    ], dtype=np.float64)
    return features[has_neighbor], target[has_neighbor], median_prediction[has_neighbor]


def collect_samples(archive_path: Path, frame_step: int, samples_per_frame: int, seed: int):
    rng = np.random.default_rng(seed)
    feature_batches, target_batches, baseline_batches = [], [], []
    with ZipFile(archive_path) as archive:
        pairs = paired_frames(archive)
        for index, (depth_path, confidence_path) in enumerate(pairs):
            if index % frame_step:
                continue
            depth = decode_grayscale_png(archive.read(depth_path))
            confidence = decode_grayscale_png(archive.read(confidence_path))
            sample = sample_frame(depth, confidence, rng, samples_per_frame)
            if sample is None:
                continue
            features, targets, baseline = sample
            feature_batches.append(features)
            target_batches.append(targets)
            baseline_batches.append(baseline)
    if not feature_batches:
        raise ValueError(f"No usable depth samples found in {archive_path}")
    return np.concatenate(feature_batches), np.concatenate(target_batches), np.concatenate(baseline_batches)


def error_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    difference = (prediction - target) * 65535.0
    return {
        "rmse_native_depth_units": round(float(np.sqrt(np.mean(difference ** 2))), 3),
        "mae_native_depth_units": round(float(np.mean(np.abs(difference))), 3),
        "sample_count": int(len(target)),
    }


def choose_recommended_method(model_rmse: float, baseline_rmse: float) -> str:
    return "linear_local_model" if model_rmse < baseline_rmse else "neighbor_median_baseline"


def train(args) -> dict:
    train_paths = [Path(path).resolve() for path in args.train]
    holdout_path = Path(args.holdout).resolve()
    if holdout_path in train_paths:
        raise ValueError("The holdout archive must not also appear in the training archives")
    for path in [*train_paths, holdout_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    train_batches = [
        collect_samples(path, args.frame_step, args.samples_per_frame, args.seed + index)
        for index, path in enumerate(train_paths)
    ]
    training_features = np.concatenate([batch[0] for batch in train_batches])
    training_targets = np.concatenate([batch[1] for batch in train_batches])
    holdout_features, holdout_targets, holdout_baseline = collect_samples(
        holdout_path, args.frame_step, args.samples_per_frame, args.seed + len(train_paths)
    )

    feature_sets = {
        "all": list(range(19)),
        "no_position": list(range(17)),
        "depth_only": list(range(8)),
        "depth_plus_center_confidence": list(range(8)) + [12],
    }
    ablations = {}
    for name, columns in feature_sets.items():
        design = np.column_stack((np.ones(len(training_features)), training_features[:, columns]))
        regularizer = np.eye(design.shape[1]) * args.ridge
        regularizer[0, 0] = 0
        coefficients = np.linalg.solve(design.T @ design + regularizer, design.T @ training_targets)
        prediction = np.clip(
            np.column_stack((np.ones(len(holdout_features)), holdout_features[:, columns])) @ coefficients,
            0,
            1,
        )
        ablations[name] = error_metrics(prediction, holdout_targets)
        if name == "all":
            full_coefficients = coefficients
            full_prediction = prediction

    model_metrics = error_metrics(full_prediction, holdout_targets)
    baseline_metrics = error_metrics(holdout_baseline / 65535.0, holdout_targets)
    selected_method = choose_recommended_method(
        model_metrics["rmse_native_depth_units"], baseline_metrics["rmse_native_depth_units"]
    )

    report = {
        "model": "linear-local-depth-completion-v1",
        "evaluation_kind": "capture-held-out prediction of the scanner's own high-confidence depth samples",
        "not_physical_accuracy": True,
        "label_source": "high-confidence depth PNG values used as pseudo-labels; no laser/tape truth",
        "training_archives": [path.name for path in train_paths],
        "holdout_archive": holdout_path.name,
        "training_sample_count": int(len(training_targets)),
        "feature_names": FEATURE_NAMES,
        "model_metrics": model_metrics,
        "neighbor_median_baseline_metrics": baseline_metrics,
        "feature_ablations": ablations,
        "recommended_method": selected_method,
        "selection_reason": (
            "The trained linear candidate did not beat the neighbor-median baseline on the held-out capture."
            if selected_method == "neighbor_median_baseline"
            else "The trained linear candidate beat the neighbor-median baseline on the held-out capture."
        ),
        "root_cause_hypothesis": (
            "A global linear fit is less robust than local median aggregation around depth discontinuities; "
            "the tested feature ablations did not improve held-out sensor-depth prediction."
            if selected_method == "neighbor_median_baseline"
            else None
        ),
        "limitations": [
            "Evaluates sensor-depth interpolation, not room dimensions or real-world distance error.",
            "The Drive archives contain no laser/tape measurements or damage annotations.",
            "Do not use this baseline to claim any case-study accuracy gate passes.",
        ],
        "trained_candidate_coefficients": [round(float(value), 12) for value in full_coefficients],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a sensor-depth completion baseline from RGB-D scan archives.")
    parser.add_argument("--train", nargs="+", required=True, help="Training RGB-D ZIP archives")
    parser.add_argument("--holdout", required=True, help="Capture archive held out from training")
    parser.add_argument("--output", default="models/depth_completion_baseline.json")
    parser.add_argument("--frame-step", type=int, default=20)
    parser.add_argument("--samples-per-frame", type=int, default=384)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--ridge", type=float, default=1e-4)
    args = parser.parse_args()
    if args.frame_step < 1 or args.samples_per_frame < 1 or args.ridge < 0:
        parser.error("frame step and sample count must be positive; ridge must be non-negative")
    try:
        report = train(args)
    except (BadZipFile, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "model": report["model"],
        "training_sample_count": report["training_sample_count"],
        "model_metrics": report["model_metrics"],
        "neighbor_median_baseline_metrics": report["neighbor_median_baseline_metrics"],
        "recommended_method": report["recommended_method"],
        "output": args.output,
        "not_physical_accuracy": report["not_physical_accuracy"],
    }, indent=2))


if __name__ == "__main__":
    main()