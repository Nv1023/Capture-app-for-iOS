from __future__ import annotations

import json
import re
import sys
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from flask import Flask, jsonify, render_template, request, send_file, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from scripts.lidar_pipeline import build_partial_output, reconstruct_point_cloud, write_point_cloud


ROOT = Path(__file__).resolve().parent
CAPTURE_ROOT = ROOT / "captures"
ALLOWED_IMAGE = {".jpg", ".jpeg", ".png", ".heic", ".webp"}
ALLOWED_VIDEO = {".mov", ".mp4", ".m4v"}
ALLOWED_SENSOR = {".json", ".csv", ".npy", ".bin", ".ply", ".png", ".tiff", ".tif"}

app = Flask(__name__)
app.config.update(MAX_CONTENT_LENGTH=512 * 1024 * 1024, CAPTURE_ROOT=CAPTURE_ROOT)


def utc_now() -> str:
	return datetime.now(timezone.utc).isoformat(timespec="seconds")


def capture_dir(capture_id: str) -> Path:
	if not re.fullmatch(r"[a-f0-9]{32}", capture_id):
		raise ValueError("Invalid capture ID")
	return Path(app.config["CAPTURE_ROOT"]) / capture_id


def read_manifest(capture_id: str) -> dict:
	path = capture_dir(capture_id) / "manifest.json"
	if not path.is_file():
		raise FileNotFoundError("Capture not found")
	return json.loads(path.read_text(encoding="utf-8"))


def safe_relative_path(value: str) -> PurePosixPath:
	normalized = value.replace("\\", "/")
	path = PurePosixPath(normalized)
	if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
		raise ValueError("File paths must stay inside their room folder")
	return path


def validate_photo_files(files) -> tuple[int, list[str]]:
	rooms: dict[str, int] = {}
	for uploaded in files:
		relative = safe_relative_path(uploaded.filename or "")
		if len(relative.parts) < 2:
			raise ValueError("Photo files must be inside one folder per room")
		if Path(relative.name).suffix.lower() not in ALLOWED_IMAGE:
			raise ValueError(f"Unsupported photo format: {relative.name}")
		room = relative.parts[0]
		rooms[room] = rooms.get(room, 0) + 1
	if not rooms:
		raise ValueError("Add at least one room folder")
	invalid = [name for name, count in rooms.items() if not 2 <= count <= 8]
	if invalid:
		raise ValueError("Each room folder must contain 2 to 8 still images")
	return len(rooms), [f"{name}: {count} stills" for name, count in sorted(rooms.items())]


def validate_lidar_file(uploaded) -> None:
	if Path(uploaded.filename or "").suffix.lower() != ".zip":
		raise ValueError("LiDAR tier expects a ZIP bundle containing depth, poses, and intrinsics")
	uploaded.stream.seek(0)
	try:
		with zipfile.ZipFile(uploaded.stream) as archive:
			names = [item.filename.lower() for item in archive.infolist() if not item.is_dir()]
	except zipfile.BadZipFile as error:
		raise ValueError("The LiDAR bundle is not a valid ZIP archive") from error
	uploaded.stream.seek(0)
	required = {
		"depth": any("depth" in name and Path(name).suffix in ALLOWED_SENSOR for name in names),
		"pose": any(("pose" in name or "trajectory" in name or "odometry" in name) and Path(name).suffix in ALLOWED_SENSOR for name in names),
		"intrinsics": any(("intrinsic" in name or "camera_matrix" in name) and Path(name).suffix in ALLOWED_SENSOR for name in names),
	}
	missing = [label for label, present in required.items() if not present]
	if missing:
		raise ValueError("LiDAR bundle is missing: " + ", ".join(missing))


def make_output(manifest: dict, lidar_result: dict | None = None) -> dict:
	limitations = [
		"Photo and video visual reconstruction is not implemented.",
		"Room detection, connectivity, and floorplan graph are not implemented.",
		"Damage detection and concealed-damage rules are not implemented.",
		"No laser/tape ground truth is available to calibrate measurement intervals.",
	]
	if lidar_result is None:
		limitations.insert(0, "This input tier has no reconstruction adapter in this build.")
	else:
		limitations.extend(lidar_result["limitations"])
	floor_geometry = lidar_result.get("floor_geometry_candidate", {}) if lidar_result else {}
	planes = floor_geometry.get("planes", {})
	floor_area = floor_geometry.get("observed_floor_hull_area_m2")
	ceiling_height = planes.get("ceiling_height_candidate_m")
	if lidar_result:
		pipeline_stages = lidar_result["stages"]
	else:
		tier_stage = {
			"photos": {"name": "visual_reconstruction", "status": "not_implemented"},
			"video": {"name": "visual_tracking_and_reconstruction", "status": "not_implemented"},
		}.get(manifest["tier"], {"name": "lidar_reconstruction", "status": "not_run"})
		pipeline_stages = {
			"tier_reconstruction": tier_stage,
			"common_geometry": {"status": "blocked_by_tier_reconstruction"},
			"room_detection": {"status": "blocked"},
			"room_connectivity": {"status": "blocked"},
			"global_alignment": {"status": "blocked"},
			"floorplan_graph": {"status": "blocked"},
			"damage_detection": {"status": "not_implemented"},
			"concealed_damage": {"status": "not_implemented"},
			"confidence_calibration": {"status": "not_calibrated"},
			"render": {"status": "not_available"},
		}
	return {
		"schema_version": "0.2.0-experimental",
		"capture_id": manifest["capture_id"],
		"created_at": utc_now(),
		"input_tier": manifest["tier"],
		"processing_status": "partial_geometry_unvalidated" if lidar_result else "not_measured",
		"limitations": limitations,
		"pipeline_stages": pipeline_stages,
		"property": {
			"rooms": [],
			"stitched_plan": {
				"status": "geometry_preview_only" if lidar_result else "not_measured",
				"render_path": "artifacts/geometry-preview.svg" if lidar_result else None,
				"footprint_m2": None,
				"footprint_interval_m2": None,
				"observed_floor_hull_m2": floor_area,
			},
			"drift_handling": {
				"method": "recorded odometry used without correction" if lidar_result else None,
				"ablation": "not_run",
			},
		},
		"measurements": {
			"wall_lengths": [],
			"ceiling_heights": ([{"room_id": None, "estimate_m": ceiling_height, "confidence_interval_m": None, "status": "candidate_unvalidated"}] if ceiling_height is not None else []),
			"floor_areas": ([{"room_id": None, "observed_hull_estimate_m2": floor_area, "confidence_interval_m2": None, "status": "observed_floor_hull_not_room_area"}] if floor_area is not None else []),
			"openings": [],
			"damage_regions": [],
			"concealed_damage_flags": [],
			"scope_line_items": [],
		},
		"benchmark": {"gate_status": "not_run", "ground_truth": None, "repeatability": None, "incumbent_comparison": None},
		"raw_inputs": manifest["files"],
	}


def persist_json(path: Path, payload: dict) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_suffix(path.suffix + ".tmp")
	temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
	temporary.replace(path)


def process_capture(capture_id: str) -> dict:
	manifest = read_manifest(capture_id)
	destination = capture_dir(capture_id)
	lidar_result = None
	if manifest["tier"] == "lidar":
		input_path = (destination / manifest["files"][0]["path"]).resolve()
		input_path.relative_to(destination.resolve())
		points, metadata = reconstruct_point_cloud(input_path)
		artifact_dir = destination / "artifacts"
		write_point_cloud(points, metadata, artifact_dir)
		lidar_result = build_partial_output(points, metadata, artifact_dir)
		persist_json(artifact_dir / "pipeline-output.json", lidar_result)
	output = make_output(manifest, lidar_result)
	persist_json(destination / "output.json", output)
	manifest["processing_status"] = output["processing_status"]
	manifest["output_path"] = "output.json"
	persist_json(destination / "manifest.json", manifest)
	return output


@app.get("/")
def index():
	return render_template("index.html")


@app.get("/api/health")
def health():
	return jsonify({"status": "ok", "inference_available": False, "lidar_geometry_available": True, "schema_version": "0.2.0-experimental"})


@app.get("/schema/capture-output.schema.json")
def output_schema():
	return send_file(ROOT / "schemas" / "capture-output.schema.json", mimetype="application/schema+json")


@app.get("/training-report")
def training_report():
	path = ROOT / "models" / "depth_completion_baseline.json"
	if not path.is_file():
		return jsonify({"error": "No training report has been generated"}), 404
	return send_file(path, mimetype="application/json")


@app.get("/reference-data/<path:filename>")
def reference_data(filename: str):
	return send_from_directory(ROOT / "artifacts" / "reference", filename)


@app.get("/api/captures")
def list_captures():
	root = Path(app.config["CAPTURE_ROOT"])
	captures = []
	if root.exists():
		for path in root.glob("*/manifest.json"):
			try:
				captures.append(json.loads(path.read_text(encoding="utf-8")))
			except (OSError, json.JSONDecodeError):
				continue
	return jsonify(sorted(captures, key=lambda item: item["created_at"], reverse=True))


@app.post("/api/captures")
def create_capture():
	tier = request.form.get("tier", "")
	title = request.form.get("title", "").strip()[:100] or "Untitled capture"
	files = [item for item in request.files.getlist("files") if item.filename]
	try:
		if tier == "photos":
			room_count, room_summary = validate_photo_files(files)
		elif tier == "video":
			if len(files) != 1 or Path(files[0].filename).suffix.lower() not in ALLOWED_VIDEO:
				raise ValueError("Video tier expects one .mov, .mp4, or .m4v walkthrough")
			room_count, room_summary = None, []
		elif tier == "lidar":
			if len(files) != 1:
				raise ValueError("LiDAR tier expects one ZIP bundle")
			validate_lidar_file(files[0])
			room_count, room_summary = None, []
		else:
			raise ValueError("Choose photos, video, or LiDAR")
	except ValueError as error:
		return jsonify({"error": str(error)}), 400

	capture_id = uuid.uuid4().hex
	destination = capture_dir(capture_id)
	saved_files = []
	for index, uploaded in enumerate(files, start=1):
		original = uploaded.filename.replace("\\", "/")
		suffix = Path(original).suffix.lower()
		if tier == "photos":
			relative = safe_relative_path(original)
			room_folder = secure_filename(relative.parts[0]) or "room"
			file_name = secure_filename(relative.name) or f"photo{suffix}"
			output = destination / "raw" / room_folder / f"{index:03d}_{file_name}"
		else:
			file_name = secure_filename(Path(original).name) or f"input{suffix}"
			output = destination / "raw" / f"{index:03d}_{file_name}"
		output.parent.mkdir(parents=True, exist_ok=True)
		uploaded.save(output)
		saved_files.append({"path": output.relative_to(destination).as_posix(), "original_name": Path(original).name, "bytes": output.stat().st_size})

	manifest = {
		"capture_id": capture_id,
		"title": title,
		"tier": tier,
		"created_at": utc_now(),
		"intake_status": "validated",
		"processing_status": "not_run",
		"room_count": room_count,
		"room_folders": room_summary,
		"files": saved_files,
		"pipeline_available": False,
	}
	persist_json(destination / "manifest.json", manifest)
	return jsonify(manifest), 201


@app.post("/api/captures/<capture_id>/run")
def run_capture(capture_id: str):
	try:
		output = process_capture(capture_id)
	except (ValueError, FileNotFoundError):
		return jsonify({"error": "Capture not found"}), 404
	except (OSError, zipfile.BadZipFile) as error:
		return jsonify({"error": f"Capture processing failed: {error}"}), 422
	return jsonify(output), 200


@app.get("/api/captures/<capture_id>/output")
def get_output(capture_id: str):
	try:
		path = capture_dir(capture_id) / "output.json"
	except ValueError:
		return jsonify({"error": "Capture not found"}), 404
	if not path.is_file():
		return jsonify({"error": "Run this capture to generate its output contract"}), 404
	return send_file(path, mimetype="application/json", as_attachment=True, download_name=f"{capture_id}-output.json")


@app.errorhandler(RequestEntityTooLarge)
def upload_too_large(_error):
	return jsonify({"error": "Upload exceeds the 512 MB per-request limit"}), 413


if __name__ == "__main__":
	if len(sys.argv) == 3 and sys.argv[1] == "process":
		output = process_capture(sys.argv[2])
		print(json.dumps({"capture_id": sys.argv[2], "processing_status": output["processing_status"], "output": "output.json"}, indent=2))
	else:
		app.run(host="127.0.0.1", port=5000, debug=False)
