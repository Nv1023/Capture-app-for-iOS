import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

from app import app


class CaptureWorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.original_capture_root = app.config["CAPTURE_ROOT"]
        app.config.update(TESTING=True, CAPTURE_ROOT=Path(self.tempdir.name))
        self.client = app.test_client()

    def tearDown(self):
        app.config.update(TESTING=False, CAPTURE_ROOT=self.original_capture_root)
        self.tempdir.cleanup()

    def create_capture(self, tier, files):
        return self.client.post(
            "/api/captures",
            data={"tier": tier, "files": files},
            content_type="multipart/form-data",
        )

    def test_photo_tier_requires_two_to_eight_images_per_room(self):
        too_few = self.create_capture("photos", [(BytesIO(b"a"), "Kitchen/one.jpg")])
        self.assertEqual(too_few.status_code, 400)

        valid = self.create_capture(
            "photos",
            [(BytesIO(b"a"), "Kitchen/one.jpg"), (BytesIO(b"b"), "Kitchen/two.jpg")],
        )
        self.assertEqual(valid.status_code, 201)
        self.assertEqual(valid.json["room_count"], 1)

    def test_video_tier_accepts_one_walkthrough(self):
        response = self.create_capture("video", [(BytesIO(b"video"), "walkthrough.mov")])
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json["tier"], "video")

    def test_lidar_bundle_requires_depth_pose_and_intrinsics(self):
        archive = BytesIO()
        with ZipFile(archive, "w") as bundle:
            bundle.writestr("capture/depth/000001.png", "depth")
            bundle.writestr("capture/odometry.csv", "pose")
            bundle.writestr("capture/camera_matrix.csv", "intrinsics")
        accepted = self.create_capture("lidar", [(BytesIO(archive.getvalue()), "capture.zip")])
        self.assertEqual(accepted.status_code, 201)

        rejected = self.create_capture("lidar", [(BytesIO(b"not a zip"), "capture.zip")])
        self.assertEqual(rejected.status_code, 400)

    def test_run_exports_unmeasured_contract_and_schema(self):
        capture = self.create_capture("video", [(BytesIO(b"video"), "walkthrough.mp4")])
        capture_id = capture.json["capture_id"]
        output = self.client.post(f"/api/captures/{capture_id}/run")
        self.assertEqual(output.status_code, 200)
        self.assertEqual(output.json["processing_status"], "not_measured")
        self.assertEqual(output.json["pipeline_stages"]["tier_reconstruction"]["status"], "not_implemented")
        self.assertEqual(output.json["pipeline_stages"]["common_geometry"]["status"], "blocked_by_tier_reconstruction")
        self.assertIsNone(output.json["property"]["stitched_plan"]["footprint_m2"])
        download = self.client.get(f"/api/captures/{capture_id}/output")
        self.assertEqual(download.status_code, 200)
        download.close()

        schema = self.client.get("/schema/capture-output.schema.json")
        self.assertEqual(schema.status_code, 200)
        self.assertEqual(schema.mimetype, "application/schema+json")
        self.assertEqual(json.loads(schema.data)["properties"]["schema_version"]["const"], "0.2.0-experimental")
        schema.close()

    def test_training_report_route_serves_generated_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report_path = root / "models" / "depth_completion_baseline.json"
            report_path.parent.mkdir(parents=True)
            report_path.write_text(json.dumps({"not_physical_accuracy": True}), encoding="utf-8")
            with patch("app.ROOT", root):
                response = self.client.get("/training-report")
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json["not_physical_accuracy"])
            response.close()

    def test_lidar_run_writes_partial_geometry_artifacts(self):
        archive = BytesIO()
        with ZipFile(archive, "w") as bundle:
            bundle.writestr("capture/depth/000001.png", "depth")
            bundle.writestr("capture/odometry.csv", "pose")
            bundle.writestr("capture/camera_matrix.csv", "intrinsics")
        capture = self.create_capture("lidar", [(BytesIO(archive.getvalue()), "scan.zip")])
        capture_id = capture.json["capture_id"]
        points = np.array([[x, y, z] for x in np.linspace(0, 3, 12) for z in np.linspace(0, 2, 12) for y in (-1.0, 0.0, 1.0)])
        metadata = {
            "source_archive": "scan.zip",
            "point_count": len(points),
            "coordinate_assumptions": {"depth_unit": "test", "camera_axes": "test", "pose_axes": "test", "confidence_filter": "test", "intrinsics": "test"},
        }
        with patch("app.reconstruct_point_cloud", return_value=(points, metadata)):
            result = self.client.post(f"/api/captures/{capture_id}/run")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json["processing_status"], "partial_geometry_unvalidated")
        self.assertEqual(result.json["pipeline_stages"]["room_detection"]["status"], "floor_region_candidates_not_room_labels")
        self.assertIsNone(result.json["property"]["stitched_plan"]["footprint_m2"])
        artifact_dir = Path(self.tempdir.name) / capture_id / "artifacts"
        self.assertTrue((artifact_dir / "world_points.npz").is_file())
        self.assertTrue((artifact_dir / "geometry-preview.svg").is_file())
        self.assertTrue((artifact_dir / "pipeline-output.json").is_file())


if __name__ == "__main__":
    unittest.main()