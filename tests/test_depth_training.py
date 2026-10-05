import struct
import unittest
import zlib

import numpy as np

from scripts.train_depth_completion import choose_recommended_method, decode_grayscale_png, sample_frame
from scripts.lidar_pipeline import analyze_floor_geometry, connected_floor_regions, convex_hull, depth_to_camera_points, polygon_area, quaternion_rotate


def make_grayscale_png(rows, bit_depth):
    bytes_per_sample = bit_depth // 8
    height = len(rows)
    width = len(rows[0])
    raw = bytearray()
    for row in rows:
        raw.append(0)
        for value in row:
            raw.extend(value.to_bytes(bytes_per_sample, "big"))

    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, bit_depth, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


class DepthTrainingTests(unittest.TestCase):
    def test_decodes_16_bit_depth_png(self):
        decoded = decode_grayscale_png(make_grayscale_png([[1, 512], [1024, 65535]], 16))
        np.testing.assert_array_equal(decoded, [[1, 512], [1024, 65535]])

    def test_decodes_8_bit_confidence_png(self):
        decoded = decode_grayscale_png(make_grayscale_png([[0, 127, 255]], 8))
        np.testing.assert_array_equal(decoded, [[0, 127, 255]])

    def test_training_sample_uses_neighbor_features_not_center_depth(self):
        depth = np.arange(25, dtype=np.float32).reshape(5, 5) + 1
        confidence = np.full((5, 5), 255, dtype=np.float32)
        features, targets, baseline = sample_frame(depth, confidence, np.random.default_rng(3), 25)
        self.assertEqual(features.shape[1], 19)
        self.assertEqual(len(features), len(targets))
        self.assertTrue(np.all(np.isfinite(baseline)))

    def test_lower_error_baseline_is_selected(self):
        self.assertEqual(choose_recommended_method(2.953, 2.139), "neighbor_median_baseline")
        self.assertEqual(choose_recommended_method(1.2, 2.1), "linear_local_model")

    def test_backprojection_uses_intrinsics_and_metric_scale(self):
        depth = np.full((4, 4), 1000, dtype=np.float32)
        confidence = np.full((4, 4), 2, dtype=np.float32)
        points = depth_to_camera_points(depth, confidence, np.array([4.0, 4.0, 2.0, 2.0]), pixel_stride=1)
        self.assertEqual(points.shape, (16, 3))
        np.testing.assert_allclose(points[10], [0, 0, 1], atol=1e-6)

    def test_identity_pose_preserves_camera_points(self):
        points = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 2.0]])
        identity = np.broadcast_to(np.array([0.0, 0.0, 0.0, 1.0]), (len(points), 4))
        np.testing.assert_allclose(quaternion_rotate(points, identity), points)

    def test_convex_hull_and_area_for_rectangular_floor_points(self):
        points = np.array([[0, 0], [2, 0], [2, 3], [0, 3], [1, 1], [1, 2]])
        hull = convex_hull(points)
        self.assertEqual(len(hull), 4)
        self.assertAlmostEqual(polygon_area(hull), 6.0)

    def test_floor_candidate_is_explicitly_unvalidated(self):
        points = np.array([[x, y, z] for x in np.linspace(0, 3, 12) for z in np.linspace(0, 2, 12) for y in (-1.0, 0.0, 1.0)])
        result = analyze_floor_geometry(points)
        self.assertEqual(result["surface_status"], "candidate_unvalidated")
        self.assertIsNone(result["confidence_interval_m2"])
        self.assertGreaterEqual(len(result["floor_region_candidates"]), 1)

    def test_floor_region_component_detector_keeps_separate_patches(self):
        points = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.1], [2.0, 2.0], [2.21, 2.0], [2.21, 2.21]])
        regions = connected_floor_regions(points, cell_size_m=0.2, minimum_cells=2)
        self.assertEqual([region["cell_count"] for region in regions], [3, 2])


if __name__ == "__main__":
    unittest.main()