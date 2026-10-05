# Roomfield Capture Workbench

A local-first intake and evidence-tracking prototype based on the Applied AI Engineer case study. It validates photo-folder, video, and LiDAR-bundle inputs, stores originals locally, and produces a versioned JSON output contract.

## Important scope

The LiDAR route now produces a sparse pose-projected point cloud, floor/ceiling candidates, an observed-floor hull, and an SVG geometry preview. Its depth unit and pose conventions are inferred, not documented. It does **not** yet detect rooms, correct drift, build room connectivity/floorplan graphs, reconstruct from photos/video, detect damage, calibrate intervals, or score physical-accuracy gates. The emitted geometry is partial and unvalidated; do not treat it as a survey, estimate, or construction document.

## Run locally

Python 3.10 or newer is recommended. From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>. Inputs are stored under `captures/` and are not uploaded to a service. The upload limit is 512 MB per request. Do not expose the development server to a public network.

## Intake requirements

- **Photos:** choose a directory containing one subfolder per room. Each room folder must contain 2–8 JPEG, PNG, HEIC, or WebP stills. This validates folder structure and extensions, not image quality or camera metadata.
- **Video:** one original MOV, MP4, or M4V walkthrough.
- **LiDAR:** one ZIP with files whose names identify depth, pose (or trajectory), and intrinsics (or camera matrix). This checks names only, not synchronization, calibration, or units. A validated raw-data capture app and export protocol remain open work.

The web interface creates a capture ID and stores originals. Use its **RUN** action, or run the generated output contract from a terminal:

```powershell
python app.py process <capture_id>
```

The output appears at `captures/<capture_id>/output.json`; schema: [`schemas/capture-output.schema.json`](schemas/capture-output.schema.json). LiDAR produces a `partial_geometry_unvalidated` result with raw geometry artifacts; Photos/Video remain `not_measured` rather than receiving fabricated results.

Run the intake and output-contract tests with `python -m unittest discover -s tests`.

Run the LiDAR geometry pipeline directly on a source archive:

```powershell
python scripts/lidar_pipeline.py datasets/raw/single_scan_with_ceiling.zip --output-dir artifacts/reference
```

This writes a compressed world-space point cloud, stage-by-stage JSON, and an explicitly unvalidated geometry preview. It does not apply loop closure or claim a final room plan.

## Drive scan data and training baseline

The shared Drive folder contains `single_room.zip`, `single_scan_floor_only.zip`, and `single_scan_with_ceiling.zip`. Keep the original archives under `datasets/raw/`. LiDAR intake recognizes their paired 16-bit depth PNGs, confidence PNGs, `odometry.csv`, `imu.csv`, RGB video, and `camera_matrix.csv`.

Train the experimental local depth-completion baseline after all three archives are present:

```powershell
python scripts/train_depth_completion.py `
	--train datasets/raw/single_room.zip datasets/raw/single_scan_floor_only.zip `
	--holdout datasets/raw/single_scan_with_ceiling.zip
```

The script hides sampled high-confidence sensor-depth pixels, predicts them from neighboring depth/confidence values, and compares a fitted linear model with a neighbor-median baseline. It splits by capture archive, not by frame. Its targets are the scanner's own readings, so its report measures sensor-depth consistency only. It is **not** laser/tape ground truth, a floor-plan accuracy score, or evidence that room dimensions meet any case-study gate. The Drive archives do not include explicit dimension measurements or damage annotations.

Current held-out result (`single_scan_with_ceiling.zip`, 187,392 samples): the trained linear candidate RMSE is 2.953 raw depth units (MAE 0.725), versus 2.139 (MAE 0.630) for the neighbor-median baseline. The candidate is therefore not selected. Feature ablations did not close the gap. The deterministic report and candidate coefficients are in `models/depth_completion_baseline.json`; `/training-report` serves that report while the app is running.

## Case-study coverage

The app includes the brief's gate thresholds, a draft iPhone Camera protocol, device capability matrix, fix-loop record, and compliance matrix. These are tracking surfaces, not proof of compliance. Accuracy values, room layouts, repeatability, drift ablation, staged-damage results, incumbent comparison, and raw benchmark bundles are all unrun. A production submission still needs a working reconstruction and damage pipeline, a verified LiDAR exporter, a measured multi-room benchmark across all three tiers, laser/tape truth, a consumer-app export, before/after fix artifacts, and a concise technical report.

## Local API

- `GET /api/health` reports app availability and that inference is not installed.
- `GET /api/captures` lists locally stored capture manifests.
- `POST /api/captures` validates tier-specific multipart input and stores originals.
- `POST /api/captures/<capture_id>/run` creates the honest, currently unmeasured output contract.
- `GET /api/captures/<capture_id>/output` downloads the generated JSON.