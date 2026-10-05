# Shared Drive Scan Data

Source: [Assignment - YC Startup](https://drive.google.com/drive/folders/1QIzPgDXL27EzL3RB_H-Cf75NABtD68SE), downloaded on 2026-10-04.

| Archive | Drive size | Contents observed |
| --- | ---: | --- |
| `single_room.zip` | 84.4 MB | 1,715 paired depth/confidence frames, RGB video, odometry, IMU, camera matrix |
| `single_scan_floor_only.zip` | 264 MB | 5,251 paired depth/confidence frames, RGB video, odometry, IMU, camera matrix |
| `single_scan_with_ceiling.zip` | 484.9 MB | 9,745 paired depth/confidence frames, RGB video, odometry, IMU, camera matrix |

All three archives passed ZIP CRC checks. They contain 16,711 paired raw sensor frames in total, but no filenames indicating dimension ground truth, damage labels, or annotations. The depth-completion experiment trained on `single_room.zip` and `single_scan_floor_only.zip`, then held out `single_scan_with_ceiling.zip`: linear candidate RMSE 2.953 versus neighbor-median RMSE 2.139 in raw depth units. The trained linear model did not beat the baseline, so it is not selected for deployment. Its metrics are not real-world accuracy measurements; see `../models/depth_completion_baseline.json` for the full report and ablations.

Keep the original archives in `raw/`; do not modify them. They are large binary artifacts and should be handled locally.