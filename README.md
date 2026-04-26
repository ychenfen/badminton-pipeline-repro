# Reproduce `end1_fix_swap2_precision_full_fx` On A New Computer

This folder is a portable bundle of the most credible generation chain:

1. TrackNet inference from original video.
2. Player analytics overlay rendering (precision path style).
3. Bullet-time FX rendering.

## 1) Folder Contents

- `run_all.ps1`: one-command pipeline runner.
- `requirements_repro.txt`: Python dependencies.
- `scripts/tracknet_runtime/*`: TrackNet runtime files used by `predict.py`.
- `scripts/overlay/overlay_player_analytics.py`: overlay renderer.
- `scripts/fx/video_fx_bullet_time.py`: final FX renderer.
- `weights/TrackNet_best.pt`: TrackNet checkpoint.
- `weights/yolov8s-pose.pt`: pose model for overlay stage.

## 2) Environment Setup (Windows)

1. Install Python 3.10+ (3.12 tested on this machine).
2. Open PowerShell in this folder.
3. Install dependencies:

```powershell
python -m pip install -r .\requirements_repro.txt
```

4. If you have NVIDIA GPU, install a CUDA-enabled `torch` build matching your CUDA driver (recommended for speed).

## 3) Run Full Pipeline

### Fast start (fixed court points)

```powershell
.\run_all.ps1 -InputVideo "D:\yumaoqiu\866ba79f9b46ce0d9b8b1d55eb82832c.mp4" -WorkRoot "D:\yumaoqiu_repro"
```

### Better alignment (manual court selection)

Use this when camera geometry differs and you want closer visual alignment:

```powershell
.\run_all.ps1 -InputVideo "D:\yumaoqiu\866ba79f9b46ce0d9b8b1d55eb82832c.mp4" -WorkRoot "D:\yumaoqiu_repro" -ManualCourtSelection
```

When manual mode is enabled, click court corners in this order:

- TL -> TR -> BR -> BL

## 4) Outputs

After a successful run, output files are created in `-WorkRoot`:

- `tracknet_v3_result_regen\<video_stem>_tracknetv3.mp4`
- `tracknet_v3_result_regen\<video_stem>_ball.csv`
- `end1_fix_swap2_precision_full_regen.mp4`
- `end1_fix_swap2_precision_full_fx_regen.mp4`

## 5) Notes For Cross-Computer Repro

- Exact pixel-perfect match may vary due to:
  - CUDA / torch versions
  - ultralytics / tracker implementation versions
  - court point selection differences
- The final stage (`video_fx_bullet_time.py`) is deterministic with the same input and parameters.
- This bundle intentionally avoids overwriting your original `D:\yumaoqiu\...` files.
