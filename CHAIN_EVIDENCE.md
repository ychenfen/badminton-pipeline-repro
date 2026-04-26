# Most Credible Generation Chain (Original -> Target)

Target file:

- `D:\yumaoqiu\end1_fix_swap2_precision_full_fx.mp4`

## Stage A: Original video to TrackNet outputs

Input:

- `D:\yumaoqiu\866ba79f9b46ce0d9b8b1d55eb82832c.mp4`

Script family:

- `tracknet/predict.py`

Observed output naming logic in script:

- `<video_stem>.mp4`
- `<video_stem>_ball.csv`

## Stage B: TrackNet video to precision/full overlay

Script:

- `overlay_player_analytics.py`

Observed defaults in script:

- default input: `D:\yumaoqiu\tracknet_v3_result\866ba79f9b46ce0d9b8b1d55eb82832c_tracknetv3.mp4`
- default output: `D:\yumaoqiu\end1.mp4`

Most credible derived intermediate:

- `D:\yumaoqiu\end1_fix_swap2_precision_full.mp4`

## Stage C: Precision/full overlay to final FX

Script:

- `video_fx_bullet_time.py`

Hard-coded defaults in script:

- input: `D:\yumaoqiu\end1_fix_swap2_precision_full.mp4`
- output: `D:\yumaoqiu\end1_fix_swap2_precision_full_fx.mp4`

This directly explains the target file naming and final render stage.

## Why this chain is trusted

- Script default paths point to the same naming lineage.
- File timeline order matches the stage progression.
- Frame-count behavior is consistent with FX expansion in `video_fx_bullet_time.py`.

