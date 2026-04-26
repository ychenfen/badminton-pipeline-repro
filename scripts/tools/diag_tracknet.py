"""Probe TrackNet output: load model, run on a few real frames, print heatmap stats."""
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tracknet_runtime"))

from utils.general import HEIGHT, WIDTH, get_model  # type: ignore


def load_frames(video, count, start=200):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(count):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def main():
    video = "/Users/yuchenxu/Desktop/pipeline_repro_bundle/short.mp4"
    weights = "/Users/yuchenxu/Desktop/pipeline_repro_bundle/weights/TrackNet_best.pt"

    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    seq_len = ckpt["param_dict"]["seq_len"]
    bg_mode = ckpt["param_dict"]["bg_mode"]
    print(f"ckpt seq_len={seq_len}  bg_mode={bg_mode}")

    model = get_model("TrackNet", seq_len, bg_mode)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Build median over a wider sample of the video, like the dataset does
    cap_full = cv2.VideoCapture(video)
    n_total = int(cap_full.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_idx = list(range(0, n_total, max(1, n_total // 200)))
    median_pool = []
    for i in sample_idx:
        cap_full.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap_full.read()
        if ok:
            median_pool.append(fr)
    cap_full.release()
    median_bgr = np.median(np.stack(median_pool, axis=0), axis=0).astype(np.uint8)
    median_rgb = median_bgr[..., ::-1]
    median_small = cv2.resize(median_rgb, (WIDTH, HEIGHT))
    median_chw = np.moveaxis(median_small, -1, 0).astype(np.float32)  # (3, H, W)

    print(f"median pool size: {len(median_pool)}  shape: {median_bgr.shape}")

    for start in [50, 200, 350, 500]:
        raw = load_frames(video, seq_len, start=start)
        if len(raw) < seq_len:
            print(f"start={start}: not enough frames")
            continue
        seq = []
        for f in raw:
            small = cv2.resize(f[..., ::-1], (WIDTH, HEIGHT))  # BGR->RGB then resize
            seq.append(np.moveaxis(small, -1, 0).astype(np.float32))  # (3, H, W)
        seq_chw = np.concatenate(seq, axis=0)  # (3*L, H, W)

        if bg_mode == "concat":
            x_arr = np.concatenate([median_chw, seq_chw], axis=0)  # (3*(L+1), H, W)
        elif bg_mode == "subtract":
            sub = []
            for f in raw:
                diff = np.sum(np.abs(f[..., ::-1].astype(np.float32) - median_rgb.astype(np.float32)), axis=-1).astype(np.uint8)
                diff_small = cv2.resize(diff, (WIDTH, HEIGHT))
                sub.append(diff_small[None])
            x_arr = np.concatenate(sub, axis=0).astype(np.float32)  # (L, H, W)
        else:
            x_arr = seq_chw

        x = (x_arr / 255.0).reshape(1, *x_arr.shape)
        with torch.no_grad():
            y = model(torch.from_numpy(x).float()).numpy()
        per_max = y[0].reshape(seq_len, -1).max(axis=1)
        n_pos = int((per_max > 0.5).sum())
        print(f"start={start}: y max={y.max():.4f} mean={y.mean():.5f}  per-frame max={per_max.round(3).tolist()}  >0.5: {n_pos}/{seq_len}")


if __name__ == "__main__":
    main()
