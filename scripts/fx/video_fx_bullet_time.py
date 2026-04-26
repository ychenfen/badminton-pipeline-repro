import argparse
import math
import os
from typing import List

import cv2
import numpy as np


DEFAULT_INPUT = r"D:\yumaoqiu\end1_fix_swap2_precision_full.mp4"
DEFAULT_OUTPUT = r"D:\yumaoqiu\end1_fix_swap2_precision_full_fx.mp4"


def parse_time_list(value: str, fps: float) -> List[int]:
    if not value:
        return []
    out = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        sec = float(part)
        out.append(max(0, int(round(sec * fps))))
    return sorted(set(out))


def unsharp(frame: np.ndarray, amount: float = 1.15, sigma: float = 1.0) -> np.ndarray:
    blur = cv2.GaussianBlur(frame, (0, 0), sigmaX=sigma, sigmaY=sigma)
    out = cv2.addWeighted(frame, amount, blur, 1.0 - amount, 0)
    return out


def zoom_around(frame: np.ndarray, scale: float, cx: int, cy: int) -> np.ndarray:
    if scale <= 1.0001:
        return frame
    h, w = frame.shape[:2]
    crop_w = max(2, int(w / scale))
    crop_h = max(2, int(h / scale))
    x1 = int(np.clip(cx - crop_w // 2, 0, w - crop_w))
    y1 = int(np.clip(cy - crop_h // 2, 0, h - crop_h))
    crop = frame[y1 : y1 + crop_h, x1 : x1 + crop_w]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def chromatic_shift(frame: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return frame
    h, w = frame.shape[:2]
    m_r = np.float32([[1, 0, px], [0, 1, 0]])
    m_b = np.float32([[1, 0, -px], [0, 1, 0]])
    b, g, r = cv2.split(frame)
    r2 = cv2.warpAffine(r, m_r, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    b2 = cv2.warpAffine(b, m_b, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return cv2.merge([b2, g, r2])


def build_vignette(h: int, w: int) -> np.ndarray:
    kx = cv2.getGaussianKernel(w, w * 0.55)
    ky = cv2.getGaussianKernel(h, h * 0.55)
    mask = (ky @ kx.T).astype(np.float32)
    mask /= max(1e-6, float(mask.max()))
    return mask


def smoothstep01(x: float) -> float:
    t = float(np.clip(x, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def generate_uniform_centers(total_frames: int, fps: float, count: int, margin_sec: float) -> List[int]:
    if count <= 0 or total_frames <= 1:
        return []
    margin = max(0, int(round(margin_sec * fps)))
    lo = margin
    hi = max(lo, total_frames - 1 - margin)
    if hi <= lo:
        lo = 0
        hi = total_frames - 1
    if hi <= lo:
        return []
    if count == 1:
        return [int(round((lo + hi) * 0.5))]

    out = []
    for i in range(count):
        pos = lo + (hi - lo) * ((i + 1) / float(count + 1))
        out.append(int(round(pos)))
    return sorted(set(out))


def detect_auto_bullets(video_path: str, target_count: int, min_gap_sec: float, max_scan_frames: int = 0) -> List[int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    min_gap = max(1, int(round(min_gap_sec * fps)))

    prev = None
    energy = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (320, 180), interpolation=cv2.INTER_AREA)
        if prev is None:
            energy.append(0.0)
        else:
            diff = cv2.absdiff(prev, g)
            energy.append(float(np.mean(diff)))
        prev = g
        idx += 1
        if max_scan_frames > 0 and idx >= max_scan_frames:
            break
    cap.release()

    if len(energy) < 5:
        return []
    arr = np.array(energy, dtype=np.float32)
    kernel = np.ones((9,), dtype=np.float32) / 9.0
    arr_s = np.convolve(arr, kernel, mode="same")

    # local peaks
    peak_idx = []
    for i in range(2, len(arr_s) - 2):
        if arr_s[i] > arr_s[i - 1] and arr_s[i] >= arr_s[i + 1]:
            peak_idx.append(i)
    if len(peak_idx) == 0:
        return []

    peak_idx.sort(key=lambda i: float(arr_s[i]), reverse=True)
    chosen = []
    for i in peak_idx:
        if all(abs(i - c) >= min_gap for c in chosen):
            chosen.append(i)
            if len(chosen) >= target_count:
                break
    chosen.sort()
    return chosen


def orbit_frame(
    frame: np.ndarray,
    t: float,
    orbit_radius_px: float,
    orbit_rot_deg: float,
    orbit_zoom: float,
) -> np.ndarray:
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    theta = 2.0 * math.pi * float(np.clip(t, 0.0, 1.0))
    tx = orbit_radius_px * math.cos(theta)
    ty = orbit_radius_px * 0.42 * math.sin(theta)
    rot = orbit_rot_deg * math.sin(theta)
    scale = 1.0 + (orbit_zoom - 1.0) * (0.55 + 0.45 * math.sin(theta + math.pi * 0.5))

    mat = cv2.getRotationMatrix2D((cx, cy), rot, scale)
    mat[0, 2] += tx
    mat[1, 2] += ty
    out = cv2.warpAffine(frame, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out


def apply_base_fx(
    frame: np.ndarray,
    trail_buf: np.ndarray,
    glow_strength: float,
    trail_decay: float,
    sharpen: float,
    chroma_shift_px: int,
) -> np.ndarray:
    out = frame.astype(np.float32)
    if glow_strength > 1e-4:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, (120, 70, 70), (179, 255, 255))  # purple/magenta
        m2 = cv2.inRange(hsv, (5, 80, 100), (40, 255, 255))    # orange/yellow
        mask = cv2.max(m1, m2)
        colored = frame.astype(np.float32) * (mask[:, :, None].astype(np.float32) / 255.0)
        trail_buf *= trail_decay
        trail_buf += colored * 0.8
        out += trail_buf * glow_strength
    else:
        trail_buf *= trail_decay

    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    if sharpen > 1e-4:
        out_u8 = unsharp(out_u8, amount=1.0 + sharpen, sigma=1.0)
    if chroma_shift_px > 0:
        out_u8 = chromatic_shift(out_u8, px=chroma_shift_px)
    return out_u8


def run(args: argparse.Namespace) -> None:
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input video not found: {args.input}")
    args.slow_smooth_alpha = float(np.clip(args.slow_smooth_alpha, 0.0, 1.0))

    cap_probe = cv2.VideoCapture(args.input)
    if not cap_probe.isOpened():
        raise RuntimeError(f"Cannot open input video: {args.input}")
    fps = cap_probe.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_probe.release()

    centers = parse_time_list(args.bullet_times, fps)
    if args.uniform_bullet_count > 0:
        uniform_centers = generate_uniform_centers(
            total_frames=total_frames,
            fps=fps,
            count=args.uniform_bullet_count,
            margin_sec=args.uniform_margin_sec,
        )
        centers = sorted(set(centers + uniform_centers))
    if args.auto_bullet_count > 0:
        scan_frames = 0 if args.auto_scan_sec <= 0 else int(round(args.auto_scan_sec * fps))
        auto_centers = detect_auto_bullets(
            args.input,
            target_count=args.auto_bullet_count,
            min_gap_sec=args.auto_min_gap_sec,
            max_scan_frames=scan_frames,
        )
        centers = sorted(set(centers + auto_centers))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output writer: {args.output}")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        writer.release()
        raise RuntimeError(f"Cannot open input video (2nd pass): {args.input}")

    trail_buf = np.zeros((h, w, 3), dtype=np.float32)
    frame_idx = 0
    center_set = set(centers)
    slow_left = 0
    prev_base = None
    slow_state = None

    print(f"[INFO] input={args.input}")
    print(f"[INFO] output={args.output}")
    print(f"[INFO] size={w}x{h}, fps={fps:.2f}, frames={total_frames}")
    print(f"[INFO] bullet_count={len(centers)}")
    if len(centers) > 0:
        sec_list = ",".join(f"{c / fps:.2f}" for c in centers[:20])
        print(f"[INFO] bullet_centers_sec={sec_list}")
    else:
        print("[INFO] bullet_centers_sec=(none)")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        base = apply_base_fx(
            frame,
            trail_buf,
            glow_strength=args.glow_strength,
            trail_decay=args.trail_decay,
            sharpen=args.sharpen_strength,
            chroma_shift_px=args.base_chroma_shift_px,
        )

        if frame_idx in center_set:
            freeze_n = max(1, args.freeze_frames)
            for k in range(freeze_n):
                t_raw = 0.0 if freeze_n <= 1 else (k / float(freeze_n - 1))
                # Eased camera timing prevents visible jerk at start/end of freeze orbit.
                t = smoothstep01(t_raw)
                fx = orbit_frame(
                    base,
                    t=t,
                    orbit_radius_px=args.orbit_radius_px,
                    orbit_rot_deg=args.orbit_rot_deg,
                    orbit_zoom=args.orbit_zoom,
                )
                writer.write(fx)
            slow_left = max(slow_left, max(0, args.slow_frames))
            repeat = max(1, args.slow_repeat)
            if args.slow_interp and prev_base is not None:
                frm = prev_base.astype(np.float32)
                to = base.astype(np.float32)
                for r in range(1, repeat + 1):
                    a = smoothstep01(r / float(repeat))
                    target = frm * (1.0 - a) + to * a
                    if slow_state is None:
                        slow_state = target.copy()
                    else:
                        slow_state = slow_state * (1.0 - args.slow_smooth_alpha) + target * args.slow_smooth_alpha
                    writer.write(np.clip(slow_state, 0, 255).astype(np.uint8))
            else:
                for _ in range(repeat):
                    writer.write(base)
        else:
            repeat = max(1, args.slow_repeat) if slow_left > 0 else 1
            if slow_left > 0 and args.slow_interp and prev_base is not None:
                frm = prev_base.astype(np.float32)
                to = base.astype(np.float32)
                for r in range(1, repeat + 1):
                    a = smoothstep01(r / float(repeat))
                    target = frm * (1.0 - a) + to * a
                    if slow_state is None:
                        slow_state = target.copy()
                    else:
                        slow_state = slow_state * (1.0 - args.slow_smooth_alpha) + target * args.slow_smooth_alpha
                    writer.write(np.clip(slow_state, 0, 255).astype(np.uint8))
            else:
                for _ in range(repeat):
                    writer.write(base)
                slow_state = base.astype(np.float32)
            if slow_left > 0:
                slow_left -= 1

        prev_base = base
        frame_idx += 1
        if frame_idx % 300 == 0:
            print(f"[INFO] progress {frame_idx}/{total_frames}")
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

    cap.release()
    writer.release()
    print(f"[DONE] processed_frames={frame_idx}")
    print(f"[DONE] output={args.output}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Add bullet-time and cinematic FX to rendered badminton video")
    p.add_argument("--input", type=str, default=DEFAULT_INPUT)
    p.add_argument("--output", type=str, default=DEFAULT_OUTPUT)

    p.add_argument("--bullet_times", type=str, default="", help="Comma-separated bullet-time center seconds, e.g. 12.4,48.0")
    p.add_argument("--uniform_bullet_count", type=int, default=12, help="Uniformly distribute N bullet-time events across full video")
    p.add_argument("--uniform_margin_sec", type=float, default=12.0, help="Keep uniform events away from start/end edges")
    p.add_argument("--auto_bullet_count", type=int, default=0, help="Auto-detect top N bullet-time peaks")
    p.add_argument("--auto_min_gap_sec", type=float, default=1.8, help="Min gap between auto bullet moments")
    p.add_argument("--auto_scan_sec", type=float, default=0.0, help="Scan only first N seconds for auto peaks (0 = full video)")
    p.add_argument("--freeze_frames", type=int, default=28, help="Instant freeze frame count")
    p.add_argument("--orbit_radius_px", type=float, default=24.0, help="Virtual orbit radius during freeze")
    p.add_argument("--orbit_rot_deg", type=float, default=8.0, help="Virtual camera roll angle during freeze")
    p.add_argument("--orbit_zoom", type=float, default=1.08, help="Virtual camera zoom during freeze")
    p.add_argument("--slow_frames", type=int, default=40, help="How many source frames stay in slow motion after freeze")
    p.add_argument("--slow_repeat", type=int, default=6, help="Extreme slow factor (frame repeat)")
    p.add_argument("--slow_interp", dest="slow_interp", action="store_true", help="Interpolate in slow motion for smoother output")
    p.add_argument("--no_slow_interp", dest="slow_interp", action="store_false", help="Disable slow interpolation")
    p.add_argument("--slow_smooth_alpha", type=float, default=0.42, help="EMA smoothing strength during slow motion interpolation")
    p.set_defaults(slow_interp=True)

    p.add_argument("--glow_strength", type=float, default=0.0, help="Glow trail intensity; 0 keeps original look")
    p.add_argument("--trail_decay", type=float, default=0.90)
    p.add_argument("--sharpen_strength", type=float, default=0.0, help="Base unsharp amount; 0 keeps original look")
    p.add_argument("--base_chroma_shift_px", type=int, default=0, help="Base chroma shift; 0 keeps original look")
    p.add_argument("--max_frames", type=int, default=0)
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run(args)
