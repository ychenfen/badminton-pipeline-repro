#!/usr/bin/env python3
"""BWF Court Coverage Heatmap & Shuttle Hit/Landing Scatter Plot Generator.

Reads either `analytics.json` (exported by `overlay_player_analytics.py --export_json`)
or `*_ball.csv` (with `--court_points`) and renders a publication-grade 2-panel PNG:
  - Panel 1: Player Court Coverage Heatmap (or Shuttle Trajectory Density) on a
    standard 6.1m x 13.4m BWF court.
  - Panel 2: Shuttle Hit / Landing Scatter Plot on BWF court markings.
"""

import argparse
import csv
import json
import math
import os
import struct
import sys
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

DEFAULT_COURT_POINTS = "352,342,628,343,944,527,52,532"


def parse_court_points(value: str) -> List[Tuple[float, float]]:
    tokens = [v.strip() for v in value.split(",") if v.strip() != ""]
    if len(tokens) != 8:
        raise ValueError("court_points must be x1,y1,x2,y2,x3,y3,x4,y4")
    vals = [float(v) for v in tokens]
    return [
        (vals[0], vals[1]),
        (vals[2], vals[3]),
        (vals[4], vals[5]),
        (vals[6], vals[7]),
    ]


def compute_homography_4pt(
    src: Sequence[Tuple[float, float]], dst: Sequence[Tuple[float, float]]
) -> List[List[float]]:
    """Compute exact 3x3 projective homography matrix mapping 4 src pts to 4 dst pts."""
    # Solve 8x8 linear system A * h = b via Gaussian elimination with partial pivoting
    A: List[List[float]] = []
    b: List[float] = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        b.append(u)
        A.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        b.append(v)

    n = 8
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(A[r][col]))
        if abs(A[pivot][col]) < 1e-12:
            raise ValueError("Degenerate court corner points for homography")
        A[col], A[pivot] = A[pivot], A[col]
        b[col], b[pivot] = b[pivot], b[col]
        div = A[col][col]
        for j in range(col, n):
            A[col][j] /= div
        b[col] /= div
        for r in range(n):
            if r != col and abs(A[r][col]) > 1e-15:
                factor = A[r][col]
                for j in range(col, n):
                    A[r][j] -= factor * A[col][j]
                b[r] -= factor * b[col]

    return [
        [b[0], b[1], b[2]],
        [b[3], b[4], b[5]],
        [b[6], b[7], 1.0],
    ]


def project_point(
    pt: Tuple[float, float],
    H: Sequence[Sequence[float]],
    court_w: float = 6.1,
    court_l: float = 13.4,
) -> Optional[Tuple[float, float]]:
    x, y = float(pt[0]), float(pt[1])
    w = H[2][0] * x + H[2][1] * y + H[2][2]
    if abs(w) < 1e-9:
        return None
    mx = (H[0][0] * x + H[0][1] * y + H[0][2]) / w
    my = (H[1][0] * x + H[1][1] * y + H[1][2]) / w
    mx = max(0.0, min(court_w, mx))
    my = max(0.0, min(court_l, my))
    return (mx, my)


def extract_court_data(
    input_path: str,
    court_points_str: str = DEFAULT_COURT_POINTS,
    court_w: float = 6.1,
    court_l: float = 13.4,
) -> Dict[str, Any]:
    """Load player and shuttle court coordinates (in meters) from JSON or CSV."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    player_pts: List[Tuple[float, float]] = []
    ball_pts: List[Tuple[float, float]] = []
    hit_pts: List[Tuple[float, float]] = []

    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".json":
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        court_w = float(data.get("court_width_m", court_w))
        court_l = float(data.get("court_length_m", court_l))
        frames = data.get("frames", [])
        raw_ball_seq: List[Tuple[int, float, float]] = []

        for fr in frames:
            f_idx = int(fr.get("frame", fr.get("i", 0)))
            for side in ("near", "far"):
                side_obj = fr.get(side) or {}
                cm = side_obj.get("court_m")
                if cm and len(cm) == 2:
                    player_pts.append((float(cm[0]), float(cm[1])))
            ball_obj = fr.get("ball") or {}
            bcm = ball_obj.get("court_m")
            if bcm and len(bcm) == 2:
                bx_m, by_m = float(bcm[0]), float(bcm[1])
                ball_pts.append((bx_m, by_m))
                raw_ball_seq.append((f_idx, bx_m, by_m))

        hit_pts = _detect_meter_hits(raw_ball_seq)
    else:
        src_quad = parse_court_points(court_points_str)
        dst_quad = [(0.0, 0.0), (court_w, 0.0), (court_w, court_l), (0.0, court_l)]
        H = compute_homography_4pt(src_quad, dst_quad)

        raw_ball_seq = []
        with open(input_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    f_idx = int(float(row.get("Frame", "0")))
                    vis = int(float(row.get("Visibility", "0")))
                    bx = float(row.get("X", "0"))
                    by = float(row.get("Y", "0"))
                except Exception:
                    continue
                if vis > 0 and (bx != 0.0 or by != 0.0):
                    m = project_point((bx, by), H, court_w, court_l)
                    if m is not None:
                        ball_pts.append(m)
                        raw_ball_seq.append((f_idx, m[0], m[1]))

        hit_pts = _detect_meter_hits(raw_ball_seq)

    return {
        "court_width_m": court_w,
        "court_length_m": court_l,
        "player_pts": player_pts,
        "ball_pts": ball_pts,
        "hit_pts": hit_pts,
    }


def _detect_meter_hits(
    seq: Sequence[Tuple[int, float, float]], min_spacing: int = 6
) -> List[Tuple[float, float]]:
    """Identify trajectory reversal/hit points in court-meter coordinates."""
    if len(seq) < 5:
        return [ (p[1], p[2]) for p in seq ]

    candidates: List[Tuple[int, float, float, float]] = []
    for i in range(2, len(seq) - 2):
        f_prev, x_prev, y_prev = seq[i - 2]
        f_cur, x_cur, y_cur = seq[i]
        f_next, x_next, y_next = seq[i + 2]
        if (f_cur - f_prev) > 12 or (f_next - f_cur) > 12:
            continue
        vx1, vy1 = x_cur - x_prev, y_cur - y_prev
        vx2, vy2 = x_next - x_cur, y_next - y_cur
        s1 = math.hypot(vx1, vy1)
        s2 = math.hypot(vx2, vy2)
        if max(s1, s2) < 0.25:
            continue
        dv = math.hypot(vx2 - vx1, vy2 - vy1)
        cos_t = (vx1 * vx2 + vy1 * vy2) / max(s1 * s2, 1e-6)
        if (vy1 * vy2 < 0 and abs(vy2 - vy1) >= 0.25) or cos_t < 0.35:
            candidates.append((f_cur, dv, x_cur, y_cur))

    candidates.sort(key=lambda item: item[1], reverse=True)
    chosen_frames: List[int] = []
    hits: List[Tuple[float, float]] = []
    for f_cur, _, x_cur, y_cur in candidates:
        if all(abs(f_cur - cf) >= min_spacing for cf in chosen_frames):
            chosen_frames.append(f_cur)
            hits.append((x_cur, y_cur))

    if not hits and seq:
        # Fallback: sample representative points if trajectory is short/linear
        step = max(1, len(seq) // 10)
        hits = [(p[1], p[2]) for p in seq[::step]]
    return hits


def _thermal_color(val: float) -> Tuple[int, int, int]:
    """Map normalized density [0, 1] to an Inferno-like RGB color."""
    v = max(0.0, min(1.0, val))
    stops = [
        (0.00, (14, 22, 38)),
        (0.20, (58, 28, 113)),
        (0.45, (181, 54, 122)),
        (0.70, (237, 105, 37)),
        (0.90, (251, 180, 26)),
        (1.00, (252, 255, 164)),
    ]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t0 <= v <= t1:
            alpha = (v - t0) / max(t1 - t0, 1e-6)
            return (
                int(round(c0[0] + alpha * (c1[0] - c0[0]))),
                int(round(c0[1] + alpha * (c1[1] - c0[1]))),
                int(round(c0[2] + alpha * (c1[2] - c0[2]))),
            )
    return stops[-1][1]


def _write_png_rgb(path: str, width: int, height: int, rgb_bytes: bytes) -> None:
    """Pure-stdlib PNG encoder for an RGB888 buffer of shape (height, width, 3)."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", crc)

    raw_rows = bytearray()
    stride = width * 3
    for y in range(height):
        raw_rows.append(0)  # filter type 0 (None)
        row_start = y * stride
        raw_rows.extend(rgb_bytes[row_start : row_start + stride])

    ihdr = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    compressed = zlib.compress(bytes(raw_rows), level=6)
    png_data = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(png_data)


def render_court_heatmap(
    input_path: str,
    output_path: str = "court_heatmap.png",
    court_points_str: str = DEFAULT_COURT_POINTS,
    court_w: float = 6.1,
    court_l: float = 13.4,
) -> Tuple[str, Dict[str, Any]]:
    """Render a 2-panel BWF court PNG (Panel 1: Coverage Heatmap, Panel 2: Hit Scatter)."""
    data = extract_court_data(
        input_path=input_path,
        court_points_str=court_points_str,
        court_w=court_w,
        court_l=court_l,
    )
    court_w = data["court_width_m"]
    court_l = data["court_length_m"]
    density_pts = data["player_pts"] if len(data["player_pts"]) > 0 else data["ball_pts"]
    ball_pts = data["ball_pts"]
    hit_pts = data["hit_pts"]

    # Canvas geometry (2 panels side-by-side)
    img_w, img_h = 920, 620
    court_px_w, court_px_h = 244, 536  # Exact 6.1 : 13.4 aspect ratio (40 px / meter)
    p1_x0, p1_y0 = 88, 52
    p2_x0, p2_y0 = 548, 52

    # Initialize RGB buffer with dark slate background (11, 15, 25)
    buf = bytearray([11, 15, 25] * (img_w * img_h))

    def set_px(x: int, y: int, c: Tuple[int, int, int], alpha: float = 1.0) -> None:
        if 0 <= x < img_w and 0 <= y < img_h:
            idx = (y * img_w + x) * 3
            if alpha >= 0.999:
                buf[idx] = c[0]
                buf[idx + 1] = c[1]
                buf[idx + 2] = c[2]
            else:
                inv = 1.0 - alpha
                buf[idx] = int(buf[idx] * inv + c[0] * alpha)
                buf[idx + 1] = int(buf[idx + 1] * inv + c[1] * alpha)
                buf[idx + 2] = int(buf[idx + 2] * inv + c[2] * alpha)

    def fill_rect(x0: int, y0: int, w: int, h: int, c: Tuple[int, int, int]) -> None:
        for yy in range(max(0, y0), min(img_h, y0 + h)):
            row_base = yy * img_w * 3
            for xx in range(max(0, x0), min(img_w, x0 + w)):
                idx = row_base + xx * 3
                buf[idx] = c[0]
                buf[idx + 1] = c[1]
                buf[idx + 2] = c[2]

    def draw_hline(x0: int, x1: int, y: int, c: Tuple[int, int, int], thick: int = 2) -> None:
        for dy in range(-(thick // 2), (thick + 1) // 2):
            for xx in range(min(x0, x1), max(x0, x1) + 1):
                set_px(xx, y + dy, c)

    def draw_vline(x: int, y0: int, y1: int, c: Tuple[int, int, int], thick: int = 2) -> None:
        for dx in range(-(thick // 2), (thick + 1) // 2):
            for yy in range(min(y0, y1), max(y0, y1) + 1):
                set_px(x + dx, yy, c)

    def draw_circle(cx: int, cy: int, r: int, c: Tuple[int, int, int], fill: bool = True, alpha: float = 1.0) -> None:
        r2 = r * r
        inner2 = max(0, (r - 2) * (r - 2))
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                d2 = dx * dx + dy * dy
                if d2 <= r2 and (fill or d2 >= inner2):
                    set_px(cx + dx, cy + dy, c, alpha)

    def draw_bwf_court_lines(cx0: int, cy0: int) -> None:
        line_c = (225, 232, 240)
        sub_c = (148, 163, 184)
        net_c = (56, 189, 248)

        # Outer doubles boundary
        draw_hline(cx0, cx0 + court_px_w, cy0, line_c, 2)
        draw_hline(cx0, cx0 + court_px_w, cy0 + court_px_h, line_c, 2)
        draw_vline(cx0, cy0, cy0 + court_px_h, line_c, 2)
        draw_vline(cx0 + court_px_w, cy0, cy0 + court_px_h, line_c, 2)

        # Singles sidelines (0.46m inset on each side)
        s_in = int(round((0.46 / court_w) * court_px_w))
        draw_vline(cx0 + s_in, cy0, cy0 + court_px_h, sub_c, 1)
        draw_vline(cx0 + court_px_w - s_in, cy0, cy0 + court_px_h, sub_c, 1)

        # Short service lines (1.98m from net at y = 6.7m)
        net_y = cy0 + court_px_h // 2
        short_offset = int(round((1.98 / court_l) * court_px_h))
        top_short_y = net_y - short_offset
        bot_short_y = net_y + short_offset
        draw_hline(cx0, cx0 + court_px_w, top_short_y, sub_c, 1)
        draw_hline(cx0, cx0 + court_px_w, bot_short_y, sub_c, 1)

        # Long service lines for doubles (0.76m from baselines)
        long_offset = int(round((0.76 / court_l) * court_px_h))
        draw_hline(cx0, cx0 + court_px_w, cy0 + long_offset, sub_c, 1)
        draw_hline(cx0, cx0 + court_px_w, cy0 + court_px_h - long_offset, sub_c, 1)

        # Center line (between short service line and baseline in each half)
        mid_x = cx0 + court_px_w // 2
        draw_vline(mid_x, cy0, top_short_y, sub_c, 1)
        draw_vline(mid_x, bot_short_y, cy0 + court_px_h, sub_c, 1)

        # Net line (y = 6.7m)
        draw_hline(cx0 - 6, cx0 + court_px_w + 6, net_y, net_c, 3)

    # Panel 1: Compute 2D Gaussian-smoothed density grid (61 x 134 cells)
    gw, gh = 61, 134
    grid = [[0.0 for _ in range(gw)] for _ in range(gh)]
    sigma = 3.2
    rad = 8
    two_s2 = 2.0 * sigma * sigma

    for mx, my in density_pts:
        gx = int(max(0, min(gw - 1, round((mx / max(court_w, 1e-6)) * (gw - 1)))))
        gy = int(max(0, min(gh - 1, round((my / max(court_l, 1e-6)) * (gh - 1)))))
        for dy in range(-rad, rad + 1):
            ny = gy + dy
            if 0 <= ny < gh:
                for dx in range(-rad, rad + 1):
                    nx = gx + dx
                    if 0 <= nx < gw:
                        w_val = math.exp(-(dx * dx + dy * dy) / two_s2)
                        grid[ny][nx] += w_val

    max_val = max((max(row) for row in grid), default=0.0)

    # Render Panel 1 heatmap surface
    for py in range(court_px_h):
        gy_f = (py / float(court_px_h - 1)) * (gh - 1)
        gy_i = int(min(gh - 1, max(0, round(gy_f))))
        for px in range(court_px_w):
            gx_f = (px / float(court_px_w - 1)) * (gw - 1)
            gx_i = int(min(gw - 1, max(0, round(gx_f))))
            norm_v = (grid[gy_i][gx_i] / max_val) if max_val > 1e-9 else 0.0
            # Apply slight gamma curve for visual contrast
            color = _thermal_color(norm_v ** 0.75)
            set_px(p1_x0 + px, p1_y0 + py, color)

    draw_bwf_court_lines(p1_x0, p1_y0)

    # Render Panel 2 court surface and scatter points
    fill_rect(p2_x0, p2_y0, court_px_w, court_px_h, (18, 42, 34))
    draw_bwf_court_lines(p2_x0, p2_y0)

    # Plot subtle shuttle trajectory points
    for mx, my in ball_pts:
        px = p2_x0 + int(round((mx / max(court_w, 1e-6)) * (court_px_w - 1)))
        py = p2_y0 + int(round((my / max(court_l, 1e-6)) * (court_px_h - 1)))
        draw_circle(px, py, 2, (56, 189, 248), fill=True, alpha=0.45)

    # Plot highlighted shuttle hit/landing scatter points
    for mx, my in hit_pts:
        px = p2_x0 + int(round((mx / max(court_w, 1e-6)) * (court_px_w - 1)))
        py = p2_y0 + int(round((my / max(court_l, 1e-6)) * (court_px_h - 1)))
        draw_circle(px, py, 7, (251, 191, 36), fill=False, alpha=0.90)
        draw_circle(px, py, 4, (239, 68, 68), fill=True, alpha=1.0)

    _write_png_rgb(output_path, img_w, img_h, bytes(buf))
    summary = {
        "output_path": output_path,
        "width": img_w,
        "height": img_h,
        "num_density_points": len(density_pts),
        "num_ball_points": len(ball_pts),
        "num_hit_points": len(hit_pts),
    }
    print(
        f"[heatmap] rendered 2-panel BWF court heatmap ({img_w}x{img_h}) | "
        f"density_pts={len(density_pts)}, ball_pts={len(ball_pts)}, hits={len(hit_pts)} -> {output_path}"
    )
    return output_path, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a 2-panel BWF court coverage heatmap and hit/landing scatter plot PNG."
    )
    parser.add_argument("input_path", nargs="?", default="", help="Path to analytics.json or *_ball.csv")
    parser.add_argument("--input", "--in_json", "--ball_csv", dest="input_flag", default="", help="Path to analytics.json or *_ball.csv")
    parser.add_argument("--output", "--out_png", dest="output_path", default="court_heatmap.png", help="Output PNG path (default: court_heatmap.png)")
    parser.add_argument(
        "--court_points",
        type=str,
        default=DEFAULT_COURT_POINTS,
        help="4 court corners x1,y1,x2,y2,x3,y3,x4,y4 when reading *_ball.csv",
    )
    parser.add_argument("--court_width_m", type=float, default=6.1, help="BWF court width in meters (default: 6.1)")
    parser.add_argument("--court_length_m", type=float, default=13.4, help="BWF court length in meters (default: 13.4)")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    in_path = args.input_flag or args.input_path
    if not in_path:
        parser.error("An input path (analytics.json or *_ball.csv) is required.")

    render_court_heatmap(
        input_path=in_path,
        output_path=args.output_path,
        court_points_str=args.court_points,
        court_w=args.court_width_m,
        court_l=args.court_length_m,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
