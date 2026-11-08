#!/usr/bin/env python3
"""Shuttle trajectory gap interpolation and 2D constant-velocity Kalman smoothing.

Interpolates short missing-detection gaps (<= max_gap frames) linearly, leaves
longer gaps as Visibility=0 (out-of-play), and smooths active segments with a
4-state [x, y, vx, vy] constant-velocity 2D Kalman filter (uses NumPy/Pandas
when installed, with a zero-dependency stdlib fallback).

Outputs CSV with columns: Frame,Visibility,X,Y,Source ('model', 'interp', 'none').
"""

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore


class KalmanFilter2D:
    """4-state [x, y, vx, vy] constant-velocity 2D Kalman filter."""

    def __init__(self, dt: float = 1.0, process_noise: float = 12.0, meas_noise: float = 6.0):
        self.dt = float(dt)
        self.process_noise = float(process_noise)
        self.meas_noise = float(meas_noise)

        if np is not None:
            dt2 = 0.5 * self.dt * self.dt
            self.F = np.array(
                [
                    [1.0, 0.0, self.dt, 0.0],
                    [0.0, 1.0, 0.0, self.dt],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            self.H = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )
            q = self.process_noise
            self.Q = q * np.array(
                [
                    [dt2 * dt2, 0.0, dt2 * self.dt, 0.0],
                    [0.0, dt2 * dt2, 0.0, dt2 * self.dt],
                    [dt2 * self.dt, 0.0, self.dt * self.dt, 0.0],
                    [0.0, dt2 * self.dt, 0.0, self.dt * self.dt],
                ],
                dtype=np.float64,
            )
            r = self.meas_noise
            self.R = np.array([[r, 0.0], [0.0, r]], dtype=np.float64)
            self.I = np.eye(4, dtype=np.float64)

    def _filter_1d_list(self, zs: Sequence[float]) -> List[float]:
        """Exact 2-state [pos, vel] 1D Kalman filter + RTS smoother (since X and Y are decoupled)."""
        n = len(zs)
        if n <= 2:
            return [float(z) for z in zs]

        dt = self.dt
        dt2 = 0.5 * dt * dt
        q = self.process_noise
        r = self.meas_noise

        q00 = q * dt2 * dt2
        q01 = q * dt2 * dt
        q11 = q * dt * dt

        x0 = float(zs[0])
        x1 = (float(zs[1]) - float(zs[0])) / dt
        p00, p01, p10, p11 = r, 0.0, 0.0, 100.0

        filt_x: List[Tuple[float, float]] = []
        filt_p: List[Tuple[float, float, float, float]] = []
        pred_x: List[Tuple[float, float]] = []
        pred_p: List[Tuple[float, float, float, float]] = []

        for i in range(n):
            if i > 0:
                xp0 = x0 + dt * x1
                xp1 = x1
                pp00 = p00 + dt * (p10 + p01) + dt * dt * p11 + q00
                pp01 = p01 + dt * p11 + q01
                pp10 = p10 + dt * p11 + q01
                pp11 = p11 + q11
            else:
                xp0, xp1 = x0, x1
                pp00, pp01, pp10, pp11 = p00, p01, p10, p11

            pred_x.append((xp0, xp1))
            pred_p.append((pp00, pp01, pp10, pp11))

            y_res = float(zs[i]) - xp0
            s = pp00 + r
            k0 = pp00 / s
            k1 = pp10 / s

            x0 = xp0 + k0 * y_res
            x1 = xp1 + k1 * y_res
            p00 = (1.0 - k0) * pp00
            p01 = (1.0 - k0) * pp01
            p10 = pp10 - k1 * pp00
            p11 = pp11 - k1 * pp01

            filt_x.append((x0, x1))
            filt_p.append((p00, p01, p10, p11))

        # RTS backward smoothing pass
        sm_x = list(filt_x)
        for i in range(n - 2, -1, -1):
            fp00, fp01, fp10, fp11 = filt_p[i]
            pp00, pp01, pp10, pp11 = pred_p[i + 1]
            # Pf * F^T = [[fp00 + dt*fp01, fp01], [fp10 + dt*fp11, fp11]]
            a00 = fp00 + dt * fp01
            a01 = fp01
            a10 = fp10 + dt * fp11
            a11 = fp11
            det = pp00 * pp11 - pp01 * pp10
            if abs(det) > 1e-12:
                inv00 = pp11 / det
                inv01 = -pp01 / det
                inv10 = -pp10 / det
                inv11 = pp00 / det
                c00 = a00 * inv00 + a01 * inv10
                c01 = a00 * inv01 + a01 * inv11
                c10 = a10 * inv00 + a11 * inv10
                c11 = a10 * inv01 + a11 * inv11
                dx0 = sm_x[i + 1][0] - pred_x[i + 1][0]
                dx1 = sm_x[i + 1][1] - pred_x[i + 1][1]
                sm_x[i] = (
                    filt_x[i][0] + c00 * dx0 + c01 * dx1,
                    filt_x[i][1] + c10 * dx0 + c11 * dx1,
                )

        return [pt[0] for pt in sm_x]

    def filter_segment(self, points: Any) -> Any:
        """Apply forward 4-state [x, y, vx, vy] Kalman filter + RTS smoother to a (N, 2) segment."""
        if np is not None and isinstance(points, np.ndarray):
            n = len(points)
            if n <= 2:
                return points.astype(np.float64).copy()

            vx0 = float(points[1, 0] - points[0, 0]) / self.dt
            vy0 = float(points[1, 1] - points[0, 1]) / self.dt
            x = np.array([float(points[0, 0]), float(points[0, 1]), vx0, vy0], dtype=np.float64)
            P = np.diag([self.R[0, 0], self.R[1, 1], 100.0, 100.0]).astype(np.float64)

            xs_filt = np.zeros((n, 4), dtype=np.float64)
            Ps_filt = np.zeros((n, 4, 4), dtype=np.float64)
            xs_pred = np.zeros((n, 4), dtype=np.float64)
            Ps_pred = np.zeros((n, 4, 4), dtype=np.float64)

            for i in range(n):
                if i > 0:
                    x_pred = self.F @ x
                    P_pred = self.F @ P @ self.F.T + self.Q
                else:
                    x_pred = x.copy()
                    P_pred = P.copy()

                xs_pred[i] = x_pred
                Ps_pred[i] = P_pred

                z = points[i].astype(np.float64)
                y_res = z - (self.H @ x_pred)
                S = self.H @ P_pred @ self.H.T + self.R
                K = P_pred @ self.H.T @ np.linalg.inv(S)
                x = x_pred + K @ y_res
                P = (self.I - K @ self.H) @ P_pred

                xs_filt[i] = x
                Ps_filt[i] = P

            xs_smooth = xs_filt.copy()
            for i in range(n - 2, -1, -1):
                P_pred_next = Ps_pred[i + 1]
                C = Ps_filt[i] @ self.F.T @ np.linalg.pinv(P_pred_next)
                xs_smooth[i] = xs_filt[i] + C @ (xs_smooth[i + 1] - xs_pred[i + 1])

            return xs_smooth[:, :2]

        pts_list = list(points)
        xs = [float(p[0]) for p in pts_list]
        ys = [float(p[1]) for p in pts_list]
        sx = self._filter_1d_list(xs)
        sy = self._filter_1d_list(ys)
        res = list(zip(sx, sy))
        return np.array(res, dtype=np.float64) if np is not None else res


def smooth_ball_rows(
    rows: Sequence[Dict[str, Any]],
    max_gap: int = 8,
    process_noise: float = 12.0,
    meas_noise: float = 6.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Interpolate short gaps and Kalman-smooth visible/interpolated segments on row dicts."""
    n = len(rows)
    frames: List[int] = []
    vis: List[int] = []
    x_out: List[float] = []
    y_out: List[float] = []
    sources: List[str] = []

    for r in rows:
        f_i = int(float(r.get("Frame", 0)))
        v_i = int(float(r.get("Visibility", 0)))
        x_i = float(r.get("X", 0))
        y_i = float(r.get("Y", 0))
        is_vis = 1 if (v_i > 0 and (x_i != 0.0 or y_i != 0.0)) else 0
        frames.append(f_i)
        vis.append(is_vis)
        x_out.append(x_i if is_vis else 0.0)
        y_out.append(y_i if is_vis else 0.0)
        sources.append("model" if is_vis else "none")

    raw_visible_count = sum(vis)

    # Step 1: Linearly interpolate internal gaps of length <= max_gap
    i = 0
    while i < n:
        if vis[i] == 0:
            gap_start = i
            while i < n and vis[i] == 0:
                i += 1
            gap_end = i
            gap_len = gap_end - gap_start
            if gap_start > 0 and gap_end < n and 1 <= gap_len <= max_gap:
                x0, y0 = x_out[gap_start - 1], y_out[gap_start - 1]
                x1, y1 = x_out[gap_end], y_out[gap_end]
                for k in range(gap_len):
                    alpha = float(k + 1) / float(gap_len + 1)
                    idx = gap_start + k
                    x_out[idx] = x0 + alpha * (x1 - x0)
                    y_out[idx] = y0 + alpha * (y1 - y0)
                    vis[idx] = 1
                    sources[idx] = "interp"
        else:
            i += 1

    # Step 2: Apply 2D Kalman smoothing to each contiguous active segment
    kf = KalmanFilter2D(dt=1.0, process_noise=process_noise, meas_noise=meas_noise)
    i = 0
    while i < n:
        if vis[i] > 0:
            seg_start = i
            while i < n and vis[i] > 0:
                i += 1
            seg_end = i
            seg_pts = list(zip(x_out[seg_start:seg_end], y_out[seg_start:seg_end]))
            if np is not None:
                smoothed = kf.filter_segment(np.array(seg_pts, dtype=np.float64))
            else:
                smoothed = kf.filter_segment(seg_pts)
            for idx_offset, pt in enumerate(smoothed):
                x_out[seg_start + idx_offset] = max(0, int(round(float(pt[0]))))
                y_out[seg_start + idx_offset] = max(0, int(round(float(pt[1]))))
        else:
            x_out[i] = 0
            y_out[i] = 0
            i += 1

    post_visible_count = sum(vis)
    interp_count = sum(1 for s in sources if s == "interp")

    out_rows = [
        {
            "Frame": int(frames[k]),
            "Visibility": int(vis[k]),
            "X": int(round(x_out[k])),
            "Y": int(round(y_out[k])),
            "Source": sources[k],
        }
        for k in range(n)
    ]
    stats = {
        "total_frames": n,
        "raw_visible_frames": raw_visible_count,
        "interp_frames": interp_count,
        "post_visible_frames": post_visible_count,
        "raw_detection_rate": (raw_visible_count / float(n)) if n > 0 else 0.0,
        "post_coverage_rate": (post_visible_count / float(n)) if n > 0 else 0.0,
    }
    return out_rows, stats


def smooth_ball_dataframe(
    df: Any,
    max_gap: int = 8,
    process_noise: float = 12.0,
    meas_noise: float = 6.0,
) -> Tuple[Any, Dict[str, float]]:
    """Interpolate short gaps and Kalman-smooth a pandas DataFrame (or list of dicts)."""
    if pd is not None and isinstance(df, pd.DataFrame):
        rows = df.to_dict(orient="records")
        out_rows, stats = smooth_ball_rows(
            rows, max_gap=max_gap, process_noise=process_noise, meas_noise=meas_noise
        )
        return pd.DataFrame(out_rows), stats
    return smooth_ball_rows(
        list(df), max_gap=max_gap, process_noise=process_noise, meas_noise=meas_noise
    )


def smooth_ball_csv(
    in_csv: str,
    out_csv: Optional[str] = None,
    max_gap: int = 8,
    process_noise: float = 12.0,
    meas_noise: float = 6.0,
) -> Tuple[str, Dict[str, float]]:
    """Read raw TrackNet *_ball.csv, interpolate + Kalman smooth, and save CSV."""
    if not os.path.exists(in_csv):
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    if not out_csv:
        stem, ext = os.path.splitext(in_csv)
        out_csv = f"{stem}_smoothed{ext or '.csv'}"

    with open(in_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    out_rows, stats = smooth_ball_rows(
        rows,
        max_gap=max_gap,
        process_noise=process_noise,
        meas_noise=meas_noise,
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Frame", "Visibility", "X", "Y", "Source"])
        writer.writeheader()
        writer.writerows(out_rows)

    print(
        f"[smooth_ball_csv] total={stats['total_frames']} frames | "
        f"raw={stats['raw_visible_frames']} ({stats['raw_detection_rate']*100:.1f}%) | "
        f"interp=+{stats['interp_frames']} | "
        f"final={stats['post_visible_frames']} ({stats['post_coverage_rate']*100:.1f}%) -> {out_csv}"
    )
    return out_csv, stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interpolate short missing shuttle gaps and apply 2D Kalman smoothing."
    )
    parser.add_argument("input_csv", nargs="?", default="", help="Path to input *_ball.csv")
    parser.add_argument("--in_csv", "--input", dest="in_csv_flag", default="", help="Path to input *_ball.csv")
    parser.add_argument("--out_csv", "--output", dest="out_csv", default="", help="Path to output smoothed CSV")
    parser.add_argument("--inplace", action="store_true", help="Overwrite input CSV in place")
    parser.add_argument("--max_gap", type=int, default=8, help="Max missing gap (frames) to interpolate (default: 8)")
    parser.add_argument("--process_noise", type=float, default=12.0, help="Kalman process noise q (default: 12.0)")
    parser.add_argument("--meas_noise", type=float, default=6.0, help="Kalman measurement noise r (default: 6.0)")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    in_csv = args.in_csv_flag or args.input_csv
    if not in_csv:
        parser.error("An input CSV path is required (positional or --in_csv).")

    out_csv = in_csv if args.inplace else (args.out_csv or None)
    smooth_ball_csv(
        in_csv=in_csv,
        out_csv=out_csv,
        max_gap=args.max_gap,
        process_noise=args.process_noise,
        meas_noise=args.meas_noise,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
