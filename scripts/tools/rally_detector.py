#!/usr/bin/env python3
"""Automatic Badminton Rally Boundary Segmentation, Hit Detection & FFmpeg Clipper.

Provides both an importable `RallyDetector` class and a CLI tool to:
1. Segment match videos into individual rallies `[start_frame, end_frame]` from `*_ball.csv`.
2. Detect shuttlecock hit events (racket contacts / trajectory reversals) within each rally.
3. Export structured JSON (`rallies.json`) and CSV (`rallies.csv`) analytics reports.
4. Batch-clip individual rally MP4s and concatenate a `highlights.mp4` reel via FFmpeg.
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore


class RallyDetector:
    """Reusable rally boundary and hit-event detector from TrackNet shuttle trajectories."""

    def __init__(
        self,
        fps: float = 21.0,
        min_rally_frames: int = 30,
        max_missing_frames: int = 25,
        min_hit_spacing: int = 6,
        min_hit_speed_px: float = 3.0,
        stationary_window: int = 10,
        stationary_spread_px: float = 4.0,
    ):
        self.fps = float(max(fps, 1e-3))
        self.min_rally_frames = int(min_rally_frames)
        self.max_missing_frames = int(max_missing_frames)
        self.min_hit_spacing = int(max(1, min_hit_spacing))
        self.min_hit_speed_px = float(min_hit_speed_px)
        self.stationary_window = int(max(3, stationary_window))
        self.stationary_spread_px = float(stationary_spread_px)
        self.last_rallies: List[Dict[str, Any]] = []

    def _compute_active_mask(
        self, vis: Sequence[int], xs: Sequence[float], ys: Sequence[float]
    ) -> List[bool]:
        """Determine per-frame active play mask (visible and not stationary on floor)."""
        n = len(vis)
        active = [(int(vis[i]) > 0 and (float(xs[i]) != 0.0 or float(ys[i]) != 0.0)) for i in range(n)]
        if n < self.stationary_window:
            return active

        win = self.stationary_window
        suppressed = list(active)
        for i in range(n - win + 1):
            if all(active[i : i + win]):
                wx = xs[i : i + win]
                wy = ys[i : i + win]
                spread = math.hypot(max(wx) - min(wx), max(wy) - min(wy))
                if spread <= self.stationary_spread_px:
                    for k in range(i, i + win):
                        suppressed[k] = False
        return suppressed

    def detect_hits_in_segment(
        self,
        frames: Sequence[int],
        xs: Sequence[float],
        ys: Sequence[float],
        vis: Sequence[int],
    ) -> List[int]:
        """Detect racket hit frames within a single rally segment.

        Hits are identified where the shuttle velocity vector reverses or turns sharply,
        with non-maximum suppression enforcing `min_hit_spacing` frames between hits.
        """
        n = len(frames)
        if n < 5:
            return []

        valid_idx = [i for i in range(n) if int(vis[i]) > 0 and (float(xs[i]) != 0.0 or float(ys[i]) != 0.0)]
        if len(valid_idx) < 5:
            return []

        # Linearly interpolate internal missing positions across the rally
        x_interp: List[float] = [0.0] * n
        y_interp: List[float] = [0.0] * n
        first_v = valid_idx[0]
        last_v = valid_idx[-1]
        for i in range(first_v + 1):
            x_interp[i] = float(xs[first_v])
            y_interp[i] = float(ys[first_v])
        for i in range(last_v, n):
            x_interp[i] = float(xs[last_v])
            y_interp[i] = float(ys[last_v])
        for p in range(len(valid_idx) - 1):
            i0, i1 = valid_idx[p], valid_idx[p + 1]
            x0, y0 = float(xs[i0]), float(ys[i0])
            x1, y1 = float(xs[i1]), float(ys[i1])
            span = max(1, i1 - i0)
            for k in range(i0, i1 + 1):
                alpha = float(k - i0) / float(span)
                x_interp[k] = x0 + alpha * (x1 - x0)
                y_interp[k] = y0 + alpha * (y1 - y0)

        # Light 3-point smoothing kernel [0.25, 0.5, 0.25] to reduce pixel quantization jitter
        x_smooth: List[float] = [0.0] * n
        y_smooth: List[float] = [0.0] * n
        for i in range(n):
            im1 = max(0, i - 1)
            ip1 = min(n - 1, i + 1)
            x_smooth[i] = 0.25 * x_interp[im1] + 0.50 * x_interp[i] + 0.25 * x_interp[ip1]
            y_smooth[i] = 0.25 * y_interp[im1] + 0.50 * y_interp[i] + 0.25 * y_interp[ip1]

        candidates: List[Tuple[int, float]] = []
        step = 2
        for i in range(step, n - step):
            vx_in = (x_smooth[i] - x_smooth[i - step]) / float(step)
            vy_in = (y_smooth[i] - y_smooth[i - step]) / float(step)
            vx_out = (x_smooth[i + step] - x_smooth[i]) / float(step)
            vy_out = (y_smooth[i + step] - y_smooth[i]) / float(step)

            speed_in = math.hypot(vx_in, vy_in)
            speed_out = math.hypot(vx_out, vy_out)
            max_speed = max(speed_in, speed_out)
            if max_speed < self.min_hit_speed_px:
                continue

            dv = math.hypot(vx_out - vx_in, vy_out - vy_in)
            denom = max(speed_in * speed_out, 1e-6)
            cos_theta = (vx_in * vx_out + vy_in * vy_out) / denom

            y_reversal = (vy_in * vy_out < 0) and (abs(vy_out - vy_in) >= self.min_hit_speed_px)
            x_reversal = (vx_in * vx_out < 0) and (abs(vx_out - vx_in) >= self.min_hit_speed_px)
            sharp_turn = (
                speed_in >= 0.5 * self.min_hit_speed_px
                and speed_out >= 0.5 * self.min_hit_speed_px
                and cos_theta < 0.35
                and dv >= self.min_hit_speed_px
            )
            sudden_launch = (
                speed_in < 0.5 * self.min_hit_speed_px
                and speed_out >= 1.5 * self.min_hit_speed_px
            )

            if y_reversal or x_reversal or sharp_turn or sudden_launch:
                candidates.append((i, dv))

        if not candidates:
            return []

        candidates_sorted = sorted(candidates, key=lambda item: item[1], reverse=True)
        selected_local: List[int] = []
        for idx, _ in candidates_sorted:
            if all(abs(idx - sel) >= self.min_hit_spacing for sel in selected_local):
                selected_local.append(idx)

        selected_local.sort()
        return [int(frames[idx]) for idx in selected_local]

    def detect_from_rows(self, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Segment rallies and detect hits from a sequence of TrackNet ball row dicts."""
        if not rows:
            self.last_rallies = []
            return []

        frames = [int(float(r.get("Frame", 0))) for r in rows]
        vis = [int(float(r.get("Visibility", 0))) for r in rows]
        xs = [float(r.get("X", 0)) for r in rows]
        ys = [float(r.get("Y", 0)) for r in rows]

        active = self._compute_active_mask(vis, xs, ys)
        n = len(frames)

        # Bridge missing gaps <= max_missing_frames between active detections
        bridged = list(active)
        i = 0
        while i < n:
            if not bridged[i]:
                gap_start = i
                while i < n and not bridged[i]:
                    i += 1
                gap_end = i
                gap_len = gap_end - gap_start
                if gap_start > 0 and gap_end < n and gap_len <= self.max_missing_frames:
                    for k in range(gap_start, gap_end):
                        bridged[k] = True
            else:
                i += 1

        rallies: List[Dict[str, Any]] = []
        i = 0
        rally_id = 1
        while i < n:
            if bridged[i]:
                seg_start = i
                while i < n and bridged[i]:
                    i += 1
                seg_end = i - 1

                while seg_start < seg_end and not active[seg_start]:
                    seg_start += 1
                while seg_end > seg_start and not active[seg_end]:
                    seg_end -= 1

                start_frame = int(frames[seg_start])
                end_frame = int(frames[seg_end])
                duration_frames = end_frame - start_frame + 1

                if duration_frames >= self.min_rally_frames:
                    hit_frames = self.detect_hits_in_segment(
                        frames[seg_start : seg_end + 1],
                        xs[seg_start : seg_end + 1],
                        ys[seg_start : seg_end + 1],
                        vis[seg_start : seg_end + 1],
                    )
                    start_sec = round(start_frame / self.fps, 3)
                    end_sec = round(end_frame / self.fps, 3)
                    duration_sec = round(duration_frames / self.fps, 3)
                    rallies.append(
                        {
                            "rally_id": rally_id,
                            "start_frame": start_frame,
                            "end_frame": end_frame,
                            "start_sec": start_sec,
                            "end_sec": end_sec,
                            "duration_sec": duration_sec,
                            "num_hits": len(hit_frames),
                            "hit_frames": hit_frames,
                        }
                    )
                    rally_id += 1
            else:
                i += 1

        self.last_rallies = rallies
        return rallies

    def detect_from_dataframe(self, df: Any) -> List[Dict[str, Any]]:
        """Segment rallies and detect hits from a TrackNet ball DataFrame or list of dicts."""
        if pd is not None and isinstance(df, pd.DataFrame):
            if df.empty:
                self.last_rallies = []
                return []
            return self.detect_from_rows(df.to_dict(orient="records"))
        return self.detect_from_rows(list(df))

    def detect_from_csv(self, csv_path: str) -> List[Dict[str, Any]]:
        """Load TrackNet `*_ball.csv` and return detected rally dictionaries."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Ball CSV not found: {csv_path}")
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return self.detect_from_rows(rows)

    def detect(
        self, source: Union[str, os.PathLike, Any]
    ) -> List[Dict[str, Any]]:
        """Detect rallies from a CSV file path, DataFrame, or sequence of row dicts."""
        if isinstance(source, (str, os.PathLike)):
            return self.detect_from_csv(str(source))
        return self.detect_from_dataframe(source)

    def get_boundaries(
        self, rallies: Optional[List[Dict[str, Any]]] = None
    ) -> List[Tuple[int, int]]:
        """Return list of `(start_frame, end_frame)` tuples for detected rallies."""
        target = rallies if rallies is not None else self.last_rallies
        return [(int(r["start_frame"]), int(r["end_frame"])) for r in target]

    def export_json(
        self,
        rallies: List[Dict[str, Any]],
        out_json: str,
        source_csv: str = "",
    ) -> str:
        """Write rally segmentation and hit analytics to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        total_duration = round(sum(float(r["duration_sec"]) for r in rallies), 3)
        total_hits = sum(int(r["num_hits"]) for r in rallies)
        payload = {
            "source_csv": source_csv,
            "fps": self.fps,
            "min_rally_frames": self.min_rally_frames,
            "max_missing_frames": self.max_missing_frames,
            "num_rallies": len(rallies),
            "total_rally_duration_sec": total_duration,
            "total_hits": total_hits,
            "rallies": rallies,
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return out_json

    def export_csv(self, rallies: List[Dict[str, Any]], out_csv: str) -> str:
        """Write rally segmentation summary to CSV."""
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        fieldnames = [
            "rally_id",
            "start_frame",
            "end_frame",
            "start_sec",
            "end_sec",
            "duration_sec",
            "num_hits",
            "hit_frames",
        ]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rallies:
                row = dict(r)
                row["hit_frames"] = ";".join(str(h) for h in r.get("hit_frames", []))
                writer.writerow(row)
        return out_csv

    def cut_rallies(
        self,
        video_path: str,
        rallies: List[Dict[str, Any]],
        out_dir: str,
        concat_highlights: bool = True,
        pad_before_sec: float = 0.25,
        pad_after_sec: float = 0.35,
    ) -> List[str]:
        """Batch-clip rally videos using FFmpeg and optionally concatenate `highlights.mp4`."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")
        os.makedirs(out_dir, exist_ok=True)

        clip_paths: List[str] = []
        for r in rallies:
            rid = int(r["rally_id"])
            start_s = max(0.0, float(r["start_sec"]) - pad_before_sec)
            end_s = float(r["end_sec"]) + pad_after_sec
            clip_path = os.path.join(out_dir, f"rally_{rid:02d}.mp4")
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                f"{start_s:.3f}",
                "-to",
                f"{end_s:.3f}",
                "-i",
                video_path,
                "-c",
                "copy",
                clip_path,
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            clip_paths.append(clip_path)

        if concat_highlights and len(clip_paths) > 0:
            list_file = os.path.join(out_dir, "concat_list.txt")
            with open(list_file, "w", encoding="utf-8") as f:
                for p in clip_paths:
                    abs_p = os.path.abspath(p).replace("'", "'\\''")
                    f.write(f"file '{abs_p}'\n")
            highlights_path = os.path.join(out_dir, "highlights.mp4")
            concat_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_file,
                "-c",
                "copy",
                highlights_path,
            ]
            subprocess.run(concat_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return clip_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect badminton rally boundaries, hit events, and optionally cut rally clips via FFmpeg."
    )
    parser.add_argument("ball_csv", nargs="?", default="", help="Path to TrackNet *_ball.csv")
    parser.add_argument("--ball_csv", "--in_csv", dest="ball_csv_flag", default="", help="Path to TrackNet *_ball.csv")
    parser.add_argument("--fps", type=float, default=21.0, help="Video frame rate (default: 21.0)")
    parser.add_argument("--min_rally_frames", type=int, default=30, help="Minimum frames for a valid rally (default: 30)")
    parser.add_argument(
        "--max_missing_frames",
        type=int,
        default=25,
        help="Max missing frames bridged within a single rally (default: 25)",
    )
    parser.add_argument("--min_hit_spacing", type=int, default=6, help="Min frames between consecutive hits (default: 6)")
    parser.add_argument("--out_json", type=str, default="", help="Output JSON report path (default: <out_dir>/rallies.json)")
    parser.add_argument("--out_csv", type=str, default="", help="Output CSV report path (default: <out_dir>/rallies.csv)")
    parser.add_argument("--cut_video", type=str, default="", help="Optional input MP4 to slice into rally_XX.mp4 clips")
    parser.add_argument("--out_dir", type=str, default="", help="Output directory for reports and rally video clips")
    parser.add_argument(
        "--no_highlights",
        action="store_true",
        help="Disable concatenating rally clips into highlights.mp4 when --cut_video is set",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    csv_path = args.ball_csv_flag or args.ball_csv
    if not csv_path:
        parser.error("A ball CSV path is required (positional or --ball_csv).")

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(csv_path)) or "."
    os.makedirs(out_dir, exist_ok=True)

    out_json = args.out_json or os.path.join(out_dir, "rallies.json")
    out_csv = args.out_csv or os.path.join(out_dir, "rallies.csv")

    detector = RallyDetector(
        fps=args.fps,
        min_rally_frames=args.min_rally_frames,
        max_missing_frames=args.max_missing_frames,
        min_hit_spacing=args.min_hit_spacing,
    )
    rallies = detector.detect_from_csv(csv_path)
    detector.export_json(rallies, out_json, source_csv=csv_path)
    detector.export_csv(rallies, out_csv)

    print(
        f"[RallyDetector] detected {len(rallies)} rallies "
        f"(total hits={sum(r['num_hits'] for r in rallies)}) -> {out_json}, {out_csv}"
    )
    for r in rallies:
        print(
            f"  Rally #{r['rally_id']:02d}: frames [{r['start_frame']}, {r['end_frame']}] "
            f"({r['start_sec']:.2f}s - {r['end_sec']:.2f}s, {r['duration_sec']:.2f}s) | "
            f"hits={r['num_hits']}"
        )

    if args.cut_video:
        clips = detector.cut_rallies(
            video_path=args.cut_video,
            rallies=rallies,
            out_dir=out_dir,
            concat_highlights=not args.no_highlights,
        )
        print(f"[RallyDetector] cut {len(clips)} rally clips into {out_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
