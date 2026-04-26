"""Render a single preview frame of the slimmed stats panel for visual check."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from overlay_player_analytics import MotionStats, draw_stats_panel  # type: ignore


def fake_stats(current, rally_dist, rally_max, total_dist):
    s = MotionStats()
    s.current_speed = current
    s.rally_distance = rally_dist
    s.rally_time = max(rally_dist / max(current, 0.5), 1.0)
    s.rally_max_speed = rally_max
    s.total_distance = total_dist
    s.total_time = s.rally_time * 3
    s.total_max_speed = max(rally_max, current)
    return s


def main():
    h, w = 1080, 1920
    frame = np.full((h, w, 3), 38, dtype=np.uint8)

    cv2.putText(
        frame,
        "stats panel preview (4 fields per player)",
        (440, 540),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (170, 170, 170),
        2,
        cv2.LINE_AA,
    )

    far = fake_stats(current=2.85, rally_dist=12.40, rally_max=4.62, total_dist=58.10)
    near = fake_stats(current=3.71, rally_dist=14.05, rally_max=5.18, total_dist=64.32)

    draw_stats_panel(frame, rally_idx=4, far_stats=far, near_stats=near, pipeline_fps=23.6)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "panel_preview.png")
    cv2.imwrite(out, frame)
    print(f"[OK] wrote {out}  size={w}x{h}")


if __name__ == "__main__":
    main()
