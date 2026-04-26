import argparse
import csv
import math
import os
import time
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


DEFAULT_VIDEO_PATH = r"D:\yumaoqiu\tracknet_v3_result\866ba79f9b46ce0d9b8b1d55eb82832c_tracknetv3.mp4"
DEFAULT_OUTPUT_PATH = r"D:\yumaoqiu\end1.mp4"


FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
_FONT_CACHE = {}

ZH_TITLE = "\u4e0a\u534a\u573a\u7403\u5458\u7edf\u8ba1"
ZH_RALLY = "\u56de\u5408"
ZH_TOP = "\u4e0a\u534a\u573a"
ZH_BOTTOM = "\u4e0b\u534a\u573a"
ZH_ERR_COURT = "\u672a\u63d0\u4f9b --court_points\uff0c\u4e14\u5df2\u5173\u95ed\u624b\u52a8\u9009\u70b9\u3002\u8bf7\u5f00\u542f --select_court_points \u6216\u4f20\u5165 --court_points\u3002"


def parse_court_points(value):
    if not value:
        return None
    tokens = [v.strip() for v in value.split(",") if v.strip() != ""]
    if len(tokens) != 8:
        raise ValueError("court_points must be x1,y1,x2,y2,x3,y3,x4,y4")
    vals = [float(v) for v in tokens]
    return np.array(
        [[vals[0], vals[1]], [vals[2], vals[3]], [vals[4], vals[5]], [vals[6], vals[7]]],
        dtype=np.float32,
    )


def infer_ball_csv(video_path):
    base = os.path.splitext(os.path.basename(video_path))[0]
    if base.endswith("_tracknetv3"):
        stem = base.replace("_tracknetv3", "")
    else:
        stem = base
    candidates = [
        os.path.join(os.path.dirname(video_path), f"{stem}_ball.csv"),
        os.path.join(os.path.dirname(video_path), f"{base}_ball.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def load_ball_dict(csv_path):
    ball = {}
    if not csv_path or not os.path.exists(csv_path):
        return ball
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(float(row.get("Frame", "0")))
                vis = int(float(row.get("Visibility", "0")))
                x = int(float(row.get("X", "0")))
                y = int(float(row.get("Y", "0")))
                ball[idx] = (vis, x, y)
            except Exception:
                continue
    return ball


def estimate_ball_center_shift(ball_dict, homography, court_quad, court_w_m, court_h_m):
    if len(ball_dict) == 0:
        return 0.0

    xs = []
    keys = sorted(ball_dict.keys())
    step = max(1, len(keys) // 2000)
    for i in keys[::step]:
        vis, bx, by = ball_dict[i]
        if vis <= 0:
            continue
        img_pt = clamp_point_to_quad((bx, by), court_quad)
        m = to_court_m(img_pt, homography)
        if m is None:
            continue
        x = float(np.clip(m[0], 0.0, court_w_m))
        y = float(np.clip(m[1], 0.0, court_h_m))
        if 0.05 <= y <= (court_h_m - 0.05):
            xs.append(x)

    if len(xs) < 50:
        return 0.0

    center = court_w_m * 0.5
    median_x = float(np.median(np.array(xs)))
    shift = center - median_x
    max_shift = court_w_m * 0.25
    return float(np.clip(shift, -max_shift, max_shift))


def default_court_points(w, h):
    # Calibrated for common badminton broadcast perspective.
    return np.array(
        [
            [0.3667 * w, 0.4265 * h],  # TL
            [0.6385 * w, 0.4265 * h],  # TR
            [0.7490 * w, 0.9650 * h],  # BR
            [0.2540 * w, 0.9650 * h],  # BL
        ],
        dtype=np.float32,
    )


def select_four_points(frame):
    show = frame.copy()
    base = frame.copy()
    points = []

    def on_mouse(event, x, y, flags, param):
        del flags, param
        nonlocal show, points
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.circle(show, (x, y), 7, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(
                show,
                str(len(points)),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    win_name = "Click court corners: TL -> TR -> BR -> BL"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1400, 800)
    cv2.setMouseCallback(win_name, on_mouse)

    while True:
        canvas = show.copy()
        tip = "Click TL -> TR -> BR -> BL | Enter=OK | r=Reset | q=Quit"
        cv2.putText(canvas, tip, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(win_name, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("r"):
            points = []
            show = base.copy()
        elif key == 13:
            if len(points) == 4:
                break
        elif key == ord("q") or key == 27:
            cv2.destroyAllWindows()
            raise RuntimeError("Canceled selecting court points.")

    cv2.destroyAllWindows()
    return np.array(points, dtype=np.float32)


def point_in_quad(point, quad):
    quad_int = quad.astype(np.int32)
    return cv2.pointPolygonTest(quad_int, (float(point[0]), float(point[1])), False) >= 0


def build_player_filter_quad(court_quad, top_inner_ratio=0.0):
    quad = court_quad.astype(np.float32).copy()
    if top_inner_ratio <= 0:
        return quad
    ratio = float(np.clip(top_inner_ratio, 0.0, 0.25))
    # Pull top edge inward to suppress audience/judges outside the far baseline.
    quad[0] = quad[0] + (quad[3] - quad[0]) * ratio
    quad[1] = quad[1] + (quad[2] - quad[1]) * ratio
    return quad


def build_detect_roi(court_quad, frame_w, frame_h, pad_x=28, pad_top=84, pad_bottom=18):
    xs = court_quad[:, 0]
    ys = court_quad[:, 1]
    x1 = int(max(0, math.floor(float(np.min(xs)) - pad_x)))
    x2 = int(min(frame_w, math.ceil(float(np.max(xs)) + pad_x)))
    y1 = int(max(0, math.floor(float(np.min(ys)) - pad_top)))
    y2 = int(min(frame_h, math.ceil(float(np.max(ys)) + pad_bottom)))
    if x2 <= x1:
        x1, x2 = 0, frame_w
    if y2 <= y1:
        y1, y2 = 0, frame_h
    return x1, y1, x2, y2


def estimate_foot_point(box_xyxy, keypoints=None, confs=None, ankle_min_conf=0.22, fallback_lift_ratio=0.0):
    x1, y1, x2, y2 = box_xyxy
    bh = max(1.0, float(y2 - y1))
    lift = float(np.clip(fallback_lift_ratio, 0.0, 0.28)) * bh
    fallback = (int((x1 + x2) / 2.0), int(y2 - lift))

    if keypoints is None or len(keypoints) < 17:
        return fallback

    ankle_pts = []
    for idx in (15, 16):
        if idx >= len(keypoints):
            continue
        if confs is not None and idx < len(confs) and float(confs[idx]) < ankle_min_conf:
            continue
        ankle_pts.append((float(keypoints[idx][0]), float(keypoints[idx][1])))

    if len(ankle_pts) == 0:
        return fallback
    if len(ankle_pts) == 1:
        ax, ay = ankle_pts[0]
    else:
        ax = 0.5 * (ankle_pts[0][0] + ankle_pts[1][0])
        ay = max(ankle_pts[0][1], ankle_pts[1][1])

    return (int(np.clip(ax, x1, x2)), int(np.clip(ay, y1, y2)))


def closest_point_on_segment(point, seg_a, seg_b):
    px, py = float(point[0]), float(point[1])
    ax, ay = float(seg_a[0]), float(seg_a[1])
    bx, by = float(seg_b[0]), float(seg_b[1])
    abx, aby = bx - ax, by - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-9:
        return (ax, ay)
    t = ((px - ax) * abx + (py - ay) * aby) / denom
    t = float(np.clip(t, 0.0, 1.0))
    return (ax + t * abx, ay + t * aby)


def clamp_point_to_quad(point, quad):
    if point_in_quad(point, quad):
        return (float(point[0]), float(point[1]))
    best = None
    best_d = float("inf")
    for i in range(4):
        a = quad[i]
        b = quad[(i + 1) % 4]
        q = closest_point_on_segment(point, a, b)
        d = math.hypot(float(point[0]) - q[0], float(point[1]) - q[1])
        if d < best_d:
            best_d = d
            best = q
    return best if best is not None else (float(point[0]), float(point[1]))


def euclidean(p1, p2):
    return math.hypot(float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1]))


def find_closest_candidate(candidates, ref_point, used_idx=None):
    if ref_point is None or len(candidates) == 0:
        return None

    best_i = None
    best_d = float("inf")
    for i, c in enumerate(candidates):
        if used_idx is not None and i in used_idx:
            continue
        d = euclidean(c["foot"], ref_point)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def update_track_memory(memory_dict, track_id, foot_xy, max_len=90):
    hist = memory_dict.get(track_id)
    if hist is None:
        hist = deque(maxlen=max_len)
        memory_dict[track_id] = hist
    hist.append((float(foot_xy[0]), float(foot_xy[1])))
    return hist


def is_static_top_track(history, top_y, min_frames=42, top_zone_px=18, top_ratio=0.88, static_move_px=30.0):
    if history is None or len(history) < min_frames:
        return False
    ys = [h[1] for h in history]
    top_hits = sum(1 for y in ys if y <= (top_y + top_zone_px))
    if (top_hits / float(len(history))) < top_ratio:
        return False

    xs = [h[0] for h in history]
    spread = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    return spread <= static_move_px


def assign_players(
    candidates,
    prev_near_pt,
    prev_far_pt,
    prev_near_id=None,
    prev_far_id=None,
    far_lock_max_jump=110.0,
    far_max_up_jump=30.0,
    side_line_y=None,
    side_band_px=12.0,
):
    if len(candidates) == 0:
        return None, None

    candidates_sorted = sorted(candidates, key=lambda x: x["foot"][1])
    has_side_line = side_line_y is not None

    def pick_with_history(pool, prev_id, prev_pt, prefer_max_y):
        if len(pool) == 0:
            return None
        if prev_id is not None:
            by_id = next((c for c in pool if c["track_id"] == prev_id), None)
            if by_id is not None:
                return by_id
        if prev_pt is not None:
            idx = find_closest_candidate(pool, prev_pt)
            if idx is not None:
                return pool[idx]
        if prefer_max_y:
            return max(pool, key=lambda c: c["foot"][1])
        return min(pool, key=lambda c: c["foot"][1])

    if has_side_line:
        near_pool = [c for c in candidates_sorted if c["foot"][1] >= (side_line_y - side_band_px)]
        far_pool = [c for c in candidates_sorted if c["foot"][1] <= (side_line_y + side_band_px)]

        near_candidate = pick_with_history(near_pool, prev_near_id, prev_near_pt, prefer_max_y=True)
        far_candidate = pick_with_history(far_pool, prev_far_id, prev_far_pt, prefer_max_y=False)

        if near_candidate is not None and far_candidate is not None and near_candidate["track_id"] == far_candidate["track_id"]:
            # Prefer continuity on whichever side has stronger historical binding.
            keep_far = (prev_far_id is not None and far_candidate["track_id"] == prev_far_id)
            keep_near = (prev_near_id is not None and near_candidate["track_id"] == prev_near_id)
            if keep_far and not keep_near:
                remain = [c for c in near_pool if c["track_id"] != far_candidate["track_id"]]
                near_candidate = max(remain, key=lambda c: c["foot"][1]) if len(remain) > 0 else None
            elif keep_near and not keep_far:
                remain = [c for c in far_pool if c["track_id"] != near_candidate["track_id"]]
                far_candidate = min(remain, key=lambda c: c["foot"][1]) if len(remain) > 0 else None
            else:
                d_near = euclidean(near_candidate["foot"], prev_near_pt) if prev_near_pt is not None else float("inf")
                d_far = euclidean(far_candidate["foot"], prev_far_pt) if prev_far_pt is not None else float("inf")
                if d_far <= d_near:
                    remain = [c for c in near_pool if c["track_id"] != far_candidate["track_id"]]
                    near_candidate = max(remain, key=lambda c: c["foot"][1]) if len(remain) > 0 else None
                else:
                    remain = [c for c in far_pool if c["track_id"] != near_candidate["track_id"]]
                    far_candidate = min(remain, key=lambda c: c["foot"][1]) if len(remain) > 0 else None

        if far_candidate is not None and prev_far_pt is not None and (prev_far_id is None or far_candidate["track_id"] != prev_far_id):
            far_jump = euclidean(far_candidate["foot"], prev_far_pt)
            jumped_up_too_much = far_candidate["foot"][1] < (prev_far_pt[1] - far_max_up_jump)
            if jumped_up_too_much or far_jump > far_lock_max_jump:
                far_candidate = None

        return near_candidate, far_candidate

    if len(candidates_sorted) >= 2:
        near_candidate = candidates_sorted[-1]
        far_candidate = candidates_sorted[0]

        if prev_near_pt is not None and prev_far_pt is not None:
            best = None
            for i, near in enumerate(candidates_sorted):
                for j, far in enumerate(candidates_sorted):
                    if i == j:
                        continue
                    ny, fy = near["foot"][1], far["foot"][1]
                    sep = ny - fy
                    penalty = 0.0
                    if sep <= 0:
                        penalty += 1000.0
                    elif sep < 18:
                        penalty += (18 - sep) * 10.0
                    if prev_near_id is not None and near["track_id"] != prev_near_id:
                        penalty += 15.0
                    if prev_far_id is not None and far["track_id"] != prev_far_id:
                        penalty += 15.0
                    cost = euclidean(near["foot"], prev_near_pt) + euclidean(far["foot"], prev_far_pt) + penalty
                    if best is None or cost < best[0]:
                        best = (cost, near, far)
            if best is not None:
                near_candidate, far_candidate = best[1], best[2]

        ordered = sorted([near_candidate, far_candidate], key=lambda x: x["foot"][1])
        return ordered[-1], ordered[0]

    one = candidates_sorted[0]
    if prev_near_pt is None and prev_far_pt is None:
        return one, None
    if prev_near_pt is not None and prev_far_pt is not None:
        mid_y = 0.5 * (prev_near_pt[1] + prev_far_pt[1])
        if one["foot"][1] >= mid_y:
            return one, None
        return None, one
    if prev_near_pt is not None:
        return one, None
    return None, one


def catmull_rom_spline(points, steps=12):
    if len(points) < 2:
        return points[:]

    pts = [points[0]] + points[:] + [points[-1]]
    curve = []

    for i in range(1, len(pts) - 2):
        p0 = np.array(pts[i - 1], dtype=np.float32)
        p1 = np.array(pts[i], dtype=np.float32)
        p2 = np.array(pts[i + 1], dtype=np.float32)
        p3 = np.array(pts[i + 2], dtype=np.float32)

        for t in np.linspace(0, 1, steps, endpoint=False):
            t2 = t * t
            t3 = t2 * t
            p = 0.5 * (
                (2 * p1)
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
            )
            curve.append((int(p[0]), int(p[1])))

    curve.append(points[-1])
    return curve


def split_valid_segments(trail):
    segments = []
    current = []
    for p in trail:
        if p is None:
            if len(current) >= 2:
                segments.append(current)
            current = []
        else:
            current.append(p)
    if len(current) >= 2:
        segments.append(current)
    return segments


def draw_dashed_polyline(frame, points, color, thickness=3, dash_len=12, gap_len=10):
    if len(points) < 2:
        return

    seg_lens = [0.0]
    total = 0.0
    for i in range(1, len(points)):
        total += euclidean(points[i - 1], points[i])
        seg_lens.append(total)

    if total < 1.0:
        return

    cycle = dash_len + gap_len
    draw_from = 0.0

    while draw_from < total:
        draw_to = min(draw_from + dash_len, total)
        dash_points = []

        for i in range(1, len(points)):
            l1, l2 = seg_lens[i - 1], seg_lens[i]
            p1, p2 = points[i - 1], points[i]

            if l1 <= draw_from <= l2 and l2 > l1:
                r = (draw_from - l1) / (l2 - l1)
                dash_points.append((int(p1[0] + (p2[0] - p1[0]) * r), int(p1[1] + (p2[1] - p1[1]) * r)))

            if draw_from <= l2 <= draw_to:
                dash_points.append(p2)

            if l1 <= draw_to <= l2 and l2 > l1:
                r = (draw_to - l1) / (l2 - l1)
                dash_points.append((int(p1[0] + (p2[0] - p1[0]) * r), int(p1[1] + (p2[1] - p1[1]) * r)))
                break

        if len(dash_points) >= 2:
            cv2.polylines(frame, [np.array(dash_points, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)

        draw_from += cycle


def draw_trail(frame, trail, color, thickness=3, dash_len=12, gap_len=10, smooth_steps=12):
    for seg in split_valid_segments(list(trail)):
        if len(seg) < 2:
            continue
        smooth_seg = catmull_rom_spline(seg, steps=smooth_steps)
        draw_dashed_polyline(frame, smooth_seg, color, thickness, dash_len, gap_len)


def draw_alpha_rect(frame, top_left, bottom_right, color, alpha):
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def get_zh_font(size):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            _FONT_CACHE[size] = ImageFont.truetype(path, size=size)
            return _FONT_CACHE[size]
    _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def draw_text_cn(frame, text_items):
    if len(text_items) == 0:
        return

    h, w = frame.shape[:2]
    x1, y1 = w, h
    x2, y2 = 0, 0
    font_items = []
    for text, xy, size, bgr in text_items:
        font = get_zh_font(size)
        bbox = font.getbbox(text)
        tw = max(1, int(bbox[2] - bbox[0]))
        th = max(1, int(bbox[3] - bbox[1]))
        x, y = int(xy[0]), int(xy[1])
        x1 = min(x1, x)
        y1 = min(y1, y)
        x2 = max(x2, x + tw + 2)
        y2 = max(y2, y + th + 2)
        font_items.append((text, x, y, font, bgr))

    if x2 <= x1 or y2 <= y1:
        return

    pad = 6
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    roi = frame[y1:y2, x1:x2]
    pil_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    for text, x, y, font, bgr in font_items:
        rgb = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
        draw.text((x - x1, y - y1), text, font=font, fill=rgb)
    roi[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


class MotionStats:
    def __init__(self):
        self.prev_m = None
        self.prev_frame_idx = None
        self.current_speed = 0.0
        self.current_speed_buffer = deque(maxlen=6)

        self.rally_distance = 0.0
        self.rally_time = 0.0
        self.rally_max_speed = 0.0

        self.total_distance = 0.0
        self.total_time = 0.0
        self.total_max_speed = 0.0

    @property
    def rally_avg_speed(self):
        return 0.0 if self.rally_time <= 1e-6 else self.rally_distance / self.rally_time

    @property
    def total_avg_speed(self):
        return 0.0 if self.total_time <= 1e-6 else self.total_distance / self.total_time

    def reset_rally(self):
        self.rally_distance = 0.0
        self.rally_time = 0.0
        self.rally_max_speed = 0.0

    def update(self, pt_m, frame_idx, fps):
        if pt_m is None:
            self.current_speed_buffer.append(0.0)
            self.current_speed = float(np.mean(self.current_speed_buffer))
            return

        inst_speed = 0.0
        if self.prev_m is not None and self.prev_frame_idx is not None and frame_idx > self.prev_frame_idx:
            dt = (frame_idx - self.prev_frame_idx) / max(fps, 1e-6)
            distance = euclidean(pt_m, self.prev_m)
            # Reject unrealistic jump caused by tracker ID switches.
            # Badminton players sprint at ~7 m/s; cap a touch above that.
            max_jump_m = 8.0 * dt + 0.05
            if distance <= max_jump_m:
                inst_speed = distance / max(dt, 1e-6)
                self.rally_distance += distance
                self.rally_time += dt
                self.rally_max_speed = max(self.rally_max_speed, inst_speed)
                self.total_distance += distance
                self.total_time += dt
                self.total_max_speed = max(self.total_max_speed, inst_speed)

        self.current_speed_buffer.append(inst_speed)
        self.current_speed = float(np.mean(self.current_speed_buffer))
        self.prev_m = pt_m
        self.prev_frame_idx = frame_idx


def to_court_m(point_xy, homography):
    if point_xy is None:
        return None
    src = np.array([[[float(point_xy[0]), float(point_xy[1])]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, homography)[0][0]
    return (float(dst[0]), float(dst[1]))


def to_map_px(point_m, map_x, map_y, map_w, map_h, court_w_m, court_h_m):
    if point_m is None:
        return None
    x_norm = float(np.clip(point_m[0] / max(court_w_m, 1e-6), 0.0, 1.0))
    y_norm = float(np.clip(point_m[1] / max(court_h_m, 1e-6), 0.0, 1.0))
    return (int(map_x + x_norm * map_w), int(map_y + y_norm * map_h))


def enforce_half_court(point_m, side, court_w_m, court_h_m, far_half_expand=1.0):
    if point_m is None:
        return None
    x, y = float(point_m[0]), float(point_m[1])
    x = float(np.clip(x, 0.0, court_w_m))
    y = float(np.clip(y, 0.0, court_h_m))
    mid = court_h_m * 0.5
    if side == "far":
        # Fold lower-half leakage back to upper half to keep far player in top court,
        # while preserving vertical motion instead of flattening to the midline.
        y = float(mid - abs(mid - y))
        # Expand far-side activity range on mini court so it matches near-side visual span.
        scale = max(1.0, float(far_half_expand))
        y = float(np.clip(mid - (mid - y) * scale, 0.0, mid - 1e-3))
    if side == "near" and y < mid:
        y = float(mid + 1e-3)
    return (x, y)


def estimate_point_by_optical_flow(prev_gray, cur_gray, prev_pt, max_move=40.0):
    if prev_gray is None or cur_gray is None or prev_pt is None:
        return None

    p0 = np.array([[[float(prev_pt[0]), float(prev_pt[1])]]], dtype=np.float32)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        cur_gray,
        p0,
        None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )
    if p1 is None or st is None or st[0][0] != 1:
        return None

    nx, ny = float(p1[0][0][0]), float(p1[0][0][1])
    if math.hypot(nx - prev_pt[0], ny - prev_pt[1]) > max_move:
        return None
    return (int(nx), int(ny))


def draw_stats_panel(frame, rally_idx, far_stats, near_stats, pipeline_fps=0.0):
    h, w = frame.shape[:2]
    x0, y0 = 12, 12
    panel_w = max(340, int(w * 0.33))

    min_line_h, max_line_h = 18, 24
    header_h = 96
    block_gap = 18
    line_h = int(np.clip((h - 80) / 22.0, min_line_h, max_line_h))
    block_h = 28 + 4 * line_h
    panel_h = header_h + block_h * 2 + block_gap + 18
    panel_h = int(np.clip(panel_h, 310, h - 18))
    draw_alpha_rect(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), 0.54)

    block1_y = y0 + header_h
    block2_y = block1_y + block_h + block_gap

    text_items = [
        (ZH_TITLE, (x0 + 12, y0 + 6), 30, (0, 210, 255)),
        (f"{ZH_RALLY}: {rally_idx}", (x0 + 12, y0 + 38), 28, (0, 175, 255)),
        (f"FPS: {pipeline_fps:.2f}", (x0 + 12, y0 + 64), 20, (190, 235, 255)),
    ]

    def add_block(y, title, stats, color):
        text_items.append((title, (x0 + 12, y), 26, color))
        rows = [
            f"\u5f53\u524d\u901f\u5ea6: {stats.current_speed:.2f} m/s",
            f"\u56de\u5408\u8ddd\u79bb: {stats.rally_distance:.2f} m",
            f"\u56de\u5408\u6700\u9ad8: {stats.rally_max_speed:.2f} m/s",
            f"\u603b\u8ddd\u79bb: {stats.total_distance:.2f} m",
        ]
        for i, row in enumerate(rows):
            text_items.append((row, (x0 + 12, y + 28 + i * line_h), 22, (240, 240, 240)))

    add_block(block1_y, ZH_TOP, far_stats, (0, 175, 255))
    add_block(block2_y, ZH_BOTTOM, near_stats, (255, 120, 220))
    draw_text_cn(frame, text_items)


def draw_mini_court(frame, far_m, near_m, ball_m, far_map_hist, near_map_hist, ball_map_hist, court_w_m, court_h_m):
    h, w = frame.shape[:2]
    panel_w = max(200, int(w * 0.18))
    panel_h = max(320, int(h * 0.62))
    px0, py0 = w - panel_w - 12, 12
    draw_alpha_rect(frame, (px0, py0), (px0 + panel_w, py0 + panel_h), (10, 16, 28), 0.64)
    cv2.putText(frame, "Mini Court", (px0 + 12, py0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2, cv2.LINE_AA)

    pad = 14
    cx0, cy0 = px0 + pad, py0 + 42
    cw, ch = panel_w - pad * 2, panel_h - 52
    cv2.rectangle(frame, (cx0, cy0), (cx0 + cw, cy0 + ch), (70, 90, 130), 1, cv2.LINE_AA)

    # Main lines
    cv2.line(frame, (cx0 + cw // 2, cy0), (cx0 + cw // 2, cy0 + ch), (70, 90, 130), 1, cv2.LINE_AA)
    cv2.line(frame, (cx0, cy0 + ch // 2), (cx0 + cw, cy0 + ch // 2), (70, 90, 130), 1, cv2.LINE_AA)

    # Service lines (approx badminton doubles court markings)
    short_service = 1.98
    long_service = 0.76
    top_short_y = int(cy0 + (short_service / court_h_m) * ch)
    bot_short_y = int(cy0 + ((court_h_m - short_service) / court_h_m) * ch)
    top_long_y = int(cy0 + (long_service / court_h_m) * ch)
    bot_long_y = int(cy0 + ((court_h_m - long_service) / court_h_m) * ch)
    cv2.line(frame, (cx0, top_short_y), (cx0 + cw, top_short_y), (55, 75, 115), 1, cv2.LINE_AA)
    cv2.line(frame, (cx0, bot_short_y), (cx0 + cw, bot_short_y), (55, 75, 115), 1, cv2.LINE_AA)
    cv2.line(frame, (cx0, top_long_y), (cx0 + cw, top_long_y), (45, 65, 105), 1, cv2.LINE_AA)
    cv2.line(frame, (cx0, bot_long_y), (cx0 + cw, bot_long_y), (45, 65, 105), 1, cv2.LINE_AA)

    far_map = to_map_px(far_m, cx0, cy0, cw, ch, court_w_m, court_h_m)
    near_map = to_map_px(near_m, cx0, cy0, cw, ch, court_w_m, court_h_m)
    ball_map = to_map_px(ball_m, cx0, cy0, cw, ch, court_w_m, court_h_m)

    if far_map is not None:
        far_map_hist.append(far_map)
    if near_map is not None:
        near_map_hist.append(near_map)
    if ball_map is not None:
        ball_map_hist.append(ball_map)

    for p in far_map_hist:
        cv2.circle(frame, p, 2, (0, 180, 255), -1, cv2.LINE_AA)
    for p in near_map_hist:
        cv2.circle(frame, p, 2, (255, 110, 210), -1, cv2.LINE_AA)
    for p in ball_map_hist:
        cv2.circle(frame, p, 2, (0, 235, 255), -1, cv2.LINE_AA)

    if far_map is not None:
        cv2.circle(frame, far_map, 5, (0, 210, 255), -1, cv2.LINE_AA)
    if near_map is not None:
        cv2.circle(frame, near_map, 5, (255, 140, 220), -1, cv2.LINE_AA)
    if ball_map is not None:
        cv2.circle(frame, ball_map, 4, (0, 235, 255), -1, cv2.LINE_AA)


def draw_skeleton(frame, keypoints, confs, color, min_conf=0.3):
    if keypoints is None:
        return

    # COCO-17 topology used by YOLO pose models.
    edges = [
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (5, 6), (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
    ]

    def visible(i):
        if keypoints is None or i >= len(keypoints):
            return False
        if confs is None or i >= len(confs):
            return True
        return float(confs[i]) >= min_conf

    for i, j in edges:
        if visible(i) and visible(j):
            p1 = (int(keypoints[i][0]), int(keypoints[i][1]))
            p2 = (int(keypoints[j][0]), int(keypoints[j][1]))
            cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)

    for i in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
        if visible(i):
            p = (int(keypoints[i][0]), int(keypoints[i][1]))
            cv2.circle(frame, p, 3, color, -1, cv2.LINE_AA)


def shift_player_for_draw(player, target_foot):
    if player is None or target_foot is None:
        return player
    src_foot = player.get("foot")
    if src_foot is None:
        return player
    dx = int(target_foot[0] - src_foot[0])
    dy = int(target_foot[1] - src_foot[1])
    if dx == 0 and dy == 0:
        return player

    out = dict(player)
    x1, y1, x2, y2 = out["box"]
    out["box"] = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
    out["foot"] = (int(target_foot[0]), int(target_foot[1]))
    if "kpts" in out and out["kpts"] is not None:
        out["kpts"] = out["kpts"] + np.array([dx, dy], dtype=np.float32)
    return out


def draw_player(frame, player, label, color, draw_foot_point=True, draw_id=False, draw_pose=True):
    x1, y1, x2, y2 = map(int, player["box"])
    fx, fy = player["foot"]
    tid = player["track_id"]
    keypoints = player.get("kpts")
    kconf = player.get("kconf")

    if draw_pose and keypoints is not None:
        draw_skeleton(frame, keypoints, kconf, color)

    # Keep labels lightweight for long videos.
    if draw_id:
        text_x = max(8, x1)
        text_y = max(26, y1 - 8)
        cv2.putText(frame, f"id:{tid}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    if draw_foot_point:
        cv2.circle(frame, (fx, fy), 5, color, -1, cv2.LINE_AA)


def run(args):
    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video not found: {args.video_path}")

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        fps = 30

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Cannot read first frame.")

    court_quad = parse_court_points(args.court_points)
    if court_quad is None:
        if args.select_court_points:
            court_quad = select_four_points(first_frame)
        else:
            raise ValueError(ZH_ERR_COURT)
    player_filter_quad = build_player_filter_quad(court_quad, args.top_inner_ratio)
    top_y = float(0.5 * (court_quad[0][1] + court_quad[1][1]))
    bot_y = float(0.5 * (court_quad[2][1] + court_quad[3][1]))
    midline_y_img = 0.5 * (top_y + bot_y)

    if args.detect_on_court_roi:
        detect_x1, detect_y1, detect_x2, detect_y2 = build_detect_roi(
            court_quad,
            w,
            h,
            pad_x=args.detect_roi_pad_x,
            pad_top=args.detect_roi_pad_top,
            pad_bottom=args.detect_roi_pad_bottom,
        )
    else:
        detect_x1, detect_y1, detect_x2, detect_y2 = 0, 0, w, h
    detect_area_ratio = ((detect_x2 - detect_x1) * (detect_y2 - detect_y1)) / max(1.0, float(w * h))

    dst = np.array(
        [[0.0, 0.0], [args.court_width_m, 0.0], [args.court_width_m, args.court_length_m], [0.0, args.court_length_m]],
        dtype=np.float32,
    )
    homography, _ = cv2.findHomography(court_quad, dst)
    if homography is None:
        cap.release()
        raise RuntimeError("Failed to compute court homography.")

    cap.release()
    cap = cv2.VideoCapture(args.video_path)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    writer = cv2.VideoWriter(args.output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open output writer: {args.output_path}")

    model = YOLO(args.yolo_model)

    near_trail = deque(maxlen=args.trail_len)
    far_trail = deque(maxlen=args.trail_len)
    near_map_hist = deque(maxlen=args.minimap_trail_len)
    far_map_hist = deque(maxlen=args.minimap_trail_len)
    ball_map_hist = deque(maxlen=args.minimap_trail_len)

    if args.draw_ball_on_minimap:
        ball_csv = args.ball_csv if args.ball_csv else infer_ball_csv(args.video_path)
        ball_dict = load_ball_dict(ball_csv)
    else:
        ball_csv = ""
        ball_dict = {}
    ball_x_shift = estimate_ball_center_shift(ball_dict, homography, court_quad, args.court_width_m, args.court_length_m)

    near_stats = MotionStats()
    far_stats = MotionStats()
    rally_idx = 1
    both_missing = 0
    prev_near_pt = None
    prev_far_pt = None
    prev_near_id = None
    prev_far_id = None
    near_missing_count = 0
    far_missing_count = 0
    in_rally = False
    prev_gray = None
    near_player_cache = None
    far_player_cache = None
    near_cache_age = 0
    far_cache_age = 0
    track_memory = {}
    offcourt_track_ids = set()

    frame_idx = 0
    detect_calls = 0
    t_start = time.perf_counter()
    print(f"[INFO] video={args.video_path}")
    print(f"[INFO] output={args.output_path}")
    print(f"[INFO] size={w}x{h}, fps={fps:.2f}, frames={total_frames}")
    print(f"[INFO] court_points={court_quad.astype(int).tolist()}")
    print(
        f"[INFO] detect_interval={max(1,args.detect_interval)}, "
        f"half={args.half}, "
        f"detect_roi={args.detect_on_court_roi}, "
        f"detect_area_ratio={detect_area_ratio:.3f}, "
        f"top_inner_ratio={args.top_inner_ratio:.3f}, "
        f"side_band_px={args.side_band_px:.1f}, "
        f"far_half_expand={args.far_half_expand:.2f}, "
        f"draw_pose_on_skipped={args.draw_pose_on_skipped}"
    )
    if args.draw_ball_on_minimap and ball_csv:
        print(f"[INFO] ball_csv={ball_csv}")
        print(f"[INFO] ball_center_shift_x={ball_x_shift:.3f}m")
    else:
        print("[INFO] skip ball trail on mini court")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        do_detect = (frame_idx == 0) or (frame_idx % max(1, args.detect_interval) == 0)
        results = []
        roi_ox, roi_oy = 0, 0
        if do_detect:
            detect_frame = frame[detect_y1:detect_y2, detect_x1:detect_x2]
            if detect_frame.size == 0:
                detect_frame = frame
            else:
                roi_ox, roi_oy = detect_x1, detect_y1
            results = model.track(
                source=detect_frame,
                persist=True,
                tracker=args.tracker_cfg,
                classes=[0],
                conf=args.conf_thres,
                verbose=False,
                imgsz=args.imgsz,
                device=args.device if args.device else None,
                half=args.half,
            )
            detect_calls += 1

        candidates = []
        if len(results) > 0:
            r = results[0]
            if r.boxes is not None and r.boxes.id is not None and len(r.boxes.id) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                ids = r.boxes.id.cpu().numpy().astype(int)
                kpts_xy = None
                kpts_conf = None
                if args.draw_pose and r.keypoints is not None and r.keypoints.xy is not None:
                    kpts_xy = r.keypoints.xy.cpu().numpy()
                    if r.keypoints.conf is not None:
                        kpts_conf = r.keypoints.conf.cpu().numpy()

                for idx, (box, tid) in enumerate(zip(boxes, ids)):
                    x1, y1, x2, y2 = box
                    x1 += roi_ox
                    y1 += roi_oy
                    x2 += roi_ox
                    y2 += roi_oy
                    bw = x2 - x1
                    bh = y2 - y1
                    if bw < args.min_box_w or bh < args.min_box_h:
                        continue

                    tid_i = int(tid)
                    if tid_i in offcourt_track_ids:
                        continue

                    person_kpts = None
                    person_kconf = None
                    if kpts_xy is not None and idx < len(kpts_xy):
                        person_kpts = kpts_xy[idx].copy()
                        person_kpts[:, 0] += roi_ox
                        person_kpts[:, 1] += roi_oy
                        if kpts_conf is not None and idx < len(kpts_conf):
                            person_kconf = kpts_conf[idx]

                    is_far_zone_hint = y2 <= (midline_y_img + args.far_zone_hint_margin)
                    foot_x, foot_y = estimate_foot_point(
                        (x1, y1, x2, y2),
                        keypoints=person_kpts,
                        confs=person_kconf,
                        ankle_min_conf=args.ankle_min_conf,
                        fallback_lift_ratio=(args.far_fallback_lift_ratio if is_far_zone_hint else 0.0),
                    )

                    hist = update_track_memory(track_memory, tid_i, (foot_x, foot_y), max_len=args.judge_hist_len)
                    if is_static_top_track(
                        hist,
                        top_y,
                        min_frames=args.judge_min_frames,
                        top_zone_px=args.judge_top_zone_px,
                        top_ratio=args.judge_top_ratio,
                        static_move_px=args.judge_static_move_px,
                    ):
                        offcourt_track_ids.add(tid_i)
                        continue

                    if not point_in_quad((foot_x, foot_y), court_quad):
                        continue
                    if not point_in_quad((foot_x, foot_y), player_filter_quad):
                        if prev_far_pt is None or euclidean((foot_x, foot_y), prev_far_pt) > args.top_inner_keep_dist:
                            continue

                    cand = {
                        "track_id": tid_i,
                        "box": (float(x1), float(y1), float(x2), float(y2)),
                        "foot": (foot_x, foot_y),
                    }
                    if person_kpts is not None:
                        cand["kpts"] = person_kpts
                        if person_kconf is not None:
                            cand["kconf"] = person_kconf
                    candidates.append(cand)

        near_player, far_player = assign_players(
            candidates,
            prev_near_pt,
            prev_far_pt,
            prev_near_id,
            prev_far_id,
            far_lock_max_jump=args.far_lock_max_jump,
            far_max_up_jump=args.far_max_up_jump,
            side_line_y=midline_y_img,
            side_band_px=args.side_band_px,
        )
        near_draw_pt = None
        far_draw_pt = None
        missing_step = 1 if do_detect else 0

        if near_player is None and far_player is None:
            both_missing += missing_step
            near_missing_count += missing_step
            far_missing_count += missing_step

            if prev_near_pt is not None and near_missing_count <= args.hold_missing_frames:
                flow_near = estimate_point_by_optical_flow(prev_gray, cur_gray, prev_near_pt, args.max_flow_move)
                if flow_near is not None and point_in_quad(flow_near, court_quad):
                    prev_near_pt = flow_near
                near_draw_pt = prev_near_pt
            elif near_missing_count > args.reset_missing_frames:
                prev_near_pt = None
                prev_near_id = None

            if prev_far_pt is not None and far_missing_count <= args.hold_missing_frames:
                flow_far = estimate_point_by_optical_flow(prev_gray, cur_gray, prev_far_pt, args.max_flow_move)
                if flow_far is not None and point_in_quad(flow_far, court_quad):
                    prev_far_pt = flow_far
                far_draw_pt = prev_far_pt
            elif far_missing_count > args.reset_missing_frames:
                prev_far_pt = None
                prev_far_id = None

            near_trail.append(near_draw_pt)
            far_trail.append(far_draw_pt)
            near_stats.update(to_court_m(near_draw_pt, homography) if near_draw_pt is not None else None, frame_idx, fps)
            far_stats.update(to_court_m(far_draw_pt, homography) if far_draw_pt is not None else None, frame_idx, fps)
            if both_missing >= args.new_rally_missing_frames:
                in_rally = False
        else:
            if not in_rally:
                if frame_idx > 0:
                    rally_idx += 1
                    near_stats.reset_rally()
                    far_stats.reset_rally()
                in_rally = True
            both_missing = 0

            if near_player is not None:
                prev_near_pt = near_player["foot"]
                prev_near_id = near_player["track_id"]
                near_missing_count = 0
                near_draw_pt = near_player["foot"]
            else:
                near_missing_count += missing_step
                if prev_near_pt is not None and near_missing_count <= args.hold_missing_frames:
                    flow_near = estimate_point_by_optical_flow(prev_gray, cur_gray, prev_near_pt, args.max_flow_move)
                    if flow_near is not None and point_in_quad(flow_near, court_quad):
                        prev_near_pt = flow_near
                    near_draw_pt = prev_near_pt
                elif near_missing_count > args.reset_missing_frames:
                    prev_near_pt = None
                    prev_near_id = None

            if far_player is not None:
                prev_far_pt = far_player["foot"]
                prev_far_id = far_player["track_id"]
                far_missing_count = 0
                far_draw_pt = far_player["foot"]
            else:
                far_missing_count += missing_step
                if prev_far_pt is not None and far_missing_count <= args.hold_missing_frames:
                    flow_far = estimate_point_by_optical_flow(prev_gray, cur_gray, prev_far_pt, args.max_flow_move)
                    if flow_far is not None and point_in_quad(flow_far, court_quad):
                        prev_far_pt = flow_far
                    far_draw_pt = prev_far_pt
                elif far_missing_count > args.reset_missing_frames:
                    prev_far_pt = None
                    prev_far_id = None

            near_trail.append(near_draw_pt)
            far_trail.append(far_draw_pt)
            near_stats.update(to_court_m(near_draw_pt, homography) if near_draw_pt is not None else None, frame_idx, fps)
            far_stats.update(to_court_m(far_draw_pt, homography) if far_draw_pt is not None else None, frame_idx, fps)

        near_player_draw = near_player
        far_player_draw = far_player

        if near_player is not None:
            near_player_cache = near_player
            near_cache_age = 0
        else:
            near_cache_age += 1
            if (
                (not do_detect)
                and args.draw_pose_on_skipped
                and near_player_cache is not None
                and near_cache_age <= args.skip_pose_max_age
            ):
                if near_draw_pt is None:
                    near_draw_pt = prev_near_pt if prev_near_pt is not None else near_player_cache["foot"]
                near_player_draw = shift_player_for_draw(near_player_cache, near_draw_pt)

        if far_player is not None:
            far_player_cache = far_player
            far_cache_age = 0
        else:
            far_cache_age += 1
            if (
                (not do_detect)
                and args.draw_pose_on_skipped
                and far_player_cache is not None
                and far_cache_age <= args.skip_pose_max_age
            ):
                if far_draw_pt is None:
                    far_draw_pt = prev_far_pt if prev_far_pt is not None else far_player_cache["foot"]
                far_player_draw = shift_player_for_draw(far_player_cache, far_draw_pt)

        draw_trail(frame, near_trail, args.near_color, args.trail_thickness, args.dash_len, args.gap_len, args.smooth_steps)
        draw_trail(frame, far_trail, args.far_color, args.trail_thickness, args.dash_len, args.gap_len, args.smooth_steps)

        ball_m_for_map = None
        if frame_idx in ball_dict:
            vis, bx, by = ball_dict[frame_idx]
            if vis > 0:
                img_pt = clamp_point_to_quad((bx, by), court_quad)
                ball_m_for_map = to_court_m(img_pt, homography)
                if ball_m_for_map is not None:
                    ball_x = ball_m_for_map[0] + ball_x_shift
                    ball_m_for_map = (
                        float(np.clip(ball_x, 0.0, args.court_width_m)),
                        float(np.clip(ball_m_for_map[1], 0.0, args.court_length_m)),
                    )

        near_m_for_map = to_court_m(near_draw_pt, homography) if near_draw_pt is not None else None
        far_m_for_map = to_court_m(far_draw_pt, homography) if far_draw_pt is not None else None
        near_m_for_map = enforce_half_court(near_m_for_map, "near", args.court_width_m, args.court_length_m)
        far_m_for_map = enforce_half_court(
            far_m_for_map,
            "far",
            args.court_width_m,
            args.court_length_m,
            far_half_expand=args.far_half_expand,
        )
        draw_mini_court(
            frame,
            far_m_for_map,
            near_m_for_map,
            ball_m_for_map,
            far_map_hist,
            near_map_hist,
            ball_map_hist,
            args.court_width_m,
            args.court_length_m,
        )

        if near_player_draw is not None:
            draw_player(frame, near_player_draw, ZH_BOTTOM, args.near_color, args.draw_foot_point, args.draw_id, args.draw_pose)
        if far_player_draw is not None:
            draw_player(frame, far_player_draw, ZH_TOP, args.far_color, args.draw_foot_point, args.draw_id, args.draw_pose)

        ui_fps = frame_idx / max(time.perf_counter() - t_start, 1e-6)
        draw_stats_panel(frame, rally_idx, far_stats, near_stats, pipeline_fps=ui_fps)

        # Optional debug: draw selected court quadrilateral.
        if args.draw_court_polygon:
            quad = court_quad.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [quad], True, (0, 255, 0), 2, cv2.LINE_AA)

        writer.write(frame)
        prev_gray = cur_gray
        frame_idx += 1

        if frame_idx % 200 == 0:
            print(f"[INFO] progress {frame_idx}/{total_frames}")

        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

    cap.release()
    writer.release()
    if frame_idx > 0:
        detect_ratio = detect_calls / float(frame_idx)
        print(f"[DONE] detect_calls={detect_calls}, detect_ratio={detect_ratio:.3f}")
    print("[DONE] finished")
    print(f"[DONE] output video: {args.output_path}")


def build_arg_parser():
    p = argparse.ArgumentParser(description="Player trajectory analytics overlay")
    p.add_argument("--video_path", type=str, default=DEFAULT_VIDEO_PATH)
    p.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH)
    p.add_argument("--ball_csv", type=str, default="")
    p.add_argument("--yolo_model", type=str, default="yolov8s-pose.pt")
    p.add_argument("--tracker_cfg", type=str, default="bytetrack.yaml")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf_thres", type=float, default=0.18)
    p.add_argument("--detect_interval", type=int, default=1, help="Run detector every N frames (1=full precision)")
    p.add_argument("--half", dest="half", action="store_true")
    p.add_argument("--no_half", dest="half", action="store_false")
    p.set_defaults(half=True)
    p.add_argument("--min_box_w", type=float, default=10.0)
    p.add_argument("--min_box_h", type=float, default=20.0)
    p.add_argument("--detect_on_court_roi", dest="detect_on_court_roi", action="store_true")
    p.add_argument("--no_detect_on_court_roi", dest="detect_on_court_roi", action="store_false")
    p.set_defaults(detect_on_court_roi=True)
    p.add_argument("--detect_roi_pad_x", type=int, default=28)
    p.add_argument("--detect_roi_pad_top", type=int, default=84)
    p.add_argument("--detect_roi_pad_bottom", type=int, default=18)
    p.add_argument("--top_inner_ratio", type=float, default=0.020, help="Suppress top off-court candidates")
    p.add_argument("--top_inner_keep_dist", type=float, default=100.0, help="Keep top candidate if close to previous far point")
    p.add_argument("--ankle_min_conf", type=float, default=0.22)
    p.add_argument("--far_fallback_lift_ratio", type=float, default=0.22, help="Lift far fallback foot when ankles are missing")
    p.add_argument("--far_zone_hint_margin", type=float, default=12.0)
    p.add_argument("--far_lock_max_jump", type=float, default=110.0)
    p.add_argument("--far_max_up_jump", type=float, default=30.0)
    p.add_argument("--side_band_px", type=float, default=16.0, help="Midline band for near/far side lock")
    p.add_argument("--far_half_expand", type=float, default=1.35, help="Expand far-side mini-court motion range")
    p.add_argument("--judge_hist_len", type=int, default=90)
    p.add_argument("--judge_min_frames", type=int, default=42)
    p.add_argument("--judge_top_zone_px", type=int, default=18)
    p.add_argument("--judge_top_ratio", type=float, default=0.88)
    p.add_argument("--judge_static_move_px", type=float, default=30.0)

    p.add_argument("--court_points", type=str, default="")
    p.add_argument("--select_court_points", dest="select_court_points", action="store_true")
    p.add_argument("--no_select_court_points", dest="select_court_points", action="store_false")
    p.set_defaults(select_court_points=True)
    p.add_argument("--court_width_m", type=float, default=6.1)
    p.add_argument("--court_length_m", type=float, default=13.4)

    p.add_argument("--trail_len", type=int, default=50)
    p.add_argument("--minimap_trail_len", type=int, default=120)
    p.add_argument("--trail_thickness", type=int, default=3)
    p.add_argument("--dash_len", type=int, default=12)
    p.add_argument("--gap_len", type=int, default=10)
    p.add_argument("--smooth_steps", type=int, default=12)
    p.add_argument("--near_color", type=lambda s: tuple(int(x) for x in s.split(",")), default=(255, 110, 210))
    p.add_argument("--far_color", type=lambda s: tuple(int(x) for x in s.split(",")), default=(0, 180, 255))

    p.add_argument("--draw_id", action="store_true")
    p.add_argument("--draw_pose", dest="draw_pose", action="store_true")
    p.add_argument("--no_draw_pose", dest="draw_pose", action="store_false")
    p.set_defaults(draw_pose=True)
    p.add_argument("--draw_pose_on_skipped", dest="draw_pose_on_skipped", action="store_true")
    p.add_argument("--no_draw_pose_on_skipped", dest="draw_pose_on_skipped", action="store_false")
    p.set_defaults(draw_pose_on_skipped=True)
    p.add_argument("--skip_pose_max_age", type=int, default=5, help="Max skipped-frame age to reuse cached pose")

    p.add_argument("--draw_foot_point", dest="draw_foot_point", action="store_true")
    p.add_argument("--no_draw_foot_point", dest="draw_foot_point", action="store_false")
    p.set_defaults(draw_foot_point=True)

    p.add_argument("--draw_ball_on_minimap", dest="draw_ball_on_minimap", action="store_true")
    p.add_argument("--no_draw_ball_on_minimap", dest="draw_ball_on_minimap", action="store_false")
    p.set_defaults(draw_ball_on_minimap=False)

    p.add_argument("--draw_court_polygon", action="store_true")

    p.add_argument("--new_rally_missing_frames", type=int, default=48)
    p.add_argument("--hold_missing_frames", type=int, default=10)
    p.add_argument("--reset_missing_frames", type=int, default=36)
    p.add_argument("--max_flow_move", type=float, default=45.0)
    p.add_argument("--max_frames", type=int, default=0)
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)
