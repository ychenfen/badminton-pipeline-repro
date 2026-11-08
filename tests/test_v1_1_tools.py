import csv
import json
import math
import os
import sys
import tempfile
import unittest
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.tools import heatmap, rally_detector, smooth_ball_csv
from scripts.tracknet_runtime import predict as tracknet_predict


class TestTrackNetMPSAndVisThresh(unittest.TestCase):
    def test_mps_fallback_env_is_set(self):
        self.assertEqual(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"), "1")

    def test_arg_parser_device_choices_and_vis_thresh(self):
        parser = tracknet_predict.build_arg_parser()
        args_default = parser.parse_args([])
        self.assertEqual(args_default.device, "auto")
        self.assertAlmostEqual(args_default.vis_thresh, 0.15, places=4)

        args_mps = parser.parse_args(["--device", "mps", "--vis_thresh", "0.18", "--eval_mode", "nonoverlap"])
        self.assertEqual(args_mps.device, "mps")
        self.assertAlmostEqual(args_mps.vis_thresh, 0.18, places=4)
        self.assertEqual(args_mps.eval_mode, "nonoverlap")

    def test_resolve_device_mps_and_cpu(self):
        dev_cpu = tracknet_predict.resolve_device("cpu")
        self.assertEqual(dev_cpu.type, "cpu")

        dev_mps = tracknet_predict.resolve_device("mps")
        self.assertEqual(dev_mps.type, "mps")

        dev_auto = tracknet_predict.resolve_device("auto")
        self.assertIn(dev_auto.type, ("cuda", "mps", "cpu"))


class TestSmoothBallCsv(unittest.TestCase):
    def test_gap_interpolation_and_kalman_smoothing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            in_csv = os.path.join(tmpdir, "raw_ball.csv")
            out_csv = os.path.join(tmpdir, "smoothed_ball.csv")

            rows = []
            # Frames 0..30: active trajectory with a 4-frame gap (frames 10..13)
            for f in range(31):
                if 10 <= f <= 13:
                    rows.append({"Frame": f, "Visibility": 0, "X": 0, "Y": 0})
                else:
                    jitter = 3 if (f % 2 == 0) else -3
                    rows.append(
                        {
                            "Frame": f,
                            "Visibility": 1,
                            "X": 200 + f * 8 + jitter,
                            "Y": 300 - f * 4 - jitter,
                        }
                    )
            # Frames 31..45: 15-frame gap (> max_gap=8), should remain Visibility=0, Source='none'
            for f in range(31, 46):
                rows.append({"Frame": f, "Visibility": 0, "X": 0, "Y": 0})
            # Frames 46..55: active again
            for f in range(46, 56):
                rows.append({"Frame": f, "Visibility": 1, "X": 400 + (f - 46) * 5, "Y": 250 + (f - 46) * 3})

            with open(in_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["Frame", "Visibility", "X", "Y"])
                writer.writeheader()
                writer.writerows(rows)

            saved_path, stats = smooth_ball_csv.smooth_ball_csv(in_csv, out_csv, max_gap=8)
            self.assertEqual(saved_path, out_csv)
            self.assertTrue(os.path.exists(out_csv))
            self.assertEqual(stats["total_frames"], 56)
            self.assertEqual(stats["interp_frames"], 4)
            self.assertEqual(stats["post_visible_frames"], stats["raw_visible_frames"] + 4)

            with open(out_csv, "r", encoding="utf-8") as f:
                out_rows = list(csv.DictReader(f))

            self.assertEqual(len(out_rows), 56)
            # Short gap (10..13) should be interpolated
            for f in range(10, 14):
                self.assertEqual(int(out_rows[f]["Visibility"]), 1)
                self.assertEqual(out_rows[f]["Source"], "interp")
                self.assertGreater(int(out_rows[f]["X"]), 250)

            # Long gap (31..45) should remain uninterpolated
            for f in range(31, 46):
                self.assertEqual(int(out_rows[f]["Visibility"]), 0)
                self.assertEqual(out_rows[f]["Source"], "none")
                self.assertEqual(int(out_rows[f]["X"]), 0)
                self.assertEqual(int(out_rows[f]["Y"]), 0)


class TestRallyDetector(unittest.TestCase):
    def _build_synthetic_match_rows(self):
        rows = []
        frame = 0
        # 10 leading empty frames
        for _ in range(10):
            rows.append({"Frame": frame, "Visibility": 0, "X": 0, "Y": 0})
            frame += 1

        # Rally 1: 60 frames with 3 clear back-and-forth shuttle arcs (hits at +15, +30, +45)
        for k in range(60):
            if k < 15:
                x = 300 + k * 10
                y = 400 - k * 12
            elif k < 30:
                x = 450 - (k - 15) * 10
                y = 220 + (k - 15) * 12
            elif k < 45:
                x = 300 + (k - 30) * 12
                y = 400 - (k - 30) * 10
            else:
                x = 480 - (k - 45) * 11
                y = 250 + (k - 45) * 11
            rows.append({"Frame": frame, "Visibility": 1, "X": int(x), "Y": int(y)})
            frame += 1

        # 35 missing frames between rallies (> max_missing_frames=25)
        for _ in range(35):
            rows.append({"Frame": frame, "Visibility": 0, "X": 0, "Y": 0})
            frame += 1

        # Short 8-frame noise blip (< min_rally_frames=30), should be filtered out
        for k in range(8):
            rows.append({"Frame": frame, "Visibility": 1, "X": 500 + k * 4, "Y": 300 - k * 4})
            frame += 1

        # 30 missing frames
        for _ in range(30):
            rows.append({"Frame": frame, "Visibility": 0, "X": 0, "Y": 0})
            frame += 1

        # Rally 2: 45 frames with a 5-frame internal occlusion and 2 hits
        for k in range(45):
            if 20 <= k < 25:
                rows.append({"Frame": frame, "Visibility": 0, "X": 0, "Y": 0})
            else:
                if k < 18:
                    x = 350 + k * 9
                    y = 420 - k * 11
                elif k < 32:
                    x = 512 - (k - 18) * 10
                    y = 222 + (k - 18) * 11
                else:
                    x = 372 + (k - 32) * 10
                    y = 376 - (k - 32) * 10
                rows.append({"Frame": frame, "Visibility": 1, "X": int(x), "Y": int(y)})
            frame += 1

        return rows

    def test_rally_segmentation_hits_and_exports(self):
        rows = self._build_synthetic_match_rows()
        detector = rally_detector.RallyDetector(
            fps=21.0,
            min_rally_frames=30,
            max_missing_frames=25,
            min_hit_spacing=6,
        )
        rallies = detector.detect(rows)
        self.assertEqual(len(rallies), 2)

        boundaries = detector.get_boundaries()
        self.assertEqual(boundaries[0], (10, 69))
        self.assertEqual(boundaries[1], (143, 187))

        # Check hit events spacing >= 6 frames
        for r in rallies:
            self.assertGreaterEqual(r["num_hits"], 2)
            hits = r["hit_frames"]
            for i in range(1, len(hits)):
                self.assertGreaterEqual(hits[i] - hits[i - 1], 6)

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "rallies.json")
            csv_path = os.path.join(tmpdir, "rallies.csv")
            detector.export_json(rallies, json_path, source_csv="synthetic_ball.csv")
            detector.export_csv(rallies, csv_path)

            with open(json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["num_rallies"], 2)
            self.assertEqual(len(payload["rallies"]), 2)

            with open(csv_path, "r", encoding="utf-8") as f:
                csv_rows = list(csv.DictReader(f))
            self.assertEqual(len(csv_rows), 2)
            self.assertEqual(int(csv_rows[0]["start_frame"]), 10)
            self.assertEqual(int(csv_rows[0]["end_frame"]), 69)

            # Test FFmpeg clipping with mocked subprocess.run
            dummy_video = os.path.join(tmpdir, "dummy.mp4")
            with open(dummy_video, "wb") as f:
                f.write(b"dummy")
            with mock.patch("subprocess.run") as mock_run:
                clips = detector.cut_rallies(dummy_video, rallies, out_dir=tmpdir, concat_highlights=True)
                self.assertEqual(len(clips), 2)
                # 2 rally clips + 1 concat highlights call = 3 ffmpeg calls
                self.assertEqual(mock_run.call_count, 3)


class TestCourtHeatmap(unittest.TestCase):
    def test_heatmap_from_analytics_json_and_ball_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Test from analytics.json
            json_path = os.path.join(tmpdir, "analytics.json")
            out_png_1 = os.path.join(tmpdir, "heatmap_from_json.png")
            frames = []
            for i in range(60):
                frames.append(
                    {
                        "frame": i,
                        "near": {"court_m": [1.5 + (i % 10) * 0.3, 8.5 + (i % 8) * 0.4]},
                        "far": {"court_m": [2.0 + (i % 10) * 0.25, 2.5 + (i % 8) * 0.35]},
                        "ball": {
                            "visible": 1,
                            "court_m": [
                                3.0 + 1.5 * math.sin(i * 0.25),
                                6.7 + 4.5 * math.cos(i * 0.20),
                            ],
                        },
                    }
                )
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "court_width_m": 6.1,
                        "court_length_m": 13.4,
                        "frames": frames,
                    },
                    f,
                )

            _, summary1 = heatmap.render_court_heatmap(json_path, out_png_1)
            self.assertTrue(os.path.exists(out_png_1))
            self.assertGreater(os.path.getsize(out_png_1), 1000)
            self.assertEqual(summary1["num_density_points"], 120)
            with open(out_png_1, "rb") as f:
                self.assertEqual(f.read(8), b"\x89PNG\r\n\x1a\n")

            # 2. Test from *_ball.csv + --court_points
            csv_path = os.path.join(tmpdir, "sample_ball.csv")
            out_png_2 = os.path.join(tmpdir, "heatmap_from_csv.png")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["Frame", "Visibility", "X", "Y"])
                writer.writeheader()
                for i in range(50):
                    writer.writerow(
                        {
                            "Frame": i,
                            "Visibility": 1,
                            "X": int(450 + 120 * math.sin(i * 0.3)),
                            "Y": int(420 + 80 * math.cos(i * 0.25)),
                        }
                    )

            _, summary2 = heatmap.render_court_heatmap(
                csv_path,
                out_png_2,
                court_points_str="352,342,628,343,944,527,52,532",
            )
            self.assertTrue(os.path.exists(out_png_2))
            self.assertEqual(summary2["num_ball_points"], 50)
            with open(out_png_2, "rb") as f:
                self.assertEqual(f.read(8), b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
