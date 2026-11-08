# 分析与调试工具集 (`scripts/tools/`)

本目录包含羽毛球视频分析 Pipeline 的独立后处理模块与调试工具，既可由 `run_all_mac.sh` 自动串联调用，也可作为独立 CLI 或 Python 模块导入使用。

---

## 1. `rally_detector.py` — 自动回合分割、击球点检测与 FFmpeg 集锦剪辑

基于 TrackNet 输出的羽毛球轨迹 CSV（`*_ball.csv`），自动识别每个回合的起止帧 `[start_frame, end_frame]`、检测球拍击球瞬间（轨迹速度方向突变/反转），并支持一键调用 FFmpeg 批量切出单回合切片与全场集锦 `highlights.mp4`。

### CLI 用法

```bash
# 输出 rallies.json 与 rallies.csv 报告，并自动切分回合视频 + 生成集锦
python3 scripts/tools/rally_detector.py \
  --ball_csv ~/yumaoqiu_repro/tracknet_v3_result_regen/short_ball.csv \
  --fps 21.0 \
  --min_rally_frames 30 \
  --max_missing_frames 25 \
  --cut_video short.mp4 \
  --out_dir ~/yumaoqiu_repro/rallies
```

### Python 模块调用

```python
from scripts.tools.rally_detector import RallyDetector

detector = RallyDetector(fps=21.0, min_rally_frames=30, max_missing_frames=25)
rallies = detector.detect_from_csv("short_ball.csv")
boundaries = detector.get_boundaries()  # [(start_frame, end_frame), ...]
```

---

## 2. `smooth_ball_csv.py` — 羽毛球轨迹补漏与 2D 卡尔曼滤波平滑

使用纯 NumPy 实现的四状态 `[x, y, vx, vy]` 匀速模型二维卡尔曼滤波器（含 RTS 反向平滑）：
- 对短于 `--max_gap`（默认 8 帧）的漏检空洞做线性插值补齐（标记 `Source=interp`）；
- 对长于 `--max_gap` 的非比赛区间保留 `Visibility=0`（标记 `Source=none`）；
- 对有效轨迹段做零相位延迟卡尔曼平滑，消除高速飞行与边缘检测抖动。

```bash
python3 scripts/tools/smooth_ball_csv.py \
  ~/yumaoqiu_repro/tracknet_v3_result_regen/short_ball.csv \
  --output ~/yumaoqiu_repro/tracknet_v3_result_regen/short_ball_smoothed.csv \
  --max_gap 8
```

---

## 3. `heatmap.py` — BWF 标准球场覆盖热力图与击球落点散点图

读取 `overlay_player_analytics.py --export_json` 导出的 `analytics.json` 或 TrackNet 的 `*_ball.csv`（配合 `--court_points`），渲染双面板出版级 PNG（`court_heatmap.png`）：
- **左面板（Panel 1）**：基于标准 6.1m × 13.4m BWF 球场坐标系的球员跑动覆盖热力图（或羽毛球轨迹密度图）；
- **右面板（Panel 2）**：BWF 球场线上的羽毛球飞行轨迹与击球/落点散点分布图。

```bash
# 从 analytics.json 生成
python3 scripts/tools/heatmap.py ~/yumaoqiu_repro/analytics.json --output court_heatmap.png

# 或直接从 *_ball.csv + 球场角点生成
python3 scripts/tools/heatmap.py \
  ~/yumaoqiu_repro/tracknet_v3_result_regen/short_ball.csv \
  --court_points "352,342,628,343,944,527,52,532" \
  --output court_heatmap.png
```

---

## 4. 标定与诊断工具

- `select_court.py <video.mp4>` — 交互式点击标注球场 4 个角点（顺序：左上 TL → 右上 TR → 右下 BR → 左下 BL），终端输出 `--court_points` 参数字符串。
- `diag_tracknet.py` — 抽样诊断 TrackNet 热力图响应峰值分布，用于确定最佳 `--vis_thresh` 二值化阈值。
- `render_panel_preview.py` — 无需运行深度模型，单独渲染左侧中文数据统计面板样式预览。
