# 羽毛球视频分析 Pipeline — 完整交接文档

适用对象：
- 刚接手这个项目、可能不熟悉计算机视觉、需要从零跑通的工程师
- AI 编程助手（Codex / Cursor / Claude / Copilot Agent 等）— §10 的每个 Task 都按"独立可执行任务包"格式写：**背景 / 目标 / 步骤 / 涉及文件 / 验证 / 已知陷阱**，AI 可直接挑一个任务从这里起步，不需要外部上下文。

如何使用本文档：
- 第一次跑通：读 §0-§5
- 想理解原理：读 §6-§7
- 跑出问题：读 §8-§9
- 想推进项目：读 §10-§11

---

## 0. 一句话讲清这个项目

这套程序吃一段羽毛球比赛视频，输出一段叠加了**球员轨迹、移动速度、累计距离、球的飞行轨迹**的视频，最后还能打上电影级"子弹时间"特效。

实际跑下来要经过三段独立的程序，串成一条链路：

```
原始视频.mp4
    │
    ▼  Step 1：TrackNet（检测羽毛球的位置）
带球轨迹的视频 + 球坐标 CSV
    │
    ▼  Step 2：Overlay（检测球员 + 画统计面板 + 画小球场轨迹图）
叠加分析的视频
    │
    ▼  Step 3：FX（加子弹时间、慢动作、画面特效）
最终成品视频
```

每一段用什么模型、为什么这么分、每一步在动什么数据，下文逐项展开。

---

## 1. 三段为什么要拆开

如果把它看成一个黑盒，"喂视频出视频"听起来像一个程序就够了。实际拆三段是因为：

1. **球**和**球员**是两个完全不同的检测问题。
   - 球只有几个像素、飞得快、容易模糊。要用专门追小目标的模型（TrackNet，连续帧热力图）。
   - 球员是大目标，常见 YOLO 系列就能搞定。
2. **几何分析**（计算速度、距离、跑动轨迹）必须先把球场从斜拍视角"拉平"成俯视图。这一步只发生在 Step 2 内部，跟检测本身关系不大。
3. **特效**（子弹时间、缩放、色散）是纯渲染层，跟模型无关。改一次特效不该重跑模型，所以单独成段。

这种分层让调试便宜很多：改面板字号、调跳变阈值、加新特效，都可以**只重跑某一段**而不用从头来。

---

## 2. 目录结构

```
pipeline_repro_bundle/
├── README.md                       # 原始 readme（Windows 思路）
├── README_MAC.md                   # Mac 启动笔记
├── HANDOVER.md                     # 本文档
├── CHAIN_EVIDENCE.md               # 原作者解释为什么这条链是"最可信"的
├── run_all.ps1                     # Windows PowerShell 一键脚本
├── run_all_mac.sh                  # macOS bash 一键脚本
├── requirements_repro.txt          # 顶层依赖
├── short.mp4                       # 跑通验证用的 30 秒短视频（你放进来的样本）
├── b13b2c0b078c64ca95063c958e2fbfd9.mp4  # 你的全长 10 分 35 秒视频
│
├── weights/
│   ├── TrackNet_best.pt            # 球检测模型权重 130 MB
│   └── yolov8s-pose.pt             # 球员姿态检测模型权重 23 MB
│
└── scripts/
    ├── tracknet_runtime/           # 第 1 段
    │   ├── predict.py              # 入口
    │   ├── model.py                # 网络结构
    │   ├── dataset.py              # 把视频切成连续帧 batch
    │   ├── test.py                 # 工具函数
    │   └── utils/
    │       └── general.py          # HEIGHT=288, WIDTH=512, get_model() 等常量
    │
    ├── overlay/
    │   └── overlay_player_analytics.py   # 第 2 段，1340+ 行的核心
    │
    ├── fx/
    │   └── video_fx_bullet_time.py       # 第 3 段
    │
    └── _select_court.py            # 工具：手动点 4 个球场角点
```

不在仓库里、运行时生成的产物：

```
~/yumaoqiu_repro/                                      # 默认输出根目录
├── tracknet_v3_result_regen/
│   ├── short.mp4                                      # TrackNet 输出视频（带球轨迹圈）
│   ├── short_tracknetv3.mp4                           # Step 2 期望的命名（手动 cp 一份）
│   └── short_ball.csv                                 # 球的逐帧坐标
├── end1_fix_swap2_precision_full_regen.mp4            # Step 2 输出
└── end1_fix_swap2_precision_full_fx_regen.mp4         # Step 3 输出（最终）
```

---

## 3. 环境配置（macOS）

测试机：MacBook Apple M4 Pro / macOS 15.7。Linux 同样能跑，Windows 走 `run_all.ps1`。

### 3.1 Python

系统 Python 3.9.6 即可（Xcode 自带的 `/usr/bin/python3`）。不用装 Anaconda。

```bash
python3 --version       # 期望 Python 3.9.x
which python3           # /usr/bin/python3
```

### 3.2 依赖

PyPI 镜像有时被代理拦截，建议显式指定 `--index-url`：

```bash
python3 -m pip install --user --index-url https://pypi.org/simple \
    numpy opencv-python pandas Pillow torch ultralytics tqdm \
    pycocotools parse lap
```

**为什么是这些**：
- `torch`：跑 TrackNet
- `ultralytics`：跑 YOLOv8 + ByteTrack
- `opencv-python`：所有图像 I/O、变换、绘图
- `pycocotools` / `parse`：TrackNet 代码 import 时连带要的（即使你不评估也要装）
- `lap`：ultralytics 的 ByteTrack 内部做"线性指派"用的（匈牙利匹配）

### 3.3 ffmpeg

切短视频和检查视频信息要用：

```bash
brew install ffmpeg
```

### 3.4 字体

Mac 自带 PingFang.ttc，能渲染中文。已经写到 `overlay_player_analytics.py` 里的字体回退列表，无需手动配置。

---

## 4. 快速跑通（30 秒短视频）

这一节是最常用的复现路径，照抄即可。所有命令在仓库根目录 `pipeline_repro_bundle/` 下执行。

### 4.1 准备一个 30 秒切片用于验证

为什么不直接跑全长？TrackNet 在 CPU 上推理一帧 ~0.16 秒，630 帧 ≈ 100 秒；如果是 13344 帧的全长视频会跑差不多 3 小时。先用短的把链路跑通。

```bash
ffmpeg -y -ss 0 -i b13b2c0b078c64ca95063c958e2fbfd9.mp4 -t 30 -c:v libx264 -preset fast -an short.mp4
```

含义：
- `-ss 0` 从第 0 秒开始
- `-t 30` 截 30 秒
- `-c:v libx264` 用 H.264 重新编码（保证 OpenCV 能读）
- `-an` 去掉音轨
- `-preset fast` 编码速度优先

### 4.2 标注球场 4 个角点（关键）

整个 pipeline 里**唯一需要人工介入**的环节。

```bash
python3 scripts/_select_court.py short.mp4
```

弹窗里照顺序点 4 个点：**TL → TR → BR → BL**（左上、右上、右下、左下，整个球场长方形的 4 角）。

点完按 `q` 关窗，终端会输出一行 `--court_points "x1,y1,x2,y2,x3,y3,x4,y4"`，复制保留。

本视频实测点位：

```
352,342,628,343,944,527,52,532
```

为什么必须人工点？因为目前没接入"自动球场线检测"。每个比赛视频的机位、角度、广告板都不一样，自动检测做不准还会引入误差。让人工点 4 个像素值是最稳妥的折中。

### 4.3 跑 TrackNet（球检测）

**关键：必须设环境变量 `TRACKNET_VIS_THRESH`，否则一帧球都检测不出来**（详见后文 §6.1.4 的踩坑记录）。

```bash
TRACKNET_VIS_THRESH=0.15 python3 scripts/tracknet_runtime/predict.py \
  --video_file ~/Desktop/pipeline_repro_bundle/short.mp4 \
  --tracknet_file weights/TrackNet_best.pt \
  --save_dir ~/yumaoqiu_repro/tracknet_v3_result_regen \
  --output_video --device auto --large_video --eval_mode nonoverlap
```

跑完会生成两个文件：

- `~/yumaoqiu_repro/tracknet_v3_result_regen/short.mp4`：在原视频上画了球轨迹圆圈的版本
- `~/yumaoqiu_repro/tracknet_v3_result_regen/short_ball.csv`：每帧的 `Frame, Visibility, X, Y`

CSV 抽样验证：

```bash
python3 -c "
import csv
rows = list(csv.DictReader(open('/Users/yuchenxu/yumaoqiu_repro/tracknet_v3_result_regen/short_ball.csv')))
total = len(rows)
visible = sum(1 for r in rows if int(float(r['Visibility']))>0)
print(f'{visible}/{total} = {100*visible/total:.1f}% visible')
"
```

期望输出：`598/630 = 94.9% visible`。低于 50% 说明阈值还要再降。

### 4.4 文件改名（Step 2 期望的命名约定）

```bash
cp ~/yumaoqiu_repro/tracknet_v3_result_regen/short.mp4 \
   ~/yumaoqiu_repro/tracknet_v3_result_regen/short_tracknetv3.mp4
```

为什么要 `cp`？原项目跑 PowerShell 脚本里会自动做这件事，但 Mac 脚本 `run_all_mac.sh` 假定 Step 1 输出叫 `_tracknetv3.mp4`，单独跑 Step 1 时没改名，要手动补。

### 4.5 跑 Overlay（球员检测 + 数据叠加）

```bash
python3 scripts/overlay/overlay_player_analytics.py \
  --video_path ~/yumaoqiu_repro/tracknet_v3_result_regen/short_tracknetv3.mp4 \
  --output_path ~/yumaoqiu_repro/end1_fix_swap2_precision_full_regen.mp4 \
  --ball_csv ~/yumaoqiu_repro/tracknet_v3_result_regen/short_ball.csv \
  --yolo_model weights/yolov8s-pose.pt \
  --tracker_cfg bytetrack.yaml \
  --no_select_court_points \
  --court_points "352,342,628,343,944,527,52,532" \
  --device mps \
  --draw_court_polygon \
  --draw_ball_on_minimap \
  --detect_interval 1
```

各参数的意义在 §6.2 详细解释。

跑完得到 `~/yumaoqiu_repro/end1_fix_swap2_precision_full_regen.mp4`，30 秒、约 13 MB。

### 4.6 跑 FX 子弹时间（可选）

```bash
python3 scripts/fx/video_fx_bullet_time.py \
  --input ~/yumaoqiu_repro/end1_fix_swap2_precision_full_regen.mp4 \
  --output ~/yumaoqiu_repro/end1_fix_swap2_precision_full_fx_regen.mp4
```

得到最终成品。

---

## 5. 全长视频怎么跑

### 5.1 时间预估

| 阶段 | CPU（M4 Pro） | MPS（M4 Pro GPU） |
|---|---|---|
| TrackNet（13344 帧） | ~3 小时 | 不支持（代码只走 CPU/CUDA） |
| Overlay（13344 帧） | ~30 分钟 | ~10 分钟 |
| FX（13344+ 帧） | ~5 分钟 | ~5 分钟（不依赖 GPU） |

最大瓶颈是 TrackNet。如果不优化，一次完整跑 ≈ 3.5 小时。优化路线见 §10。

### 5.2 命令

```bash
TRACKNET_VIS_THRESH=0.15 python3 scripts/tracknet_runtime/predict.py \
  --video_file ~/Desktop/pipeline_repro_bundle/b13b2c0b078c64ca95063c958e2fbfd9.mp4 \
  --tracknet_file weights/TrackNet_best.pt \
  --save_dir ~/yumaoqiu_repro/tracknet_v3_result_regen \
  --output_video --device auto --large_video --eval_mode nonoverlap
```

跑完同样手动 `cp` 一份带 `_tracknetv3` 后缀的，再跑 Overlay + FX。

建议挂在 `nohup` 里：

```bash
nohup bash -c "TRACKNET_VIS_THRESH=0.15 python3 scripts/tracknet_runtime/predict.py ..." \
  > /tmp/tracknet_full.log 2>&1 &
```

然后 `tail -f /tmp/tracknet_full.log` 看进度。

---

## 6. 三段详解

### 6.1 Step 1 — TrackNet（球检测）

#### 6.1.1 它在解决什么问题

YOLO 这种单帧检测器对羽毛球这种**小（5-10 像素）、快（高速飞行时模糊成一道线）、易遮挡**的目标效果差，单帧经常漏检。

TrackNet 不是单帧检测，它一次吃 4 帧（连续画面），输出 4 张"热力图"。热力图上每个像素的值表示"这里有球的概率"。利用连续帧的运动信息，模糊的球也能被识别。

简单类比：你看一张静态照片可能看不出蚊子在哪，但看 4 张连拍就能看出"有什么东西在那一带飞过"。TrackNet 学的就是这种连拍判断。

#### 6.1.2 内部数据流

```
视频 → 每 4 帧切一组 → 缩放到 512×288 → 减去/拼接背景中位数图 →
   神经网络（U-Net 风格） → 4 张热力图（尺寸都是 512×288）→
      二值化（>= 阈值的像素算"有球"）→ 找连通块的中心 → 还原到原始 960×544 坐标
```

"背景中位数图"是把整个视频随机抽样 200 帧，每个像素位置取所有帧的中位数。中位数图基本上是空场地（动的人和球被滤掉）。当前帧减去/拼接这张图，等于告诉网络"我重点关心动的东西"。

模型权重 `TrackNet_best.pt` 是预训练好的，不需要自己训。

#### 6.1.3 关键参数

| 参数 | 含义 | 默认 / 推荐 |
|---|---|---|
| `--video_file` | 输入视频路径 | （必填） |
| `--tracknet_file` | 模型权重路径 | `weights/TrackNet_best.pt` |
| `--save_dir` | 输出目录 | （必填） |
| `--output_video` | 是否生成可视化视频 | 加上 |
| `--device` | 推理设备 | `auto`（Mac 走 CPU；NVIDIA 卡走 cuda） |
| `--large_video` | 用流式 dataloader，不一次性把整段视频塞进内存 | 长视频必须加 |
| `--eval_mode` | `nonoverlap` 或 `weight` | 推荐 `nonoverlap`，速度快 8 倍，效果差距很小 |
| `TRACKNET_VIS_THRESH`（环境变量） | 二值化阈值 | **必须设 0.15-0.2，详见下条** |

#### 6.1.4 已知坑：阈值默认 0.5 在大多数视频上失效

`predict.py:35` 原版写死：

```python
y_pred = y_pred > 0.5
```

实测在 960×544 的比赛视频上，模型 heatmap 最大值通常在 **0.20–0.34** 之间，一辈子过不了 0.5 这条线。结果就是 600+ 帧全部被判 `Visibility=0`，CSV 里 X/Y 全是 0。

**最直接的修法**：把那一行改成读环境变量：

```python
thresh = float(os.environ.get("TRACKNET_VIS_THRESH", "0.2"))
y_pred = y_pred > thresh
```

跑命令时前面带 `TRACKNET_VIS_THRESH=0.15`。

阈值取多少？根据 `_diag_tracknet.py` 在 4 个采样帧实测：

| 阈值 | 检出率 | 误检风险 |
|---|---|---|
| 0.50（原版） | 0% | 0 |
| 0.20 | 80% 左右 | 极低 |
| 0.15 | 94.9% | 低 |
| 0.10 | 98% 左右 | 中等（背景偶尔出假点） |

实测 0.15 在 short.mp4 上跑出 598/630 = 94.9%，比例合理（球被网柱遮挡或离开画面的帧本来就该是 invisible）。

#### 6.1.5 输出文件含义

**`short_ball.csv`**

```
Frame,Visibility,X,Y
0,1,455,202
1,1,455,202
2,1,455,202
3,1,455,202
4,1,481,122
...
```

- `Frame`：帧号（从 0 开始）
- `Visibility`：1 = 这一帧检测到了球，0 = 没检测到
- `X, Y`：球在原始 960×544 画面里的像素坐标

**`short.mp4`**：把球的轨迹圆圈画在原视频上的可视化版本。

### 6.2 Step 2 — Overlay（球员检测 + 数据叠加）

整个项目 90% 的工程量在这一段。1340+ 行代码，干 5 件事：

1. **检测每一帧的球员**（YOLOv8s-pose）
2. **维持球员 ID 不串**（ByteTrack + 上下半场二选一逻辑）
3. **把像素坐标变换到俯视球场坐标**（透视矫正）
4. **算每个球员的距离/速度/累计**（运动学统计）
5. **把所有结果叠加到视频上**：左侧统计面板、右侧 Mini Court、球员骨架、轨迹线

#### 6.2.1 YOLOv8s-pose 是什么

YOLO（You Only Look Once）是一系列实时目标检测模型。`yolov8s-pose` 是 YOLOv8 的"小号"版本（22 MB），同时输出**检测框 + 17 个人体关键点**（头、肩、肘、腕、髋、膝、踝）。

为什么要 pose 不只用框？因为我们需要球员"脚"的位置来对应球场坐标。bbox 中心是身体中心，离地有 1 米多高；用脚踝关键点更准。代码里 `estimate_foot_point()` 优先用左右脚踝平均，关键点置信度太低再退到 bbox 底边中心。

#### 6.2.2 ByteTrack 怎么让 ID 不串

YOLO 一帧只能告诉你"这里有个人"，不能告诉你"这个人是上一帧那个人"。ByteTrack 是**多目标跟踪算法**，干的事是：

1. 每一帧拿 YOLO 输出的检测框
2. 跟历史轨迹做匈牙利匹配（IoU + 运动学预测）
3. 给每个稳定轨迹分配一个不变的 ID
4. 即使中间有几帧漏检，也能在球员重新出现时把同一个 ID 接回去

**ByteTrack 已经接好了**：`overlay_player_analytics.py` 第 1260 行 `tracker_cfg=bytetrack.yaml`，第 973 行 `model.track(... persist=True)` 调用。不需要额外配置。

#### 6.2.3 上下半场分配（assign_players）

ByteTrack 给的 ID 是稳定的，但每帧画面里可能有 0–N 个 ID。我们只关心"上半场的球员"和"下半场的球员"两个代表。`assign_players` 函数做这件事：

- 用画面里球场中线（计算自 4 角点）划上下两半
- 在上半场区域里挑一个 ID 作为 far（远端球员）
- 在下半场区域里挑一个 ID 作为 near（近端球员）
- 选择优先级：先看 ID 是不是上一帧那个，再看离上一帧位置最近的一个，再退到上下半场各自最极端的一个
- 还有反向跳变保护：如果"远端代表"突然往画面下移很多，宁可丢掉这一帧也不接

这套规则是手写的启发式，不是模型。改它要小心，每个分支都对应一类失败案例。

#### 6.2.4 透视矫正：从画面像素到球场米

电视机位是斜拍，画面里的球场是个**梯形**（远端窄，近端宽）。球员的"像素坐标位移"不等于"实际跑动距离"——同样跑 1 米，在远端可能只占 30 个像素，在近端可能占 80 个像素。要算真实距离必须先把这个梯形拉成长方形。

数学工具：**单应性变换矩阵 Homography**。它是个 3×3 矩阵 H，能把画面里 4 个点映射到指定位置。OpenCV 一行调用：

```python
src = court_quad           # 你点的 4 个画面坐标 (TL, TR, BR, BL)
dst = [[0,0], [6.1,0], [6.1,13.4], [0,13.4]]   # 标准球场尺寸（米）
H, _ = cv2.findHomography(src, dst)
```

之后任何一个球员脚点坐标 `(px, py)` 都能投影成球场上的 `(x_meter, y_meter)`：

```python
mx, my = cv2.perspectiveTransform([[px, py]], H)
```

球场标准尺寸：
- 双打宽 6.1 m → `--court_width_m`
- 全场长 13.4 m → `--court_length_m`

**思想**：把"研究斜视图"转化成"研究俯视图"。所有距离/速度/轨迹都在俯视图坐标系里算，结果跟机位无关。

#### 6.2.5 球场角点为什么必须人工标

理论上能用 Hough 变换找球场白线，再求交点。实际：
- 真实比赛里底线常被广告牌遮挡
- 不同机位畸变不一
- 误差 5 像素就会让球员速度翻倍

人工点 4 下成本低、最准。后续如果固定一个机位拍多个视频，4 个角点可以复用同一组数字。

#### 6.2.6 MotionStats 类——速度和距离怎么算

每个球员（near、far）一个 `MotionStats` 实例。每帧做这件事：

```
本帧脚点 → 投影到球场坐标 (x_m, y_m)
↓
跟上一帧坐标算欧氏距离 distance
↓
判断 distance 是不是真实运动（详见跳变阈值）
↓
若是真实：
  - 累加到 rally_distance, total_distance
  - 计算 inst_speed = distance / dt
  - 更新 rally_max_speed, total_max_speed
若不是（被判定 ID 跳变）：
  - 不累加，inst_speed = 0
↓
current_speed = 最近 6 帧 inst_speed 的滑动平均（去抖）
```

#### 6.2.7 跳变阈值（这次踩过的关键坑）

原代码硬写：

```python
if distance <= 1.2:        # 米
    inst_speed = distance / dt
```

意思是"位移大于 1.2 米的帧间跳变就当 ID 串了不算"。但 21 fps 视频的 dt = 0.048 秒，这阈值放行的速度上限是 1.2 / 0.048 = **25 m/s**。一个人羽毛球场上极限冲刺也就 7 m/s，所以 ByteTrack 偶尔串一帧 ID（球员 A 的位置突然跳到球员 B 那里），这个跳变也会被当真，被记进 `rally_max_speed`。结果就是面板显示"最高速度 24 m/s"——比博尔特还快。

**修法**：把阈值改成动态：

```python
max_jump_m = 8.0 * dt + 0.05
if distance <= max_jump_m:
    ...
```

含义：每帧最多放行 ≤ 8 m/s 的位移（留 0.05 米的小常数避免极小 dt 下被误判）。8 m/s 已经留了点余量给极端冲刺。

实测前后对比（同一视频同一帧 300）：

| 指标 | 阈值 1.2m | 8×dt+0.05 |
|---|---:|---:|
| 上半场最高速度 | 16 m/s | 9 m/s |
| 下半场最高速度 | 24 m/s | 9 m/s |
| 总距离 | 偏大（含虚假累加） | 合理 |

#### 6.2.8 Mini Court 是什么

视频右上角那个小矩形。把当前帧球员/球的真实球场坐标按比例画在一个 200×320 的小图上。颜色：

- 上半场球员：黄
- 下半场球员：粉
- 球：青

每个点是过去 N 帧的位置历史（`--minimap_trail_len 120` 控制）。

它的作用是把"长时间累计的运动"压成一张静态图——你不用看完整段视频也能一眼判断球员主要在哪片区域活动。

#### 6.2.9 左侧统计面板

原版每个球员有 7 项数据。我们已经砍到 4 项：

- 当前速度（最近 6 帧滑动平均）
- 回合距离（当前回合累计跑动）
- 回合最高（当前回合的最大瞬时速度）
- 总距离（视频开头到现在的累计跑动）

砍掉的：回合均速、总均速、总最高速度——这三项要么跟其他指标线性相关，要么对观众价值低。

#### 6.2.10 完整参数表（最常调的）

| 参数 | 含义 | 默认 | 调参建议 |
|---|---|---|---|
| `--video_path` | 输入（Step 1 输出）视频 | — | — |
| `--output_path` | 输出叠加视频 | — | — |
| `--ball_csv` | Step 1 出的球坐标 CSV | — | — |
| `--court_points` | 4 角点像素坐标 | — | 必填 |
| `--court_width_m` | 球场宽 | 6.1 | 双打 6.1，单打 5.18 |
| `--court_length_m` | 球场长 | 13.4 | 全场 13.4，半场 6.7 |
| `--device` | YOLO 推理设备 | 空（auto） | Mac 用 `mps` 加速 |
| `--detect_interval` | YOLO 跳帧间隔 | 1 | 加速时改 2 或 3 |
| `--draw_ball_on_minimap` | 是否画球轨迹 | 关 | **想看球时必须加** |
| `--draw_court_polygon` | 把球场 quad 画绿线（调试用） | 关 | 调角点时打开 |
| `--trail_len` | 主画面里球员轨迹长度 | 50 | — |
| `--minimap_trail_len` | Mini court 轨迹长度 | 120 | — |

### 6.3 Step 3 — FX（子弹时间特效）

#### 6.3.1 子弹时间是什么

电影《黑客帝国》里 Neo 躲子弹那个慢镜头围绕镜头转的镜头。这里我们用单机位视频模拟一个"轻量版"子弹时间：

- 选定一些时间点（"子弹时刻"）
- 在该时刻**冻帧**几十帧，期间用虚拟相机做小幅度的旋转 + 缩放，给观众"环绕"的错觉
- 冻帧之后接一段**慢动作**（每个原始帧重复播放 N 次，可选插值）
- 然后回到正常播放

这个 Step 完全不依赖检测结果，纯粹是对 Step 2 的输出视频做后期。

#### 6.3.2 怎么挑"子弹时刻"

三种方式可叠加：

1. **手动指定时间点**：`--bullet_times "12.4,48.0,87.2"`
2. **均匀分布 N 个**：`--uniform_bullet_count 12 --uniform_margin_sec 12`
3. **自动检测画面运动峰值**：`--auto_bullet_count 5`——脚本扫一遍视频，每帧算与上一帧的灰度差均值（"运动能量"），找 top N 个局部峰值

默认是均匀分布 12 个，跨整个视频。

#### 6.3.3 内部流程伪代码

```
for each frame in input:
    1. 应用基础特效（可选）：辉光、拖尾、锐化、色散
    2. 如果当前帧是子弹时刻：
         冻帧 freeze_frames 次（默认 28），每次画面做不同程度的轨道旋转/缩放
         接下来 slow_frames 帧进入慢动作模式
    3. 否则：
         如果还在慢动作模式：每帧 slow_repeat 次重复，可插值
         否则：直接写一帧
```

虚拟相机的轨道是个椭圆（`orbit_frame()`）：

```python
tx = orbit_radius_px * cos(2π·t)
ty = orbit_radius_px * 0.42 * sin(2π·t)
rot = orbit_rot_deg * sin(2π·t)
scale = 1 + (orbit_zoom - 1) * (...)
```

`smoothstep01` 用于让运动加速/减速平滑（避免冻帧首末尾的视觉跳变）。

#### 6.3.4 关键参数

| 参数 | 含义 | 默认 |
|---|---|---|
| `--bullet_times` | 手动子弹时刻（逗号分隔秒数） | 空 |
| `--uniform_bullet_count` | 均匀分布子弹时刻数 | 12 |
| `--auto_bullet_count` | 自动峰值检测数 | 0 |
| `--freeze_frames` | 冻帧时长 | 28 |
| `--orbit_radius_px` | 虚拟相机绕动半径 | 24 |
| `--orbit_rot_deg` | 旋转角度 | 8 |
| `--orbit_zoom` | 缩放倍率 | 1.08 |
| `--slow_frames` | 慢动作持续帧数 | 40 |
| `--slow_repeat` | 慢动作时每帧重复次数 | 6 |
| `--slow_interp` | 慢动作时是否做帧插值 | 是 |

---

## 7. 关键概念图解

### 7.1 球场 4 角点 → Homography → 俯视坐标

```
画面里的梯形                标准球场坐标系（俯视）
                              (0, 0) ─────── (6.1, 0)
   TL ──── TR                    │              │
    \      /                     │              │
     \    /     ──→ Homography ─→│              │
      \  /                       │              │
   BL ──── BR                    │              │
                              (0, 13.4)─── (6.1, 13.4)
```

`cv2.findHomography(src, dst)` 算出 3×3 矩阵 H，之后任何画面像素 (px, py) 都能映射到米制坐标 (mx, my)：

```python
[mx]       [px]
[my]   =   [py]   归一化后乘 H
[ 1]       [ 1]
```

### 7.2 足点估计

```
YOLO bbox：           pose keypoints 优先：

   ┌─────┐               头 ●
   │  ●  │               肩 ●  ●
   │ /│\ │               肘 ●  ●
   │  │  │               腕 ●  ●
   │ / \ │               髋 ●  ●
   │     │               膝 ●  ●
   └──●──┘  ← bbox 底中点  踝 ●  ●  ← 取左右踝平均
```

bbox 底中点离真实站立点高度有偏差（人不是均匀分布的方块），脚踝关键点更精确。两个脚踝置信度都 < 0.22 时退到 bbox 底中点。

### 7.3 跳变阈值的几何含义

```
帧 t-1                帧 t                   判断
脚点 A                脚点 B                  

A ●━━━ 0.3 m ━━━ ● B    距离 = 0.3 m         dt=0.048s
                         max_jump = 0.43 m   ✓ 计入跑动距离
                         inst_speed = 6.3 m/s

A ●━━━ 4.0 m ━━━━━━━━ ● B  距离 = 4.0 m       
                         max_jump = 0.43 m   ✗ 判定 ID 串了
                         不计入，inst_speed=0
```

### 7.4 ByteTrack 的关键作用

```
帧 30：A 在左前场 (id=1)，B 在右后场 (id=2)
帧 31：A 跳起来挥拍，YOLO 框宽变了 → 没有 ByteTrack 时可能给新 ID=3
                                   有 ByteTrack 时认出"这还是 id=1"
帧 32：A 落地，B 在网前 → ByteTrack 维持 id=1, id=2 不变
帧 33：A 暂时被 B 完全遮挡漏检
帧 35：A 重新可见 → ByteTrack 通过运动预测把 id=1 接回去
```

如果没有 ByteTrack，`assign_players` 拿到一堆没有时间一致性的"裸"检测框，根本没法画连续轨迹。

---

## 8. 常见问题排查

### 8.1 球检测率 0%

症状：`short_ball.csv` 里 Visibility 全是 0。

诊断：跑 `python3 scripts/_diag_tracknet.py`，看 heatmap 最大值。如果 < 0.5，说明阈值默认值过严。

修复：确保命令前加 `TRACKNET_VIS_THRESH=0.15` 环境变量，且 `predict.py:35` 已改成读环境变量。

### 8.2 球员速度极端高（>15 m/s）

症状：左侧面板显示"回合最高速度 24 m/s"。

原因：`MotionStats.update()` 里跳变阈值过松，让 ID 串变的瞬时位移被记入。

修复：检查 `overlay_player_analytics.py:602` 应该是

```python
max_jump_m = 8.0 * dt + 0.05
if distance <= max_jump_m:
```

而不是原来的 `if distance <= 1.2`。

### 8.3 中文变方块

原因：脚本默认字体路径是 Windows 的 `C:\Windows\Fonts\msyh.ttc` 等，Mac 上不存在。

修复：`overlay_player_analytics.py:18-27` 的 `FONT_CANDIDATES` 已加入 macOS 字体路径（PingFang.ttc 等）。如果还报错，自查 `/System/Library/Fonts/PingFang.ttc` 是否存在。

### 8.4 球场 quad 画歪了 / 球员位置算不对

原因：`--court_points` 4 个点的顺序错了，或位置不准。

修复：跑 `python3 scripts/_select_court.py short.mp4` 重新点。**顺序必须 TL → TR → BR → BL**。点完先用 `--draw_court_polygon` 加进 overlay 命令，看视频里绿色四边形是否贴合球场。

### 8.5 `ModuleNotFoundError: pycocotools` / `parse` / `lap`

原因：requirements 文件没列全。

修复：

```bash
python3 -m pip install --user --index-url https://pypi.org/simple pycocotools parse lap
```

`lap` 是 ByteTrack 内部的"线性指派"求解器，必装。

### 8.6 `Apple MPS known Pose bug` 警告

原因：ultralytics 在 macOS Metal GPU 上跑 pose 模型偶尔精度异常。

处理：可以忽略（实测影响微小）。如果发现关键点抖动严重，把 `--device mps` 改成 `--device cpu`，速度变慢但更稳。

### 8.7 `FileNotFoundError: short_tracknetv3.mp4`

原因：单独跑 Step 1 时没改名。

修复：

```bash
cp ~/yumaoqiu_repro/tracknet_v3_result_regen/short.mp4 \
   ~/yumaoqiu_repro/tracknet_v3_result_regen/short_tracknetv3.mp4
```

或者用 `run_all_mac.sh` 一键跑（它会自动改名）。

---

## 9. 这次会话踩过的所有坑（变更清单）

按时间顺序，每条都说明了"为什么改"和"改了哪一行"。

1. **字体回退** — `overlay_player_analytics.py:18-27`
   - 加入 `/System/Library/Fonts/PingFang.ttc` 等 macOS 字体路径
   - 不影响 Windows 行为（Windows 字体在前优先匹配）

2. **统计面板从 7 项砍到 4 项** — `overlay_player_analytics.py:678-680, 692-702`
   - 改动 `block_h` 公式从 `24 + 7 * line_h` 到 `28 + 4 * line_h`
   - `panel_h` 下限 410 → 310
   - rows 从 7 行精简到 4 行（current_speed / rally_distance / rally_max_speed / total_distance）
   - `header_h` 从 78 → 96 防止 FPS 行和球员标题重叠

3. **跳变阈值动态化** — `overlay_player_analytics.py:600-604`
   - 原 `if distance <= 1.2:` 改成 `max_jump_m = 8.0 * dt + 0.05; if distance <= max_jump_m:`
   - 修复"最高速度 24 m/s"问题

4. **TrackNet 二值化阈值参数化** — `predict.py:35`
   - 原 `y_pred = y_pred > 0.5` 改成 `thresh = float(os.environ.get("TRACKNET_VIS_THRESH", "0.2")); y_pred = y_pred > thresh`
   - 跑命令时设 `TRACKNET_VIS_THRESH=0.15`，球检测从 0% 提升到 94.9%

5. **新增工具脚本**
   - `scripts/_select_court.py`：交互式标球场角点
   - `scripts/_diag_tracknet.py`：诊断 TrackNet heatmap 输出强度
   - `scripts/overlay/_render_panel_preview.py`：单独渲染面板预览

---

## 10. 后续工作路线图（AI / Codex 可直接接手版）

> 本节按 **P0 → P3** 排序，每个任务都带：背景、目标、步骤、涉及文件路径与行号、验证方法、已知陷阱。
> Codex / Cursor / Claude 等 AI agent 可以从任意一个 Task 起步，不需要外部上下文。
>
> **接手前必读**：§6.1.4（TrackNet 阈值）、§6.2.7（跳变阈值）、§9（已变更清单）。
> 这两个坑没看会重踩，已变更清单告诉你哪些代码已经动过。

---

### Task P0.1 — 把已修参数写进 `run_all_mac.sh` 默认值

**优先级**：P0
**预估工时**：15 分钟
**依赖**：无

**背景**
当前 `run_all_mac.sh` 默认值还是 Windows 时代的，跑 Mac 上视频要每次手动传 `--court-points` 和 `TRACKNET_VIS_THRESH=0.15`。新人接手第一次跑会大概率忘掉，球检出率回到 0%。

**目标**
跑 `./run_all_mac.sh --input-video short.mp4` 就能跑通整条 pipeline，输出可看视频。无需任何额外环境变量或参数。

**步骤**

1. 读 `run_all_mac.sh`：

   ```bash
   cat /Users/yuchenxu/Desktop/pipeline_repro_bundle/run_all_mac.sh
   ```

2. 在文件顶部 `COURT_POINTS=...` 那一行附近，增加：

   ```bash
   # NEW: TrackNet binary threshold (must be lowered, see HANDOVER.md §6.1.4)
   TRACKNET_VIS_THRESH="${TRACKNET_VIS_THRESH:-0.15}"
   export TRACKNET_VIS_THRESH
   ```

3. 把 `COURT_POINTS` 默认值改成本视频实测的：

   ```bash
   COURT_POINTS="352,342,628,343,944,527,52,532"
   ```

4. 在 `STEP 1` TrackNet 命令前确认 `TRACKNET_VIS_THRESH` 已 export（上一步加的 export 行已覆盖）。

5. 在 `STEP 1` 后增加自动改名（避免 §8.7 的 `FileNotFoundError`）：

   ```bash
   if [ -f "${TRACKNET_OUT_DIR}/${VIDEO_STEM}.mp4" ] && [ ! -f "${TRACKNET_OUT_DIR}/${VIDEO_STEM}_tracknetv3.mp4" ]; then
     cp "${TRACKNET_OUT_DIR}/${VIDEO_STEM}.mp4" "${TRACKNET_OUT_DIR}/${VIDEO_STEM}_tracknetv3.mp4"
   fi
   ```

   注意 `run_all_mac.sh` 现版本已包含 `cp -f` 那一行（第 107 行），看一下是否还需要补。

**涉及文件**
- `run_all_mac.sh`（仓库根目录）

**验证**
```bash
rm -rf ~/yumaoqiu_repro
cd ~/Desktop/pipeline_repro_bundle
./run_all_mac.sh --input-video short.mp4 --yolo-device mps
```
- 整个流程一次跑通，无需手动传任何参数
- 输出 `~/yumaoqiu_repro/end1_fix_swap2_precision_full_fx_regen.mp4` 存在
- 球 csv 检测率 ≥ 90%（用 §4.3 的 awk 命令验证）

**已知陷阱**
- bash `${VAR:-default}` 写法允许外部覆盖。如果要测原阈值是否还坏，临时 `TRACKNET_VIS_THRESH=0.5 ./run_all_mac.sh ...`
- `export` 必须在调用 python 之前，否则 sub-process 收不到

---

### Task P0.2 — 清理调试临时文件

**优先级**：P0
**预估工时**：10 分钟
**依赖**：无

**背景**
我们这次会话生成了一堆临时调试文件，已经完成使命，应该清理掉避免污染仓库。

**目标**
仓库里只保留代码、权重、原视频、文档。

**步骤**

1. 列出要删的：

   ```bash
   cd ~/Desktop/pipeline_repro_bundle
   ls -la court_check.png court_check_v2.png court_check_v2_big.png \
          court_grid.png court_guess.png court_hint.png \
          first_frame.png \
          frame_30.png frame_300.png frame_600.png \
          frame_v2_300.png frame_v2_600.png \
          tn_30.png tn_300.png tn_500.png \
          scripts/overlay/_check_panel.png scripts/overlay/_check_panel_v2.png \
          scripts/overlay/_check_court.png scripts/overlay/_check_minicourt.png \
          scripts/overlay/_check_mini_v2.png \
          scripts/overlay/panel_preview.png 2>/dev/null
   ```

2. 删除（确认上面 ls 列出的是临时文件后再跑）：

   ```bash
   rm -f court_check*.png court_grid.png court_guess.png court_hint.png \
         first_frame.png frame_*.png tn_*.png \
         scripts/overlay/_check_*.png scripts/overlay/panel_preview.png
   ```

3. 工具脚本搬到 `scripts/tools/`：

   ```bash
   mkdir -p scripts/tools
   git mv scripts/_select_court.py scripts/tools/select_court.py 2>/dev/null \
     || mv scripts/_select_court.py scripts/tools/select_court.py
   git mv scripts/_diag_tracknet.py scripts/tools/diag_tracknet.py 2>/dev/null \
     || mv scripts/_diag_tracknet.py scripts/tools/diag_tracknet.py
   git mv scripts/overlay/_render_panel_preview.py scripts/tools/render_panel_preview.py 2>/dev/null \
     || mv scripts/overlay/_render_panel_preview.py scripts/tools/render_panel_preview.py
   ```

4. 创建 `scripts/tools/README.md`：

   ```markdown
   # 调试工具

   - `select_court.py <video>` — 交互式标球场 4 角点
   - `diag_tracknet.py` — 诊断 TrackNet heatmap 输出强度（用来定阈值）
   - `render_panel_preview.py` — 单独渲染左侧统计面板预览（不跑模型）
   ```

5. `HANDOVER.md` 里所有提到 `scripts/_select_court.py` 的地方批量替换成 `scripts/tools/select_court.py`：

   ```bash
   sed -i '' 's|scripts/_select_court\.py|scripts/tools/select_court.py|g' HANDOVER.md
   sed -i '' 's|scripts/_diag_tracknet\.py|scripts/tools/diag_tracknet.py|g' HANDOVER.md
   ```

**验证**
```bash
ls scripts/tools/   # 应有 select_court.py, diag_tracknet.py, render_panel_preview.py, README.md
ls *.png 2>/dev/null  # 应为空
python3 scripts/tools/select_court.py short.mp4   # 仍能正常工作
```

**已知陷阱**
- `git mv` 在没初始化 git 的目录会报错。这个仓库不是 git 仓库（看 §0），用普通 `mv`
- `select_court.py` 内部没有引用其他相对路径，可以自由移动

---

### Task P0.3 — 把全长视频也跑一遍（CPU 版）

**优先级**：P0
**预估工时**：3-4 小时（基本是挂机等）
**依赖**：P0.1

**背景**
目前只验证了 30 秒短视频。10 分 35 秒全长视频值得跑一次"基线版本"，作为后续优化的对照基准。

**目标**
得到全长 10:35 视频的最终成品，存档作为 baseline。

**步骤**

1. 准备目录：
   ```bash
   mkdir -p ~/yumaoqiu_repro_full
   ```

2. 后台挂机跑：
   ```bash
   cd ~/Desktop/pipeline_repro_bundle
   nohup ./run_all_mac.sh \
     --input-video b13b2c0b078c64ca95063c958e2fbfd9.mp4 \
     --work-root ~/yumaoqiu_repro_full \
     --yolo-device mps \
     > /tmp/full_pipeline.log 2>&1 &
   echo $! > /tmp/full_pipeline.pid
   ```

3. 监控进度：
   ```bash
   tail -f /tmp/full_pipeline.log
   ```
   关键节点：
   - TrackNet "Median image generated" → 开始模型推理
   - "1it [00:11, ...]" 出现 → 单步耗时 ≈ 11-12 秒
   - 13344 帧 / 16 batch / 1 sliding_step ≈ 850 个 iter，预计 ~3 小时
   - 进 `[STEP 2/3]` 后 ≈ 30 分钟出 overlay
   - `[STEP 3/3]` 5-10 分钟出 FX

4. 跑完归档：
   ```bash
   mkdir -p ~/yumaoqiu_repro_full/archive
   cp ~/yumaoqiu_repro_full/end1_fix_swap2_precision_full_fx_regen.mp4 \
      ~/yumaoqiu_repro_full/archive/baseline_$(date +%Y%m%d).mp4
   ```

**验证**
- `~/yumaoqiu_repro_full/end1_fix_swap2_precision_full_fx_regen.mp4` 存在且可播放
- 文件大小 200 MB ± 50 MB（取决于 FX 配置）
- 抽查 5 个不同时间点（0:30 / 2:00 / 5:00 / 8:00 / 10:00），左侧面板有合理数字、Mini court 有两人轨迹 + 球轨迹

**已知陷阱**
- Mac 进入睡眠会暂停 nohup。用 `caffeinate -dimsu nohup ./run_all_mac.sh ...` 防止
- TrackNet 用 ~3 GB 内存，确认其他大型程序关掉
- 生成文件名是 `<video_stem>` + 后缀，注意不要和短视频产物撞名

---

### Task P1.1 — TrackNet 在 macOS MPS 上加速

**优先级**：P1
**预估工时**：2-4 小时
**依赖**：无（但 P0.3 baseline 用来做加速比对照）

**背景**
当前 `predict.py` 的 device 参数 choices 是 `['auto', 'cpu', 'cuda']`，没有 `mps`。auto 在 Mac 上 fallback 到 cpu。M4 Pro 的 GPU（Metal Performance Shaders）能给 PyTorch 至少 5× 加速。

**目标**
TrackNet 全长视频推理时间从 ~3 小时压到 ~30 分钟。

**步骤**

1. 先验证 MPS 可用：
   ```bash
   python3 -c "import torch; print('MPS:', torch.backends.mps.is_available(), torch.backends.mps.is_built())"
   ```
   两个都 True 才能继续。

2. 修改 `scripts/tracknet_runtime/predict.py`：

   找到当前 device 解析代码（~第 96-101 行附近）：

   ```python
   if args.device == 'auto':
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   else:
       device = torch.device(args.device)
   ```

   改成：

   ```python
   if args.device == 'auto':
       if torch.cuda.is_available():
           device = torch.device('cuda')
       elif torch.backends.mps.is_available():
           device = torch.device('mps')
       else:
           device = torch.device('cpu')
   else:
       device = torch.device(args.device)
   ```

3. 修改 argparse choices（~第 84 行）：
   ```python
   parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'], ...)
   ```

4. 在模型加载附近（`tracknet_ckpt = torch.load(...)`）之前加：
   ```python
   # Some ops aren't implemented on MPS. Fallback to CPU when needed.
   import os
   os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
   ```

5. 跑短视频对比：
   ```bash
   time TRACKNET_VIS_THRESH=0.15 python3 scripts/tracknet_runtime/predict.py \
     --video_file short.mp4 --tracknet_file weights/TrackNet_best.pt \
     --save_dir /tmp/mps_test --output_video --device auto --large_video --eval_mode nonoverlap
   ```
   对比 device=cpu 的耗时。

**涉及文件**
- `scripts/tracknet_runtime/predict.py`（device 解析、argparse choices）

**验证**
- 30 秒短视频 cpu 跑 ~2 分钟，mps 跑 < 60 秒
- 球检出率不下降（MPS 数值精度可能 ±1%）：跑完检查 csv visible 比例 ≥ 92%
- 不出现 `MPS does not implement xxx` 报错

**已知陷阱**
- MPS 上某些 PyTorch op（特别是老版本 nn.Conv2d 边缘 case）会报"未实现"。`PYTORCH_ENABLE_MPS_FALLBACK=1` 让这种 op 自动回 CPU
- DataLoader `num_workers > 0` 在 MPS 上可能挂死。先保持 `num_workers=0`
- 模型 weight 文件 `TrackNet_best.pt` 的 `weights_only=False` 在新版 PyTorch 安全策略下可能报警告但不影响跑

---

### Task P1.2 — 缓存中间结果（detections.json）

**优先级**：P1
**预估工时**：3-5 小时
**依赖**：无

**背景**
现在改一行面板字号 / 颜色 / 阈值，就要重跑 30 分钟 YOLO。如果模型推理结果能存盘，迭代成本能降到秒级。

**目标**
首次跑生成 `detections.json` + `tracks.json`；二次以上跑 Overlay 直接读这俩文件，跳过 YOLO 推理。

**步骤**

1. 在 `overlay_player_analytics.py` 的 `run()` 函数里加缓存逻辑：

   - 入参加：
     ```python
     parser.add_argument('--cache_dir', type=str, default='', help='Cache YOLO results to/from this dir')
     parser.add_argument('--use_cache', action='store_true', help='Use cached results if available')
     ```

   - 主循环前：
     ```python
     cache_path = os.path.join(args.cache_dir, f'{video_stem}_yolo.json') if args.cache_dir else ''
     if args.use_cache and cache_path and os.path.exists(cache_path):
         print(f'[INFO] loading cached YOLO results from {cache_path}')
         with open(cache_path) as f:
             cached = json.load(f)
     else:
         cached = None
     ```

   - 主循环里检测调用前：
     ```python
     if cached is not None:
         frame_results = cached[str(frame_idx)]
         # restore boxes/ids/keypoints from JSON
     else:
         results = model.track(...)
         # serialize the results to a dict, append to a buffer to dump later
     ```

   - 主循环后，如果 `cached is None and cache_path`，把 buffer 里所有 frame 的检测结果序列化写出。

2. JSON 结构建议：
   ```json
   {
     "0": {
       "boxes": [[x1,y1,x2,y2], ...],
       "ids": [1, 2, ...],
       "keypoints": [[[x,y,conf], ...], ...]
     },
     "1": { ... }
   }
   ```

3. 注意 ByteTrack 的 ID 是 stateful 的——直接喂缓存帧检测可以工作（因为 `assign_players` 自己有历史），但要保证 keypoints 数据完整。

**涉及文件**
- `scripts/overlay/overlay_player_analytics.py`（`run()` 函数 + argparse）

**验证**
- 第一次跑：`--cache_dir /tmp/cache` 不带 `--use_cache`，跑完 `/tmp/cache/short_yolo.json` 存在且 30 MB+
- 第二次跑：加 `--use_cache`，从打印信息看到 `loading cached YOLO results`，整个 overlay 在 30 秒内跑完（vs 原来 1 分钟）
- 输出视频跟原版逐帧像素对比，球员位置完全一致

**已知陷阱**
- 缓存文件可能很大（13344 帧 × 5 KB ≈ 60 MB），用 gzip 压缩可降到 10 MB 以内
- `keypoints` 数组维度要跟 ultralytics 原始格式严格对应，否则 `assign_players` 会拿到坏数据

---

### Task P1.3 — YOLO 跳帧检测 + ByteTrack 补偿

**优先级**：P1
**预估工时**：1-2 小时
**依赖**：建议在 P1.2 完成后做（不然每次试参数都要重跑）

**背景**
当前 `--detect_interval 1` 每帧都过 YOLO。其实球员位置在 1/21 秒内变化不大，跳 2-3 帧检测、中间帧用上次结果配 ByteTrack 内部线性插值，效果几乎没有损失但速度翻倍。

**目标**
Overlay 时间从 1 分钟（30 秒视频）压到 30 秒；全长从 30 分钟压到 15 分钟。

**步骤**

1. 测试不同 `--detect_interval`：

   ```bash
   for n in 1 2 3 4; do
     time python3 scripts/overlay/overlay_player_analytics.py \
       --video_path ~/yumaoqiu_repro/tracknet_v3_result_regen/short_tracknetv3.mp4 \
       --output_path /tmp/test_di${n}.mp4 \
       --ball_csv ~/yumaoqiu_repro/tracknet_v3_result_regen/short_ball.csv \
       --court_points "352,342,628,343,944,527,52,532" \
       --device mps --no_select_court_points --detect_interval $n
   done
   ```

2. 抽样比较输出视频质量（比如帧 100/300/500），看球员位置漂移有多大。预计 `--detect_interval 3` 仍可接受。

3. 如果 ID 跟丢，`assign_players` 里加运动学预测：
   - 用上一帧位置 + 上一帧速度 × dt 估当前位置
   - 找 ByteTrack 输出里离这个估计位置最近的 ID

**验证**
- 主观看视频：跳帧版本和每帧版本对比，球员脚点位置偏差 < 5 像素
- `rally_distance` / `total_distance` 数值变化 < 5%

**已知陷阱**
- TrackNet 输出视频的"球轨迹小圆圈"是直接画在视频帧上的，跟 YOLO 跳帧无关
- `--detect_interval > 1` 时 ByteTrack 默认会内部跑卡尔曼滤波，所以 ID 仍能保持。但是漏检 5 帧以上会断
- `draw_pose_on_skipped` 默认 True，跳帧时仍画姿态——确保这个开关一直开着，不然画面会闪烁

---

### Task P2.1 — 球轨迹补漏 + 卡尔曼平滑

**优先级**：P2
**预估工时**：3-4 小时
**依赖**：P0.3 跑出 baseline（用来对比补漏前后效果）

**背景**
当前球检出率 94.9%，剩 5% 漏检（主要是高速飞行模糊或被网柱遮挡）。Mini court 上球轨迹是断断续续的孤立点。

**目标**
对漏检帧用前后帧位置插值（线性 + 卡尔曼），让球轨迹连续。漏检率从 5% 降到接近 0（但插值的不是真实测量，要打标记）。

**步骤**

1. 写一个独立 post-processor，输入 `short_ball.csv`，输出 `short_ball_smoothed.csv`：

   ```python
   # scripts/tools/smooth_ball_csv.py
   import csv, sys
   from filterpy.kalman import KalmanFilter  # pip install filterpy
   import numpy as np

   def main(in_path, out_path):
       rows = list(csv.DictReader(open(in_path)))
       # build (frame, x, y, vis) array
       ...
       # for each missing run, linear interp X/Y if gap < 8 frames
       # for >8 frames gaps, leave 0
       # apply KalmanFilter on the whole sequence to smooth jitters
       # write back with extra column "Source": "model" / "interp" / "kalman"
   ```

2. 修改 `overlay_player_analytics.py` 的 `load_ball_dict` 让它读多列 CSV，给 interp 出来的球点用半透明色画。

3. 主面板上球的描述加一行 "ball coverage: 99% (94% real + 5% interp)"。

**涉及文件**
- 新增 `scripts/tools/smooth_ball_csv.py`
- 修改 `scripts/overlay/overlay_player_analytics.py:61-77`（load_ball_dict）

**验证**
- 跑后 CSV visible 行 ≥ 99%
- Mini court 上球轨迹明显连贯
- 真实检测的球点和插值的球点颜色/透明度有区分

**已知陷阱**
- 跨 8 帧以上的"漏检空洞"通常是球真的离开画面（飞出框/落地前），强行插值会画在网线那里很丑——只对 ≤ 8 帧的空洞插
- 卡尔曼滤波的过程噪声 Q 不能太小，否则球高速变向时跟不上

---

### Task P2.2 — 击球点检测 + 自动回合分割

**优先级**：P2
**预估工时**：4-6 小时
**依赖**：P2.1（要球轨迹连续才能可靠检测拐点）

**背景**
当前 `rally_idx` 是基于"连续多帧检测不到球员"重置的，特别粗。真实回合切换是"一方击球→球落地未被回击"。如果能找到每次击球点，就能：
- 准确分回合
- 统计每回合的击球次数
- 知道哪一拍球员失误

**目标**
画面里出现"击球瞬间"高亮（红圈闪烁 0.3 秒）；面板加"本回合击球数"列；rally 自动分段。

**步骤**

1. 击球检测算法（基于球轨迹）：

   ```python
   # 在球速度向量做 sliding window 算 dot product
   # 每帧 v_t = ball_pos[t] - ball_pos[t-1]
   # 当 dot(v_t, v_{t-1}) < 0 且模长 > 阈值 → 可能击球
   # 进一步过滤：球需要靠近某个球员（< 1.5 米）才算
   ```

2. 数据结构：维护 `hits = []`，每个元素 `(frame_idx, hitter='near'/'far', ball_pos_m)`

3. Rally 切分：
   - 击球后球离开画面或停留 > 1.5 秒 → rally 结束
   - 下次首个击球 → rally 开始

4. 面板增加击球数：
   ```python
   rows = [
       f"当前速度: {stats.current_speed:.2f} m/s",
       f"回合距离: {stats.rally_distance:.2f} m",
       f"回合最高: {stats.rally_max_speed:.2f} m/s",
       f"击球次数: {stats.rally_hits}",   # NEW
   ]
   ```
   把"总距离"挪到第二个 block 的尾部，或者加 `--show_total_distance` 开关。

**涉及文件**
- `scripts/overlay/overlay_player_analytics.py`（新增 `detect_hits()`、修改 `MotionStats`、修改 `draw_stats_panel`）

**验证**
- 跑后 30 秒短视频里每个明显的回合都有 3-8 个击球点高亮
- 没有"幽灵击球"（每回合击球数和肉眼计数差距 < 20%）
- rally_idx 平均每 5-10 秒切一次（合理回合长度）

**已知陷阱**
- 网前小球的球速向量变化幅度小，容易漏。要按"局部极值"检测而不是"反向"
- 跨场地飞行时球离两个球员都远，要等球离得最近的那一帧才算击球
- 短视频里数据少，调参容易过拟合到一个回合，建议在 P0.3 全长视频上调

---

### Task P2.3 — Mini Court 拖尾 + 击球点标记

**优先级**：P2
**预估工时**：1-2 小时
**依赖**：P2.1（轨迹连续）+ P2.2（击球点）

**背景**
Mini court 当前画的是孤立小圆点。专业一点的可视化应该有：
- 移动轨迹是连续线段（深→浅渐变）
- 击球点是闪烁圆环
- 球的飞行轨迹用更亮的颜色 + 拖尾长度更长

**目标**
Mini court 视觉表达接近 broadcast-grade。

**步骤**

1. 修改 `draw_mini_court()` 函数（`overlay_player_analytics.py:711` 附近）：

   ```python
   # 球员历史点用 polylines 画线段，alpha 从 0.2 到 1.0 渐变
   for i in range(1, len(near_map_hist)):
       alpha = i / len(near_map_hist)
       color = blend(base_color, alpha)
       cv2.line(frame, near_map_hist[i-1], near_map_hist[i], color, 1, cv2.LINE_AA)
   ```

2. 击球点用 `cv2.circle` 画 3 层环（外大内小），半径随时间衰减：

   ```python
   for hit in recent_hits:
       age = frame_idx - hit.frame
       if age > 10:
           continue
       radius = int(6 + age)
       alpha = 1 - age / 10
       cv2.circle(frame, hit.map_px, radius, (0, 255, 255), 2, cv2.LINE_AA)
   ```

**涉及文件**
- `scripts/overlay/overlay_player_analytics.py:711-770`（draw_mini_court）

**验证**
- Mini court 上球员有清晰的深浅渐变轨迹线
- 击球时刻有黄色闪烁圆环
- 球轨迹比球员轨迹更亮 + 更长

**已知陷阱**
- alpha blending 用 `cv2.addWeighted` 比手动 `int(... * alpha)` 平滑得多
- 太多 cv2.circle 调用会拖慢 overlay 速度。批量绘制或预先生成 mask

---

### Task P3.1 — Web UI（Gradio）

**优先级**：P3
**预估工时**：1-2 天
**依赖**：P0.1（一键脚本）+ P1.2（缓存）

**背景**
当前所有操作都在终端。教练 / 运动员要想用得上，必须有图形界面：上传视频 → 点角点 → 等结果。

**目标**
本地起一个 Gradio 服务，浏览器打开能完成全流程。

**步骤**

1. 装依赖：
   ```bash
   pip install --user gradio
   ```

2. 写 `scripts/web_ui.py`：
   ```python
   import gradio as gr
   import subprocess

   def process(video_file, court_points_str, vis_thresh):
       # call run_all_mac.sh as subprocess, return final mp4 path
       ...

   def click_corners(image, evt: gr.SelectData):
       # accumulate 4 clicks, draw quad on image, return updated image
       ...

   with gr.Blocks() as demo:
       inp_video = gr.Video()
       first_frame = gr.Image(interactive=True)
       court_state = gr.State([])
       inp_thresh = gr.Slider(0.1, 0.5, 0.15, label="TrackNet vis threshold")
       out_video = gr.Video()
       btn = gr.Button("Run")
       ...

   demo.launch(server_port=7860)
   ```

3. 关键 UX 决策：
   - 上传视频后先抽第一帧让用户标 4 个点（gr.Image 的 `select` 事件）
   - 跑 pipeline 时显示 Step 1/2/3 进度
   - 跑完后视频内嵌播放器 + 下载按钮

**涉及文件**
- 新增 `scripts/web_ui.py`

**验证**
- `python3 scripts/web_ui.py` 启动后浏览器打开 http://localhost:7860
- 上传 short.mp4 → 标 4 角点 → 点 Run → 等若干分钟 → 看到带统计的视频
- 关掉浏览器再重启，已上传的视频能恢复（如果实现了 history）

**已知陷阱**
- Gradio 默认上传文件大小限制 100 MB，全长视频要改 `gr.Video(max_size=...)`
- subprocess.run 长时间任务要 stream stdout 到 UI，不然用户看到一片空白
- 第一帧标点用 gr.Image 的 selectable 模式，每次点击 callback 接收 `evt.index`（像素坐标）

---

### Task P3.2 — 多机位适配

**优先级**：P3
**预估工时**：1-2 天

**背景**
当前所有参数都假设主流"后场视角"机位。换前场低视角、侧场视角，球场 quad 形状变化大，部分参数失效。

**目标**
支持至少 3 种常见机位：后场标准、前场低位、侧场。

**步骤**

1. 调研 3 种机位下：
   - 球场 4 角点典型像素位置范围
   - `--court_length_m`、`--court_width_m` 是否要切换（侧场可能是 13.4 沿 x 方向）
   - `enforce_half_court` 的上下半场判定是否还成立（侧场没有"上下"，是"左右"）

2. 添加 `--camera_mode` 参数：`back_standard / front_low / side`

3. 不同模式下不同的 `--top_inner_ratio`、`--side_band_px`、`--far_half_expand` 默认值

**涉及文件**
- `scripts/overlay/overlay_player_analytics.py`（参数解析 + assign_players + enforce_half_court）

**验证**
- 至少 3 个不同机位的样本视频跑通
- 每种机位的两个球员都被检测且 ID 稳定

**已知陷阱**
- 侧场视角下"上下半场"概念失效，需要重写 `assign_players` 用左右分类
- 前场低位 perspective 极端，4 角点投影到俯视图时数值不稳定，可能要加宽 `--court_length_m` 容差

---

### Task P3.3 — 数据导出

**优先级**：P3
**预估工时**：4-6 小时

**背景**
当前所有数据都画在视频上，看完即焚。教练分析需要原始数字（CSV / JSON）。

**目标**
Pipeline 跑完同时输出一份 `analytics.json`，含每帧球员位置、速度、累计距离、击球事件、回合分段。

**步骤**

1. 在 `overlay_player_analytics.py` 的 `run()` 末尾添加：
   ```python
   if args.export_json:
       export = {
           "video": args.video_path,
           "fps": fps,
           "court_quad": court_quad.tolist(),
           "rallies": [...],
           "frames": [
               {"i": i, "near": {...}, "far": {...}, "ball": {...}}
               for i in range(frame_idx)
           ],
       }
       with open(args.export_json, 'w') as f:
           json.dump(export, f)
   ```

2. 加一个画热力图的 post-processor：
   ```bash
   python3 scripts/tools/heatmap.py analytics.json --output heatmap.png
   ```
   用 `matplotlib.pyplot.hist2d` 画球员主要活动区域。

3. 加一个生成 PDF 报告的 post-processor（可选，用 matplotlib + reportlab）。

**涉及文件**
- `scripts/overlay/overlay_player_analytics.py`（新增 export 逻辑）
- 新增 `scripts/tools/heatmap.py`
- 新增 `scripts/tools/report_pdf.py`（可选）

**验证**
- 输出的 JSON 用 `python3 -m json.tool` 可解析
- 热力图 PNG 上能清晰看到两个球员的活动区域

**已知陷阱**
- JSON 用浮点存球场坐标，注意精度。`{:.3f}` 已足够
- 全长视频的 JSON 可能 30-50 MB，最好默认压缩成 `.json.gz`

---

## 11. 任务依赖图

```
                    P0.1 修 sh 默认值
                         │
                         ▼
              P0.2 清理临时文件 ─── (独立)
                         │
                         ▼
                P0.3 全长 baseline
                  │           │
                  ▼           ▼
     P1.1 MPS 加速      P1.2 缓存 JSON
                              │
                              ▼
                       P1.3 跳帧检测
                              │
                              ▼
                    P2.1 球轨迹补漏
                              │
                              ▼
                    P2.2 击球点 + rally
                              │
                              ▼
                P2.3 Mini court 视觉增强
                              │
                              ▼
                  P3.1 Web UI ── P3.3 数据导出
                              │
                              ▼
                       P3.2 多机位适配
```

最优执行顺序：P0.1 → P0.2 → P0.3 → P1.1 → P1.2 → P1.3 → P2.1 → P2.2 → P2.3 → P3.x

---

---

## 11. 参数速查（最容易问的问题）

**Q: 跑别的视频要改什么？**
A: 三个东西：(1) `--video_file` 路径；(2) 重新标 `--court_points`；(3) 如果机位完全不同，可能要调 `--court_length_m`（全场 13.4，半场 6.7）。

**Q: 全长视频要多久？**
A: 当前 CPU 配置下，10 分 35 秒视频约 3.5 小时。瓶颈在 TrackNet。优化后（P1.1 完成）可降到 30-40 分钟。

**Q: 视频分辨率会影响检测精度吗？**
A: 会。TrackNet 输入会 resize 到 512×288，原视频分辨率越高，球占的相对像素越多，越容易检出。当前 960×544 是临界，如果原始视频是 480p 以下，球会模糊到几乎不可识别。

**Q: 球员检测漏掉了某个球员怎么办？**
A: 调低 `--conf_thres`（默认 0.18，可降到 0.12）。代价是可能误检观众。也可以放大 `--detect_roi_pad_top` 让检测区域包到画面更上方。

**Q: 想只分析一个球员？**
A: 改 `assign_players` 把另一边返回 `None`，或者干脆用 `--top_inner_ratio` 大点把上半场过滤掉。

**Q: 想用不同特效？**
A: 改 `video_fx_bullet_time.py` 的命令行参数。完全不需要改 Step 1/2。

---

## 12. 联系信息和约定

- 配置和路径默认假设 `~/yumaoqiu_repro/` 为输出根目录
- 仓库根目录是 `~/Desktop/pipeline_repro_bundle/`
- 所有 `--court_points` 字符串都按 TL→TR→BR→BL 顺序，逗号分隔，共 8 个数字
- TrackNet 阈值通过环境变量 `TRACKNET_VIS_THRESH` 控制，不传命令行参数（避免改 argparse）

接手前如果只看一个东西，看 §6.1.4（TrackNet 阈值）和 §6.2.7（跳变阈值）。这两个坑是踩了才知道、不写下来下次会重踩的。
