# MacBook Air 详细用法（复刻同款视频效果）

本文对应文件夹：

- `pipeline_repro_bundle/`

目标：从原视频复刻到最终效果视频：

- `end1_fix_swap2_precision_full_fx_regen.mp4`

## 0. 机器建议

- Apple Silicon（M1/M2/M3）优先，建议内存 16GB+。
- 纯 CPU 也能跑，但会非常慢。

## 1. 安装基础工具

```bash
xcode-select --install
```

安装 Homebrew（若未安装）：

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

安装 Python 和 ffmpeg：

```bash
brew install python ffmpeg
```

## 2. 解压与进入目录

把 `pipeline_repro_bundle.zip` 拷到 Mac 后解压，进入目录：

```bash
cd /path/to/pipeline_repro_bundle
```

## 3. 创建虚拟环境并安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements_repro.txt
```

如果你想尽量使用 Apple GPU（MPS）：

```bash
python -c "import torch; print('mps_available=', torch.backends.mps.is_available())"
```

## 4. 运行一键脚本（推荐）

先给脚本可执行权限：

```bash
chmod +x run_all_mac.sh
```

### 4.1 固定球场点（最快）

```bash
./run_all_mac.sh \
  --input-video "/Users/you/Videos/866ba79f9b46ce0d9b8b1d55eb82832c.mp4" \
  --work-root "/Users/you/yumaoqiu_repro"
```

### 4.2 手动点球场四角（更稳）

```bash
./run_all_mac.sh \
  --input-video "/Users/you/Videos/866ba79f9b46ce0d9b8b1d55eb82832c.mp4" \
  --work-root "/Users/you/yumaoqiu_repro" \
  --manual-court
```

点击顺序：

- TL -> TR -> BR -> BL

### 4.3 指定 YOLO 设备（可选）

Apple Silicon 可尝试 MPS：

```bash
./run_all_mac.sh \
  --input-video "/Users/you/Videos/866ba79f9b46ce0d9b8b1d55eb82832c.mp4" \
  --work-root "/Users/you/yumaoqiu_repro" \
  --yolo-device mps
```

若报错可改为：

```bash
--yolo-device cpu
```

## 5. 输出文件位置

运行完成后在 `--work-root` 下得到：

- `tracknet_v3_result_regen/<video_stem>_tracknetv3.mp4`
- `tracknet_v3_result_regen/<video_stem>_ball.csv`
- `end1_fix_swap2_precision_full_regen.mp4`
- `end1_fix_swap2_precision_full_fx_regen.mp4`

其中最后一个就是复刻目标效果文件。

## 6. 常见问题

1. `ModuleNotFoundError`
- 确认在项目目录执行，且已激活 `.venv` 并安装依赖。

2. OpenCV 弹窗不能点球场
- 用固定点方式运行（不加 `--manual-court`）。

3. 速度太慢
- 先用短视频测试：
  ```bash
  ffmpeg -y -ss 00:00:00 -t 00:00:30 -i "/Users/you/Videos/866...mp4" "/Users/you/Videos/866_short.mp4"
  ```
  再把 `--input-video` 改成 `866_short.mp4`。

4. `bytetrack.yaml` 找不到
- 先执行：
  ```bash
  python -c "import ultralytics; print(ultralytics.__file__)"
  ```
  然后升级 ultralytics：
  ```bash
  pip install --upgrade ultralytics
  ```

## 7. 与原机结果的一致性说明

- 最终特效阶段（`video_fx_bullet_time.py`）在同输入下是稳定的。
- 叠加阶段受环境与球场点选择影响，像素级可能有差异，但视觉风格与流程一致。

