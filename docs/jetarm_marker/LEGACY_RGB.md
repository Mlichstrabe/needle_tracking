# JetArm Marker — 旧 RGB / 模态探路路径

**非当前主叙事。** IR + depth 路线见 [tools/jetarm_marker/README.md](../../tools/jetarm_marker/README.md)。

脚本位于 `tools/jetarm_marker/legacy/`。

## 阶段 1：bag 探测与模态对比

```bash
python tools/jetarm_marker/legacy/bag_probe.py data/jetarm_marker/bags/marker_static_clean_01

python tools/jetarm_marker/legacy/export_modality_compare.py ^
  data/jetarm_marker/bags/marker_static_clean_01 ^
  --samples 12

python tools/jetarm_marker/legacy/compare_modality_report.py ^
  data/jetarm_marker/bags/marker_static_clean_01
```

输出：

```text
data/jetarm_marker/exports/<bag_name>/probe_summary.json
data/jetarm_marker/exports/<bag_name>/modality_report.json
data/jetarm_marker/exports/<bag_name>/frames/*_compare.png
```

结论（已决策）：**IR 对反光球明显优于 RGB**，主路径已切到 IR。

注意：`marker_static_clean_01` / `marker_move_clean_01` 无 `/depth_cam/ir/image_raw`，只能 RGB + depth。

## 阶段 2：RGB marker 初始化

GUI 手动点选：

```bash
python tools/jetarm_marker/legacy/init_markers.py ^
  data/jetarm_marker/exports/marker_static_clean_01/frames/frame_004_idx0173_rgb.png ^
  --bag marker_static_clean_01
```

自动 bootstrap（流水线用）：

```bash
python tools/jetarm_marker/legacy/auto_init_markers.py ^
  --bag data/jetarm_marker/bags/marker_static_clean_01 ^
  --frame-index 238 --bag-name marker_static_clean_01
```

已有 init 文件：

```text
data/jetarm_marker/inits/marker_static_clean_01_init.json
data/jetarm_marker/inits/marker_move_clean_01_init.json
```

## 阶段 3：RGB 光流 2D 跟踪

```bash
python tools/jetarm_marker/legacy/track_markers.py ^
  --bag data/jetarm_marker/bags/marker_static_clean_01 ^
  --init data/jetarm_marker/inits/marker_static_clean_01_init.json ^
  --start 173 --end 260

python tools/jetarm_marker/legacy/track_markers.py ^
  --bag data/jetarm_marker/bags/marker_move_clean_01 ^
  --init data/jetarm_marker/inits/marker_move_clean_01_init.json ^
  --start 0 --end 504 ^
  --jump-px 70 --rigid-tol 0.55 ^
  -o data/jetarm_marker/tracking/marker_move_clean_01_track2d_relaxed.csv
```

## 阶段 4：RGB + depth 三维估计

```bash
python tools/jetarm_marker/legacy/pose_from_markers.py ^
  --bag data/jetarm_marker/bags/marker_static_clean_01 ^
  --track data/jetarm_marker/tracking/marker_static_clean_01_track2d.csv ^
  --geometry data/jetarm_marker/geometry/needle_geometry.json ^
  --scene data/jetarm_marker/geometry/scene_transform.json
```

`needle_geometry.json` 为 placeholder（`measured=false`），仅作链路对照。

## 一键旧流水线

```bash
python tools/jetarm_marker/legacy/run_pipeline.py --bag marker_static_clean_01 --force
```

完成后可用主路径回放器：

```bash
python tools/jetarm_marker/replay_pose_csv.py data/jetarm_marker/tracking/marker_static_clean_01_pose.csv
```

## 历史验证结果（对照）

| bag | 路径 | 备注 |
|-----|------|------|
| marker_static_clean_01 | RGB LK | 88/88 pose，tip jitter ~0.53 mm，无 IR |
| marker_move_clean_01 | RGB LK relaxed | 129/505 pose，遮挡多 |
