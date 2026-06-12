# JetArm 4-Marker V1 实施计划

**状态**: 链路跑通；姿态可靠性验证中  
**最后更新**: 2026-06-11  
**Cursor  handoff**: [docs/jetarm_marker/STATUS.md](../jetarm_marker/STATUS.md)（与本文同步，优先读 STATUS）

## 目标

用 JetArm Orbbec/Gemini **临时代替 Aimooe**，验证：

```text
4 反光 marker -> IR 检测 -> depth -> 针轴/针尖 -> needle_tracking 3D 显示
```

**不做**精度结论、**不做**相机→头模配准（排期在最后）。

---

## 决策（已锁定）

| 项 | 结论 |
|----|------|
| 2D 检测主路线 | **IR**（优于 RGB，已现场确认） |
| 针轴 | **m2 → m1** |
| 针尖 | **m1 沿轴外推 140 mm**（手工测量，待精测） |
| m1 门控 | `\|m2-m1\| / \|m0-m3\| >= 0.55`，否则 `valid=0` |
| 工作目录 | `needle_tracking 无配准`（非 3-27 备份） |

---

## 进度总览

| 阶段 | 状态 | 说明 |
|------|------|------|
| 0 工具链 | ✅ | `detect_ir_markers`, `pose_from_ir_depth`, `replay_pose_csv` |
| 1 模态 | ✅ | IR 主路线已决；关键 bag 含 IR+depth |
| 2–3 IR 检测 | ✅ | static 49/49；move 98/98 |
| 4–7 IR+depth 姿态 | △ | move **69/98**；static **0/49**（m1 视角） |
| 8 3D 回放 | ✅ | 离线 replay，带 marker 可视化 |
| 5 几何 | ⏳ | tip 140 mm 手工；`needle_geometry.json` 仍 placeholder |
| 9 刚体/精度 | ❌ | **4 点 3D 相对关系不稳定 — 当前核心风险** |
| **下一项** | 🔜 | **实时 IR 2D 检测窗口**（见 STATUS） |

---

## 关键 bag

```text
marker_static_rgb_ir_depth_01   # IR 检测 OK；3D 姿态 0/49
marker_move_rgb_ir_depth_01     # IR 69/98 姿态；门控剔除 28 帧
```

旧 `marker_*_clean_01`（无 IR）：RGB 光流对照，非主路径。

---

## 已知问题（不可忽略）

1. m1 ↔ 关节点误识别  
2. m1 半出镜 → 针尖大跳  
3. marker 架 3D 刚体性未达标  
4. depth 在反光球处空洞/背景混入  
5. 无 camera → 头模配准  

---

## 下一步（禁止偏航）

**不要**继续盲录 bag / **不要**直接实时 3D。

**要做**：实时 IR 2D 窗口 → 稳定角度 → 2–3 段高质量 bag → 刚体定量 → 实时 3D → 配准。

详见 [STATUS.md](../jetarm_marker/STATUS.md)。

---

## 工具索引

| 优先级 | 脚本 |
|--------|------|
| ★ | `detect_ir_markers.py`, `pose_from_ir_depth.py`, `replay_pose_csv.py` |
| 辅 | `legacy/bag_probe.py`, `legacy/export_modality_compare.py`, `legacy/compare_modality_report.py` |
| 旧 | `legacy/track_markers.py`, `legacy/pose_from_markers.py`, `legacy/run_pipeline.py` |

命令细节：[tools/jetarm_marker/README.md](../../tools/jetarm_marker/README.md)
