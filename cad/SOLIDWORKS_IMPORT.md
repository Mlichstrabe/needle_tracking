# SolidWorks 导入与继续编辑

## 推荐打开顺序

1. 打开 `output/cad/integrated_probe_assembly.step`。该文件保存了零件名称和装配层级，适合整体检查。
2. 若当前 SolidWorks 版本未正确恢复层级，改为打开 `integrated_probe_multibody.step`，它包含相同位置的多实体几何。
3. 需要单独修改时，分别打开 `lower_chassis.step`、`upper_cover.step`、`printed_needle.step` 和 `cable_clamp.step`。

## SolidWorks 设置

1. 在“工具 -> 选项 -> 系统选项 -> 导入”中启用 3D Interconnect，再打开 STEP。
2. 打开后运行“工具 -> 评估 -> 输入诊断”，修复结果应为零坏面和零间隙。
3. 对需要长期编辑的零件，右键导入特征并选择“断开链接”，然后另存为 SLDPRT。
4. 若希望识别拉伸、圆角和孔，可尝试 FeatureWorks；参数化真源仍以 `cad/integrated_probe.py` 和 `probe_parameters.json` 为准。

## 不应在 SolidWorks 中直接改动的基准

- STP-23L 光轴必须保持与探针 X 轴平行。
- STP-23L 光学面到针根的轴向后移量为 30 mm。
- IMU 载板的前后方向必须保持 HWT906P 的 +Y 朝针尖、USB-C 朝尾部。
- 若修改模块间隙，先改 `probe_parameters.json` 并重新导出，避免上下壳使用不同数值。

## 打印文件

优先使用带 `print_ready` 的 STL。针杆使用 `printed_needle_a1mini_diagonal.stl`，它已旋转约 45° 并放置在 180 mm × 180 mm 平台范围内。
