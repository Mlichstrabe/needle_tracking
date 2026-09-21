# IMU + STP-23L 穿刺训练探针 CAD

## T10 一体实心平底针体（2026-09-16，当前打印入口）

按用户要求取消分段与定位销，主体由宽5 mm增至6 mm，外露200 mm加12 mm插入端。针尖、根部和插入端共用一张连续平底，圆头延伸至底面；取消T9的架空脚垫。打印版增加与底面齐平的0.20 mm厚连续可剪薄边，并预旋转45°。首选 `output/cad_laser100m_T10_onepiece/T10_PRINT_ONE_PIECE_6mm_WITH_BRIM.stl`（同名STEP亦可），完整包 `output/T10_ONE_PIECE_6mm_PRINT_PACKAGE.zip`。

源文件 `laser100m_T10_onepiece.py`，设计说明 `T10_design.md`。检查底面轴向连续性、26个水平截面、单实体STL、STEP回读、原外壳配合以及180 mm平台内的斜放包络；验证边界写在交付说明和validation.json中。实际切片和成功试印仍需确认。

## T1 贴合薄壳（2026-09-11）

用户确认极简固定架外覆一层薄壳后生成的独立版本。主体 **58 × 44 × 47 mm**，
含前突针杆座为 **61 × 44 × 47 mm**；3 个结构件和 3 个参考件组成装配。

源文件 `cad/laser100m_T1.py`，参数 `cad/laser100m_T1_parameters.json`。
运行 `uv run --with cadquery --with matplotlib --with trimesh python cad/laser100m_T1.py`。
输出 `output/cad_laser100m_T1_thin_shell/`，主装配为
`laser100m_HWT906P_T1_assembly.step`；`T1_shell_only.step` 仅含 3 个结构件。

顶部隆起适配近似 IMU 本体，前端为共用光窗；仍需实物试配。模块前端距针根仅
7 mm，150 mm 推进行程末端理论距离为 57 mm，近端可靠性未验证。详见该目录交付说明及验证报告。

## A2 紧凑外壳（2026-09-10）

按用户要求缩小 A 方案手柄：由 100 × 54 × 60 mm 改为 **86 × 50 × 52 mm**，
外包络体积减少约 31%。针杆仍外露 200 mm，激光前端仍位于针根后 30 mm。
IMU 平台降低 6 mm，激光模块上移 0.5 mm，新的安装位置需随实物装配记录。

源文件 `cad/laser100m_A2.py`，参数 `cad/laser100m_A2_parameters.json`；
运行 `uv run --with cadquery --with matplotlib --with trimesh python cad/laser100m_A2.py`。
装配 STEP、对比预览和验证报告位于 `output/cad_laser100m_A2_compact/`。
旧 A1 输出保留。A2 仍为试配版，尤其需要核对缩短尾部后的插头和线束空间。

## 100 m 裸模块 A 方案（2026-09-10）

新版 A 方案使用 HWT906P、100 m 裸测距模块和 Ø4 mm 实体杆（外露 200 mm）。
它是独立的布局与试配版本，输出在 `output/cad_laser100m_A/`：

- `laser100m_HWT906P_A_assembly.step`：带名称和颜色的装配。
- `laser100m_HWT906P_A_multibody.step`：多实体备用格式。
- `README_交付说明.md`、`validation.json`：验证结果与模型边界。

生成命令：`uv run --with cadquery --with matplotlib --with trimesh python cad/laser100m_A.py`。
参数为 `cad/laser100m_A_parameters.json`。带 `REF_` 的传感器是近似参考件，
实体杆也不是打印件；不得将装配整体直接打印。先核对 IMU 朝向、PCB 边缘、
光窗与插头避让并进行局部试配。下述内容仍描述旧 STP-23L 方案。

本目录保存参数化机械模型的源文件。模型仅用于钝头仿体、凝胶和人头模型演示，不用于人体或临床操作。

## 生成

在仓库根目录运行：

```powershell
uv run --with cadquery python cad/integrated_probe.py
```

V1 输出保留在 `output/cad/`；照片参考针体的 V2 输出写入
`output/cad_v2_photo_needle/`。SolidWorks 可直接打开其中的 STEP 文件；STL 文件用于切片。

确认采用对称圆珠钝头的 V2-B 输出写入 `output/cad_v2b_rounded_tip/`。
正式针体保持轴对称；A1 mini 打印版针尖下方带 0.48 mm 可掰除支撑薄片。

## V2 针体

- 总外露长度仍为 200 mm，前约 170 mm 为 Ø6 mm 直杆。
- 靠手柄的 30 mm 为叶片式平滑加强段，最大宽度 14 mm。
- 根部加强段与原 18 mm 卡口一体成形，卡口和外壳接口未改。
- 针尖仍为钝头，仅用于仿体、凝胶和人头模型演示。

## 首次打印顺序

1. 先打印 `fit_coupon.stl`，分别试验 0.20、0.25、0.30 mm 单侧间隙。
2. 将选中的间隙写回 `probe_parameters.json` 的 `snap.nominal_side_clearance`。
3. 重新运行生成脚本，再打印主骨架、上盖、线缆压板和针杆。
4. 空壳验收通过后再装入 HWT906P 和 STP-23L。

## 坐标

- 探针轴为 X 轴；针尖在负 X 方向。
- HWT906P 的 +Y 朝针尖，USB-C 朝手柄尾部。
- 厂家 STP-23L STEP 的 +Z 光轴被旋转到装配坐标的 -X 方向。

## T2 渐细针体（2026-09-12）

`laser100m_T2.py` / `laser100m_T2_parameters.json`：参考用户 acupuncter_needle.stl，保留外露 200 mm 和 Ø4 × 12 mm 插入段，后 100 mm 圆柱、前 100 mm 渐细至 Ø0.8 圆钝端。结构沿用 T1；输出 `output/cad_laser100m_T2_tapered_needle`。针体为重新构建的旋转实体，不是 STL 精确复制。已验证 STEP 重导入、装配干涉和 STL 水密性；强度未实测。

## T5 PLA 友好加宽内腔（2026-09-15）

`laser100m_T5.py` / `laser100m_T5_parameters.json`：外壳宽度 46 mm，内腔左右各向外放宽 1 mm，壁厚保持约 1.6 mm。外盖与中间载架取消弹性卡扣，改用刚性导向柱、封闭定位袋和胶粘定位面；中间载架可直接落座，必要时点胶。针体沿用 T3 的 Ø5 mm 主体、根部加固和圆珠笔头，外露 200 mm。输出在 `output/cad_laser100m_T5_widened_clearance/`，完整文件包为 `output/laser100m_HWT906P_T5_STEP_bundle.zip`。已验证 STEP 重导入、结构干涉和 STL 水密性；PLA 强度、装配保持力和实物净空仍需试配。

## T6 PLA 三段式针体（2026-09-15）

`laser100m_T6.py` / `laser100m_T6_parameters.json`：针体拆为根部段、中段和针尖段，分别水平打印后用两根 Ø3 mm 定位销和胶粘接合；外露仍为 200 mm，主体 Ø5 mm，针尖为圆球。T5 外壳和 PLA 刚性定位方案同时保留。输出在 `output/cad_laser100m_T6_segmented_needle/`，整套针体可导入 `T6_needle_print_set.step`，完整文件包为 `output/laser100m_HWT906P_T6_STEP_bundle.zip`。每个打印件均通过 STEP 重导入和 STL 水密性检查；孔径、强度、胶接寿命和实际打印方向仍需试配。

## T7 平底 3/4 截面针体（2026-09-15）

`laser100m_T7.py` / `laser100m_T7_parameters.json`：在 T6 的三段式针体基础上，将根部段、中段和针尖段统一切到 `axis_z - 1.0 mm` 的纵向平面，形成约 3/4 圆截面的连续平底；标称 Ø5 mm 轴段剩余高度约 3.5 mm，圆珠笔头保留圆润过渡。平面朝下打印以抑制针体滚动和滑移，定位销仍为圆柱形以匹配原圆孔。T5 外壳、宽度 46 mm、左右各 1 mm 内腔余量和胶粘定位方案不变。输出在 `output/cad_laser100m_T7_flat_bottom_needle/`，整套针体为 `T7_needle_print_set.step`，完整文件包为 `output/laser100m_HWT906P_T7_flat_bottom_STEP_bundle.zip`。已通过 STEP 重导入、零干涉、STL 水密性和端面预览检查；真实打印仍需按打印机参数试配。

## T8 分离排版打印版（2026-09-15）

`laser100m_T8_print_layout.py` / `laser100m_T8_print_layout_parameters.json`：针对切片软件把装配坐标中的三段针体排成约 212 mm 长条的问题，保留 T7 几何并新增独立打印排版。`T8_needle_print_layout.step` 将根部段、中段、针尖段分成三行，平面统一落在 `z=0`，定位销单独放置；另提供五个已经归零到打印床的 `*_flat_print.stl`。切片时使用排版 STEP 或这些 STL，不要直接加载总装 STEP。输出在 `output/cad_laser100m_T8_separated_print_layout/`，完整文件包为 `output/laser100m_HWT906P_T8_separated_print_layout_bundle.zip`。已验证排版件互不干涉、STEP 重导入和 STL 水密性；brim、床面附着和 PLA 收缩仍需实物试印。

## T9 平底连接件与针尖防滑脚（2026-09-15）

`laser100m_T9_connector_tip_adhesion.py` / `laser100m_T9_connector_tip_adhesion_parameters.json`：针对 T8 实际打印中定位销圆柱接触线过窄、针尖段仍可能脱床的问题，将两根 Ø3 mm 定位销改为带宽平底的 3/4-like D 截面，并在针尖段平底下加入三个可掰除的防滑脚。打印后拆除防滑脚、清理底面，再用定位销和胶粘接合。切片入口为 `output/cad_laser100m_T9_connector_tip_adhesion/T9_needle_print_layout.step` 或该目录中的 `T9_*flat_print.stl`；总装 STEP 仅用于 CAD 查看。完整文件包为 `output/laser100m_HWT906P_T9_connector_tip_adhesion_bundle.zip`。已验证 5 个排版实体的落床、互不干涉、STEP 重导入和 STL 水密性；防滑脚拆除、brim 和胶接强度仍需实物试配。
