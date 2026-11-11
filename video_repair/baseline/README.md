# 线粒体 Video 自动修复 v1.0

这是已确认可用的“原实例修复”流程。输入已有分割实例，不重新跑检测器，也不把 GT 当作提示或筛选依据。SAM2权重保持冻结。

## 固定流程

1. 从原mask选面积最大、无超过50原始像素轴对齐直线的帧，向Z两端独立传播。
2. 原mask首尾非空层各做8邻域连通域拆分。每个组件单独判断面积，严格 **>500像素**、且该层不是原图真实Z边界时，向对应端传播。500整不触发，多组件不累加面积。
3. 每个prompt都是其seed mask的 **tight bbox，零外扩**。坐标为原始像素XYXY半开区间。最大帧用整层mask；端层用对应组件。
4. 每路只用自己的一个条件帧，时序记忆互不共享。crop独立，初始边长=max(100,3×prompt最长边)。逐帧检查预测与crop边缘距离，必要时移动或扩窗并用同一prompt重推；目标边距>100像素，最多32次，失败明确报错。
5. 每路遇到首帧模型IoU评分 **<0.4** 时停止并排除该帧；否则直到源图真实边界。不是连续两帧，不是固定±10层。评分来自SAM的IoU质量头，不是object score。
6. 原mask与所有保留的Video mask逐像素取并集。
7. 并集按 **3D 26邻域**组装。保留与原最大无直线参考帧mask重合最多的主体；并列按原3D mask重合数、体积、较小组件编号排序。删掉其他不连通部分。不额外加1000体素/单层过滤条件。

`config.json`是这版的冻结规则清单，算法参数固定在`core/`中；不要只改JSON就认为推理规则已改变。后续改规则应另立版本并重新核验。

无合格最大帧时仍尝试端层组件；无可用传播时保留原结果并做连通性过滤。核mask从Video预测中扣除；原mask在并集步骤不重新抹除核区域，保持本次实验行为。原图全黑区域沿Z按邻层插值，越界padding不当作全黑层。详见冻结代码和`config.json`。

## 文件与运行环境

- `mito_repair.py`：统一入口。
- `core/`：冻结的裁窗、评分停止、prompt、并集和3D连通性代码。
- `assets/models/`：最终SAM2 Video best及官方Base+初始化权重。
- `assets/sam2/`：已验证SAM2源码及其附带依赖。
- `ASSETS.json`：依赖文件及权重SHA256。
- `tested_environment.json`：实测运行环境版本。Linux＋NVIDIA CUDA。服务器现成Python为`/home/cyf/miniconda3/envs/cellmap_organelle/bin/python`。

服务器完整包已经配好assets。纯代码压缩包不含大模型，可在原服务器执行：

```bash
python install_assets.py --source-experiment /path/to/sam_video_hardcases_20261102
```

该命令只复制/硬链接已有权重和源码，不联网下载、不修改源文件。模型文件作为只读资产使用。跨服务器搬运完整包时，请复制整个目录，不仅复制符号链接。

## 输入格式

原图：uint8的ZYX `.npy` 或可内存映射TIFF。细胞核：与原图完全同shape的bool `.npy`，必须由对应原图产生。

实例manifest：JSON列表，`mask_path`相对manifest所在目录解析；mask为bool的ZYX `.npy`。`origin_xyz`是此小mask在原图中的XYZ原点，单位为原始像素。每个index唯一，原点非负，不能超出原图范围。不需要GT。

```json
[
  {"index": 1, "mask_path": "instances/mito_001.npy", "origin_xyz": [100, 200, 20]},
  {"index": 2, "mask_path": "instances/mito_002.npy", "origin_xyz": [400, 300, 15]}
]
```

本工具不负责从标签体拆实例或生成核mask，避免把历史113例的GT匹配逻辑混入生产修复。

## 完整修复

```bash
CUDA_VISIBLE_DEVICES=2 python mito_repair.py run \
  --raw /path/to/raw.ome.tif \
  --nucleus /path/to/nucleus.npy \
  --manifest /path/to/instances.json \
  --output /path/to/new_repair_run \
  --panel
```

默认读取本包assets；可通过`--weights`、`--base-weights`、`--sam2-code`显式指定其他路径。想固定复现实验，应保留本包原权重及`config.json`规则。

逐实例输出：

```text
cases/instance_001/
  existing_mask.npy, existing_origin.json
  action_000/seed_mask.npy, selected_mask.npy, track.json, origin.json
  ...
  union_mask.npy          # 原mask OR 全部路线，连通过滤前
  mask.npy               # 最终保留的3D主体
  mask_origin.json       # 最终ZYX数组对应的XYZ原点
  connectivity_audit.json
  completed.json, finalized.json
results.json, completed.json, input_identity.json
panel/index.html          # --panel时生成；固定视野、Z并集+前后2层
```

原始mask/原图不覆盖。重用同一输出目录可续跑；输入路径/大小/修改时间、代码或模型身份不同会拒绝混用。入口带目录锁，防重复运行。`completed.json`存在且`limited_run=false`才是本轮完成。

可视化是“原图/原mask/修复mask”三栏，不依赖GT。黄色框只在真正提示帧出现；显示各路最大评估score，不把它误称为并集mask的置信度。使用`python -m http.server 18080 --directory /path/to/new_repair_run/panel`浏览；最终科学评估应另提供独立GT。

## 仅重做并集后的连通性过滤

```bash
python mito_repair.py finalize \
  --source-run /path/to/simple113_component500_union_tight_score04_20261103 \
  --output /path/to/new_connected_run
```

这条命令不加载SAM、不占GPU，只消费已完成的原mask＋Video并集。`--only 1 3`只处理指定编号。可用`--expected-run /path/to/reference_connected_run`逐体素核对参考结果。源目录应是已完成的推理输出；缺文件将报错。

## 本次113例基准

113例记录，其中109例有原mask，4例无原mask而跳过。50例存在游离组件，删除3159个组件、761975体素。原mask平均IoU=0.57601953，最终=0.64102409。该均值含完整113条显示记录；不是独立测试集结论，历史数据存在训练重叠。GT仅用于评估和历史对照面板，不用于本包生产推理。

`VALIDATION.json`记录本包对本次113例的逐体素复现，以及独立入口实际SAM推理复现结果。封装后不应只凭脚本可运行判断复现成功。
