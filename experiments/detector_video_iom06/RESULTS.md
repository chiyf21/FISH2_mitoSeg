# 检测器＋SAM2 Video完整GT修正版结果报告

最终核验时间：2026-11-01T08:21:38+08:00。服务器：`cyf@166.111.72.113`。

检测器新增10轮及SAM2 Video修正版10轮均完整完成，limited_run=false。三卡Video开始2026-11-01 03:47，结束08:01，约4小时15分钟；3440次梯度更新、13760次训练视频采样。预检不计入正式训练。

## 检测器

原完整30轮DFL＋RFF＋jitter0.1的best(epoch27) → 全部原训练数据＋18k排除式负样本，GPU5/6/7仅检测损失新增10轮。使用第3轮best。

| 权重 | 原完整验证检测loss | 新负样本验证检测loss | 固定200份负样本出现检测 |
|---|---:|---:|---:|
| 原epoch27 best | 3.565413 | 1.072109 | 4/200 |
| 微调epoch3 best | 3.453959 | 1.097474 | 2/200 |
| 微调epoch10 last | 3.571285 | 1.992226 | 13/200 |

原验证检测loss下降3.13%，但负样本loss没有统一改善，200份伪负样本探针不构成可靠误检率显著改善证据。未计算完整AP/召回率。原联合loss6.818不是同一指标。

冻结检测器：`/path/to/project_data/JaneliaEM_experiments/detector_video_iom06_20261031/detector_frozen.pt`。其SHA256与选定best一致。

## IoM修正与数据

IoM=seed二维实际检测框与该层完整源GT实例的交集像素数／完整GT像素数，阈值仍0.6。分母含188检测输入及3倍crop以外的GT，使用同一完整精度仿射增强网格。未用GT扩大检测框。六例独立完整源切片回归核验通过。

候选范围为此前已入池的全部实际检测seed，复用冻结检测结果，不声称重新枚举此前已淘汰的检测。

| 数据 | 原候选视频 | 修正后保留 | 剔除 | 源实例 | 每轮采样 |
|---|---:|---:|---:|---:|---:|
| train | 341415 | 176613 | 164802 | 344 | 1376 |
| val | 91009 | 44140 | 46869 | 113 | 固定452 |

原train/val划分保留。train每实例每轮随机4seed，轮间更换；val每实例固定4seed。没有把pool中全部候选作为每轮样本。

## SAM2 Video训练配置

官方SAM2.1 Hiera Base+重新初始化，GPU5/6/7三进程，全局batch4，AdamW LR1e-5、weight decay1e-4、clip1。冻结image/prompt encoder和检测器；训练mask decoder、memory attention/encoder及object pointer投影。

单个真实检测框提示，框中心3倍最长边固定XYcrop，同一原GT实例完整Z双向从seed传播；GT只用于loss。冻结encoder批量8帧，decoder conv_s0/conv_s1保留梯度；8帧截断反传，CPU预取。

L=20×Focal(alpha0.25,gamma2)+Dice+IoU评分MSE+目标存在BCE；先帧平均再视频平均。IoU评分头和存在评分头已在实际checkpoint中验证更新。

## 逐轮结果

| epoch | train loss | val loss | train mask IoU | val mask IoU | val Dice loss |
|---|---:|---:|---:|---:|---:|
| 1 | 4.612524 | 1.537153 | 0.610441 | 0.652133 | 0.325435 |
| 2 | 4.721328 | 2.497707 | 0.654215 | 0.659464 | 0.317445 |
| 3 | 1.755878 | 1.781221 | 0.660773 | 0.685339 | 0.281581 |
| 4 | 4.480173 | 1.456855 | 0.667330 | 0.670076 | 0.300866 |
| 5 | 2.350194 | 2.212829 | 0.670627 | 0.680360 | 0.294716 |
| 6 | 2.774394 | 1.369159 | 0.675773 | 0.690181 | 0.283621 |
| 7 | 2.065745 | 2.344524 | 0.683248 | 0.681803 | 0.274241 |
| 8 | 2.401668 | 1.274361 | 0.686753 | 0.693930 | 0.288165 |
| 9 | 2.529996 | 2.322232 | 0.688270 | 0.699112 | 0.279103 |
| 10 | 3.642768 | 2.358880 | 0.688839 | 0.689274 | 0.285471 |

best按验证总loss选择：第8轮，val loss1.274361，val mask IoU0.693930。最后第10轮：val loss2.358880，val mask IoU0.689274。最高验证IoU在第9轮0.699112，与最小总loss的选择口径不同。

报告mask IoU为每帧二值mask IoU先按视频平均，再按视频平均；GT和预测均空时该帧IoU为1。它不是全数据合并体素的global IoU。验证是固定452seed样本，不等于全部44140候选验证池；没有官方未微调SAM在同一固定验证集的完整基线，不能把epoch1到best的提升当作相对官方模型的净提升。

## 检查点与交付路径

- Detector best：`/path/to/project_data/JaneliaEM_experiments/detector_video_iom06_20261031/detector_detonly_ft10/best_det_sam.pt`。
- SAM best：`/path/to/project_data/JaneliaEM_experiments/detector_video_iom06_20261031/video_fast_fullgt_ft10/best.pt`。
- SAM last：`/path/to/project_data/JaneliaEM_experiments/detector_video_iom06_20261031/video_fast_fullgt_ft10/last.pt`。
- 训练配置：`/path/to/project_data/JaneliaEM_experiments/detector_video_iom06_20261031/video_fast_fullgt_ft10/config.json`。
- 训练代码：`/path/to/project_data/JaneliaEM_experiments/detector_video_iom06_20261031/train_video_fast.py`。
- 修正候选：`/path/to/project_data/JaneliaEM_experiments/detector_video_iom06_20261031/filtered_fullgt`。
- 审计：`/path/to/project_data/JaneliaEM_experiments/detector_video_iom06_20261031/final_checkpoint_audit.json`。
- loss图：`/path/to/project_data/JaneliaEM_experiments/detector_video_iom06_20261031/dashboard/loss_curves.png`。

网页：http://127.0.0.1:18071/training_pipeline/ 。第10次更新早期面板：video_tracking_fullgt/；最终best面板：video_tracking_fullgt_best/（完成标记存在后可用）。

旧video_ft10和video_fast_ft10使用裁断GT筛选，已停止并保留，不能作为本修正版合格结果。
