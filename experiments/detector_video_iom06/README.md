# 原版30轮模型 → 完整数据＋负样本微调10轮 → Video SAM2

服务器113。目录：`/path/to/project_data/JaneliaEM_experiments/detector_video_iom06_20261031`。

用户最终确认：检测器从原版已完成30轮的DFL＋RFF＋jitter0.1 best开始，新增微调10轮；数据使用原来的全部训练数据＋此前导出的负样本。完成后随机50例看框，再冻结检测器并筛选Video样本。SAM2 Video训练10轮，学习率1e-5。

## 检测器

原版：`/path/to/JaneliaEM/outputs/train_det_sam_flyem2500_s1500x2_slice2d_dfl_rffjit/best_det_sam.pt`，best epoch27，原验证loss6.8181946564。原last及e30均为epoch30；原启动脚本明确slice2d DFL、RFF、prompt jitter0.1。权重审计：`original30_audit.json`。原best SHA256：`34f1b10f1174f64dc3884887e9a571fca5d8fbdcb53f7dd4e7a5946c77be95b3`。

完整原数据目录包含原type1和type2，含已有空目标裁块；并非只选择有目标的“正crop”。保留历史loader和split：原train32430＋新增负样本train14400＝46830；原val9582＋新增负样本val3600＝13182。新增负样本就是此前从v1前80层排除核、mito、lyso和32nm保护带后导出的18000个视图（1000基础family×18），不重新挑样本或替换成新误检框。

2026-10-31用户明确改为仅检测训练，GPU5/6/7三卡DDP。原联合训练`detector_ft10/`已停止，检查点和日志保留，不计入新训练轮数。新输出`detector_detonly_ft10/`，从原epoch27 best重新初始化，AdamW LR1e-4，FP32，新增10轮。仅backbone、neck、det_heads参与前向与更新，分割、prompt及ER分支均不计算且冻结。使用历史检测分类＋框回归＋DFL损失；RFF及jitter属于未启用的prompt分支，此阶段不生效。原batch16采样器每4个batch合并为有效batch64，分给3卡为22/21/21样本，再小批前向并按样本数加权梯度；最后不足64的组使用真实样本数。保留原type/rare采样、全部数据及train/val划分。best只按原完整验证集的检测loss选择。新入口`train_detonly_ddp.py`及`launch_detonly.py`；每轮保存last/best，每5轮编号保存。

之前从5轮续到30轮的临时进程已按用户改动停止，其日志和epoch5备份保留，不把它的训练进度混入此10轮微调。

## 检测框可视化与IoM筛选

微调完整10轮之后，复制选出的best为不可变`detector_frozen.pt`，保存SHA256；随后随机抽50例（28原train＋7原val＋12负样本train＋3负样本val），展示实际输入、逐层检测框和GT，所有10层均保留。未按分数挑图。页面：`dashboard/detector_samples50/`。

冻结检测器在完整合并train/val上检测，仅将train用于后续梯度训练，val单独评估。沿用原逐层NMS、跨层关联、3D NMS0.6、top12、track score>0.3。取保留track的实际二维框，逐框、逐GT mito实例计算该层IoM＝框内GT像素数／该层GT像素数，按像素中心判断框内，≥0.6则保留。不是3D汇总IoM，也不要求每层过阈值。

一个框只匹配一个实例，优先IoM再IoU，避免同一prompt对应不同GT。每个合格二维框都是一个Video起始提示样本。新增空目标负样本仍执行检测，但没有mito GT时不会伪造Video正样本。

以该二维框中心为固定XY crop中心，边长`ceil(3×最长边)`，所有Z层保持同一视野。GT使用该框匹配到的原始3D实例ID在所有Z层上的mask，不混用邻近实例，不用GT框代替实际框。原始XY8nm、Z30nm不变，仅按SAM2标准在网络内部resize XY至1024。超过真实原图边界才填0；不会把NPZ边界当原图边界。

原NPZ增强角度和位移只保存一位小数，因此重建了原seed42的完整精度采样、旋转、平移及shuffle顺序；全部42012条元数据必须一致，GT与原NPZ逐值一致，灰度归一化也在小误差范围内一致，才允许导出候选。此步骤防止扩充视野和GT错位。原始源数据和原导出NPZ均不改写。

## Video SAM2

### 2026-11-01完整GT IoM修正（当前有效版本）

用户明确保留IoM≥0.6，分母必须是seed层完整源GT实例，而不是188×188检测输入内被裁断的mask。审计6例中3348原记1.0但完整GT仅0.416832；5521原记1.0但完整GT0.701392。独立审计见`video_iom_audit.json`和`dashboard/video_iom_audit/`。旧`filtered/`只作为候选检测来源，不可直接用于正式SAM训练。

`refilter_full_gt.py`重新核验旧候选池每一条实际框，使用`full_gt_iom.py`恢复相同增强网格上的整个GT实例切片，统计框内像素/完整切片像素；不把188输入或3倍crop当分母边界，不用GT扩大检测框。保留原检测权重、真实框、crop、split、Z范围和0.6阈值。为保持候选来源一致，仅重筛原已入池候选，不声称重新枚举此前已淘汰的所有检测框。完整导出为`filtered_fullgt/`，检查`full_gt_verified`、完整像素计数、阈值和六例独立实现回归。

旧`video_fast_ft10/`已停止、checkpoint和日志保留，属于使用有缺陷筛选的历史结果。新训练从官方初始SAM2.1权重重新开始10轮，目录`video_fast_fullgt_ft10/`，预检`video_fast_fullgt_preflight/`。仍是5/6/7三卡、global batch4、每源实例每轮4seed、固定验证4seed、完整Z、LR1e-5及原loss/冻结模块。数量以修正后config为准。追踪面板`dashboard/video_tracking_fullgt/`仅选择至少3层的实例，并区分seed完整GT IoM与模型预测IoU。

### 2026-11-01用户授权的快速版（覆盖旧全seed训练）

旧单卡`video_ft10/`已停止并保留日志；它尚无epoch checkpoint，不冒充已保存进度。新输出`video_fast_ft10/`，从官方SAM2.1 Base+重新微调10轮。每个原GT实例每轮随机4个合格seed视频，轮间换seed/view，训练348个源实例、1392条视频/轮；不足4候选时允许重复抽样。验证125源实例固定4seed、共500条，保留原split。完整Z、IoM条件、crop、唯一真实框提示、冻结模块和loss定义不变。因验证采样从全量改为固定平衡子集，快速版验证值与旧全seed验证不作为同口径结果直接比较。

GPU5/6/7三进程同步训练，global batch4，按序列长度分配4条给3卡；各rank累积局部loss/global_batch后做梯度SUM，保证全局四条视频平均，不因三卡把有效batch改为12。LR1e-5，BF16前向/FP32损失，8帧截断反传。冻结image_encoder批量8帧，暂存该chunk特征；Mask Decoder的conv_s0/conv_s1在梯度图内逐帧计算。每rank一CPU预取线程。

第10次更新保存`first_preview.pt`用于6实例追踪预览。每100次更新和整轮验证前原子保存`recovery.pt`（含模型、优化器、group cursor、各rank RNG及累计指标）；每epoch保存last/best，第5/10轮编号保存。恢复选最高完整updates，不把preflight当训练。入口`launch_video_fast.py`/`train_video_fast.py`。`watch_video_preview.py`等待真实first_preview后生成`dashboard/video_tracking_fast/`。

初始化官方SAM2.1 Hiera Base+。冻结image encoder、prompt encoder和检测器；训练mask decoder、memory attention、memory encoder及object-pointer投影（video头的明确范围）。每样本只使用筛选出的单个检测框，不喂入GT mask/额外GT点作为提示。GT仅作为监督，记忆来自模型预测。从起始层向Z两端传播，同一原始实例的完整Z范围，不插Z层、不截成仅10层。

长序列采用8帧截断反传，保留最近32个非条件输出及起始条件记忆；覆盖每个原始Z层，不因序列长而丢层。microbatch1条视频、累积4；AdamW LR1e-5、weight decay1e-4，BF16 autocast、FP32 loss。每帧`20×focal(alpha=.25,gamma=2)＋Dice＋IoU预测MSE＋object presence BCE`，先按视频帧数平均，再按样本平均。使用SAM2Base的可微track step，不能直接拿带inference_mode的视频预测接口反传。

实测预检要求mask decoder、memory attention、memory encoder均有非零梯度，冻结编码器无梯度，loss和梯度范数有限。预检输出标记limited_run，不计作正式10轮。正式每轮保存last/best，每5轮保存编号checkpoint；断线不影响运行，退出后从完整epoch checkpoint恢复。

保持历史crop级train/val划分，也保留其原有的空间/实例重叠风险；不能宣称由此获得严格实例独立的整卷分割精度。Video验证指标仅针对IoM筛选后样本，不是完整检测＋分割召回率。

## 进度、曲线和自动衔接

页面：http://127.0.0.1:18071/training_pipeline/ 。`dashboard/loss_curves.png`每20秒更新；记录原30轮、检测器新增10轮、Video10轮及step级loss。检测器/SAM2每10个optimizer update写step日志，各阶段epoch指标单独保存，可下载JSONL。未开始阶段明确显示Waiting。

`controller.py`依次等待检测器完整10轮、冻结、随机50例可视化、全量IoM筛选、Video最终数据梯度预检、Video完整10轮。阶段之间有完成标记和权重哈希检查，任务退出最多保守重试3次；仅使用113 GPU0–3的剩余显存，不停止他人的任务。错误和资源等待会写`state.json`；不得把进程退出或checkpoint存在当作完成。

主要输出：`detector_ft10/`、`frozen_detector.json`、`filtered/`、`video_ft10/`、`dashboard/`、`events.jsonl`；最终报告`RESULTS.md`，最终完成标记`completed.json`。
