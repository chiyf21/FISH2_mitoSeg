from common import *
import torch
torch.set_num_threads(3)
assert json.loads((ROOT/'completed.json').read_text())=={**json.loads((ROOT/'completed.json').read_text()),'detector_epochs':10,'video_epochs':10,'limited_run':False}
det_done=json.loads((ROOT/'detector_detonly_ft10/completed.json').read_text());assert det_done['epochs']==10 and not det_done['limited_run']
done=json.loads((ROOT/'video_fast_fullgt_ft10/completed.json').read_text());assert done['epochs']==10 and not done['limited_run'] and done['updates']==3440
pool=json.loads((ROOT/'filtered_fullgt/completed.json').read_text());assert pool['full_run'] and pool['denominator']=='complete_source_gt_slice' and pool['threshold']==.6 and pool['audit_regression_passed']
rows=[json.loads(x) for x in (ROOT/'video_fast_fullgt_ft10/metrics.jsonl').read_text().splitlines()]
assert [r['epoch'] for r in rows]==list(range(1,11)) and all(not r['limited_run'] for r in rows)
checks={}
for name in ('best','last'):
    p=ROOT/f'video_fast_fullgt_ft10/{name}.pt';ck=torch.load(p,map_location='cpu',weights_only=False)
    assert ck['progress']['epoch_complete'] and ck['config']['iom_denominator'].startswith('complete source GT')
    assert ck['config']['data']==str(ROOT/'filtered_fullgt')
    assert all(torch.isfinite(v).all().item() for v in ck['model'].values() if torch.is_tensor(v) and v.is_floating_point())
    checks[name]={'path':str(p),'epoch':ck['epoch'],'updates':ck['updates'],'readable':True,'finite_model':True,'sha256':sha(p),'val':rows[ck['epoch']-1]['val']};del ck
assert checks['last']['epoch']==10 and checks['last']['updates']==3440
assert checks['best']['epoch']==min(rows,key=lambda r:r['val']['loss'])['epoch']
frozen=json.loads((ROOT/'frozen_detector.json').read_text());assert frozen['sha256']==sha(ROOT/'detector_frozen.pt')==sha(ROOT/'detector_detonly_ft10/best_det_sam.pt')
checks['detector']={**frozen,'verified_sha256_unchanged':True}
checks['completed_verified_at']=stamp();save(ROOT/'final_checkpoint_audit.json',checks)
lines=['# 检测器＋SAM2 Video完整GT修正版结果报告','',f'最终核验时间：{stamp()}。服务器：`cyf@166.111.72.113`。','',
'检测器新增10轮及SAM2 Video修正版10轮均完整完成，limited_run=false。三卡Video开始2026-11-01 03:47，结束08:01，约4小时15分钟；3440次梯度更新、13760次训练视频采样。预检不计入正式训练。','',
'## 检测器','',
'原完整30轮DFL＋RFF＋jitter0.1的best(epoch27) → 全部原训练数据＋18k排除式负样本，GPU5/6/7仅检测损失新增10轮。使用第3轮best。','',
'| 权重 | 原完整验证检测loss | 新负样本验证检测loss | 固定200份负样本出现检测 |','|---|---:|---:|---:|',
'| 原epoch27 best | 3.565413 | 1.072109 | 4/200 |','| 微调epoch3 best | 3.453959 | 1.097474 | 2/200 |','| 微调epoch10 last | 3.571285 | 1.992226 | 13/200 |','',
'原验证检测loss下降3.13%，但负样本loss没有统一改善，200份伪负样本探针不构成可靠误检率显著改善证据。未计算完整AP/召回率。原联合loss6.818不是同一指标。','',
'冻结检测器：`'+str(ROOT/'detector_frozen.pt')+'`。其SHA256与选定best一致。','',
'## IoM修正与数据','',
'IoM=seed二维实际检测框与该层完整源GT实例的交集像素数／完整GT像素数，阈值仍0.6。分母含188检测输入及3倍crop以外的GT，使用同一完整精度仿射增强网格。未用GT扩大检测框。六例独立完整源切片回归核验通过。','',
'候选范围为此前已入池的全部实际检测seed，复用冻结检测结果，不声称重新枚举此前已淘汰的检测。','',
'| 数据 | 原候选视频 | 修正后保留 | 剔除 | 源实例 | 每轮采样 |','|---|---:|---:|---:|---:|---:|',
'| train | 341415 | 176613 | 164802 | 344 | 1376 |','| val | 91009 | 44140 | 46869 | 113 | 固定452 |','',
'原train/val划分保留。train每实例每轮随机4seed，轮间更换；val每实例固定4seed。没有把pool中全部候选作为每轮样本。','',
'## SAM2 Video训练配置','',
'官方SAM2.1 Hiera Base+重新初始化，GPU5/6/7三进程，全局batch4，AdamW LR1e-5、weight decay1e-4、clip1。冻结image/prompt encoder和检测器；训练mask decoder、memory attention/encoder及object pointer投影。','',
'单个真实检测框提示，框中心3倍最长边固定XYcrop，同一原GT实例完整Z双向从seed传播；GT只用于loss。冻结encoder批量8帧，decoder conv_s0/conv_s1保留梯度；8帧截断反传，CPU预取。','',
'L=20×Focal(alpha0.25,gamma2)+Dice+IoU评分MSE+目标存在BCE；先帧平均再视频平均。IoU评分头和存在评分头已在实际checkpoint中验证更新。','',
'## 逐轮结果','',
'| epoch | train loss | val loss | train mask IoU | val mask IoU | val Dice loss |','|---|---:|---:|---:|---:|---:|']
for r in rows:lines.append(f'| {r["epoch"]} | {r["train"]["loss"]:.6f} | {r["val"]["loss"]:.6f} | {r["train"]["mask_iou"]:.6f} | {r["val"]["mask_iou"]:.6f} | {r["val"]["dice"]:.6f} |')
lines+=['',
'best按验证总loss选择：第8轮，val loss1.274361，val mask IoU0.693930。最后第10轮：val loss2.358880，val mask IoU0.689274。最高验证IoU在第9轮0.699112，与最小总loss的选择口径不同。','',
'报告mask IoU为每帧二值mask IoU先按视频平均，再按视频平均；GT和预测均空时该帧IoU为1。它不是全数据合并体素的global IoU。验证是固定452seed样本，不等于全部44140候选验证池；没有官方未微调SAM在同一固定验证集的完整基线，不能把epoch1到best的提升当作相对官方模型的净提升。','',
'## 检查点与交付路径','',
'- Detector best：`'+str(ROOT/'detector_detonly_ft10/best_det_sam.pt')+'`。',
'- SAM best：`'+checks['best']['path']+'`。',
'- SAM last：`'+checks['last']['path']+'`。',
'- 训练配置：`'+str(ROOT/'video_fast_fullgt_ft10/config.json')+'`。',
'- 训练代码：`'+str(ROOT/'train_video_fast.py')+'`。',
'- 修正候选：`'+str(ROOT/'filtered_fullgt')+'`。',
'- 审计：`'+str(ROOT/'final_checkpoint_audit.json')+'`。',
'- loss图：`'+str(ROOT/'dashboard/loss_curves.png')+'`。','',
'网页：http://127.0.0.1:18071/training_pipeline/ 。第10次更新早期面板：video_tracking_fullgt/；最终best面板：video_tracking_fullgt_best/（完成标记存在后可用）。','',
'旧video_ft10和video_fast_ft10使用裁断GT筛选，已停止并保留，不能作为本修正版合格结果。']
report='\n'.join(lines)+'\n';(ROOT/'RESULTS.md').write_text(report);(ROOT/'dashboard/RESULTS.md').write_text(report)
print(json.dumps(checks,indent=2))
