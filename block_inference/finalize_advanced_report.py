import json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;B=P/'validation/advanced_20261104_013608/full'
a=json.loads((B/'results.json').read_text())
rows=[]
for x in a:rows.append(dict(name=x['name'],seconds=x['pipeline_seconds'],video_seconds=x['stage_seconds']['video'],strict_exact=x['comparison']['strict_exact'],comparison=x['comparison'],advanced_stats=x.get('advanced_stats'),gt=x.get('gt_evaluation')))
for name in ['T_streams','R_baseline','R_prompt','R_streams']:
 x=json.loads((B/(name+'_result.json')).read_text());stats=B/name/'video/advanced_execution_stats.json'
 rows.append(dict(name=name,seconds=x['seconds'],video_seconds=next(s['wall_seconds'] for s in x['stage_timings'] if s['stage']=='video'),strict_exact=x['comparison']['strict_exact'],comparison=x['comparison'],advanced_stats=json.loads(stats.read_text()) if stats.exists() else None,gt=x.get('gt')))
r=dict(status='completed',benchmark_directory=str(B),raw_shape_zyx=[32,256,256],gpu=1,hardware='A100',independent_fresh_outputs=True,baseline_final_instances=6,baseline_saved_frames=143,default_advanced_enabled=False,accepted_additional_speedup_percent=0,rows=rows,interpretation=['Tensor batch variants change numerical outputs and are not accepted as exact production optimizations.','B1 stream scheduling preserves all 143 saved masks, origins and scores in the first test; repeat consistency is recorded separately.','Initialization cache hits 33; prefix cache hits 0 in this block. Crop changes invalidate cached history.','Whole-pipeline startup and other-stage timing vary; use paired repeats and Video timings, not a single favorable total.','This smoke is a small block, not a reliable 20k-instance throughput estimate.','Previous safe execution_optimizations and its earlier 15.4% measurement are separate; gains must not be added.'])
(P/'ADVANCED_OPTIMIZATION_REPORT.json').write_text(json.dumps(r,indent=2))
lines=['# 四阶段高级执行优化：真实冒烟报告','', '结论：四项策略已实现并以独立开关测试；保持算法结果的新增加速未获确认，生产默认关闭高级优化。原有安全优化继续保留。','', '## 具体实现','', '1. 重放批量编码：仅预编码已经确定的重放区间，图像张量batch最多4；另测B=1双CUDA stream成组编码。时序解码仍按方向顺序，不读取未来mask。','2. 初始化复用：完整source/model、ROI、crop、seed Z/box、阈值及计算形状一致才命中，返回独立状态字典和只读张量。','3. 前缀缓存：方向、seed、crop、前一历史token和计算形状全部一致才复用mask/score/记忆，LRU预算限制显存；变crop不能复用旧历史。','4. 多track解码：独立track通过GPU broker合批，保留独立记忆；张量batch只合并兼容记忆布局，B=1 streams版并发独立请求。按原检测顺序提交结果，保留同层原有抑制语义。','', '## 计时与一致性','', '同一A100 GPU1、同一256×256×32真实原图，各方案使用新输出，均运行核分割、检测、Video与26CC。共23个seed尝试，基线最终6个3D实例、143份保留帧。','', '| 方案 | 全流程秒 | Video秒 | 逐帧及最终结果完全一致 |','|---|---:|---:|---|']
for x in rows:lines.append(f"| {x['name']} | {x['seconds']:.2f} | {x['video_seconds']:.2f} | {'是' if x['strict_exact'] else '否'} |")
lines+=['', 'B0为当前版；B1仅重放张量batch；B2再加初始化复用；B3再加前缀缓存；B4再加多track张量batch。S仅缓存。T为四项全部开启的B=1 streams实现。R为交替复测，其中R_prompt仅初始化缓存。','', '严格一致性核对覆盖：核mask、检测框、seed决策、prompt/探测记录/停止点、全部保留帧mask/origin/score以及最终标签。基线无高级优化；不是把已完成预测当作缓存测速。','', '第一次streams全流程59.42秒看似优于基线62.68秒，但Video49.48秒反而高于基线47.58秒；不能据此声称加速。大batch造成至少5114个前景体素差异，因此不自动启用。','', '初始化缓存命中33次，相同前缀命中0次。crop移动后几何输入变了，不能用旧crop特征/记忆替代；严格缓存命中有限。多track并行只在同层存在多个新seed时有机会，且GPU调度、等待和额外显存有成本。','', '这次小块实验不支持外推20k实例的可靠耗时。原有15.4%生产优化是以前独立测量，不能和本轮理论收益相加。','', '## 使用与复现','', '`advanced_optimizations`默认关闭。实验入口：`run_advanced_benchmark.py`、`run_stream_benchmark.py`、`run_repeat_benchmark.py`；配置、日志、原始输出和精确差分均在：', '', str(B), '', 'CPU缓存/方向/几何失效、独立状态打包、重放来源映射回归通过；原空间索引与稀疏组装回归通过。详细JSON包含每组原始检查和缓存计数。']
(P/'ADVANCED_OPTIMIZATION_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
v=P/'VALIDATION.json';d=json.loads(v.read_text());d['advanced_optimization_experiment']=dict(report='ADVANCED_OPTIMIZATION_REPORT.json',status='completed',production_default=False,strict_stream_repeat=next(x['strict_exact'] for x in rows if x['name']=='R_streams'),accepted_additional_speedup_percent=0);v.write_text(json.dumps(d,indent=2))
m=P/'RELEASE_MANIFEST.json';files=json.loads(m.read_text())
for name in ['advanced_execution.py','test_advanced_execution.py','ADVANCED_OPTIMIZATION_DESIGN.md','ADVANCED_OPTIMIZATION_REPORT.md','ADVANCED_OPTIMIZATION_REPORT.json','run_advanced_benchmark.py','run_stream_benchmark.py','run_repeat_benchmark.py']:files[name]={}
for name in files:
 p=P/name
 if p.is_file():files[name]=dict(size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
m.write_text(json.dumps(files,indent=2))
print(json.dumps([dict(name=x['name'],seconds=round(x['seconds'],2),video_seconds=round(x['video_seconds'],2),exact=x['strict_exact']) for x in rows]))
