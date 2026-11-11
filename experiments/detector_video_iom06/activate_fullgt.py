from common import *
import shutil
config=json.loads((ROOT/'pipeline_config.json').read_text())
config.update(iom_threshold=.6,iom_denominator='complete source GT instance slice on identical augmented pixel grid',video_data='filtered_fullgt',video_output='video_fast_fullgt_ft10',video_preflight='video_fast_fullgt_preflight',video_preview='dashboard/video_tracking_fullgt',video_initialization='official SAM2.1 Base+ fresh restart after clipped-GT audit',invalid_historical_runs=['video_ft10','video_fast_ft10'])
save(ROOT/'pipeline_config.json',config)
append(ROOT/'events.jsonl',{'time':stamp(),'event':'user_confirmed_full_source_gt_iom06','old_video_stopped':[374304,375068,375071,375110,375111,375112],'candidate_pool':'filtered_fullgt','new_run':'video_fast_fullgt_ft10','unchanged_threshold':.6})
save(ROOT/'state.json',{'stage':'refilter_complete_gt_iom06','status':'running','message':'按该层完整源GT重筛，IoM阈值保留0.6；旧SAM日志/权重已保留。','time':stamp()})
shutil.copy2(ROOT/'README.md',ROOT/'dashboard/README.md')
shutil.copy2(ROOT/'dashboard.html',ROOT/'dashboard/index.html')
old=ROOT/'dashboard/video_tracking_fast/index.html'
warning='<div style="background:#762927;color:white;padding:16px">历史结果：此面板使用旧的裁断GT筛选池。已发现IoM被高估，不能作为完整GT IoM≥0.6的合格示例。<a href="../video_iom_audit/">查看审计</a>；<a href="../video_tracking_fullgt/">修正版面板</a>。</div>'
if old.exists() and warning not in old.read_text():old.write_text(warning+old.read_text())
