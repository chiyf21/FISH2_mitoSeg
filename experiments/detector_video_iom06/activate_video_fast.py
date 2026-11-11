from common import *
import shutil
config=json.loads((ROOT/'pipeline_config.json').read_text())
config.update(video_output='video_fast_ft10',video_entry='launch_video_fast.py',video_gpus=[5,6,7],video_effective_batch=4,video_seeds_per_instance_per_epoch=4,video_val_fixed_seeds_per_instance=4,video_recovery_every_updates=100,video_first_preview_update=10,video_encoder_batch=8,video_epochs=10,video_lr=1e-5)
save(ROOT/'pipeline_config.json',config)
append(ROOT/'events.jsonl',{'time':stamp(),'event':'user_authorized_fast_video_restart','stopped_old_pids':[1510450,1593827],'retained_output':'video_ft10','new_output':'video_fast_ft10','gpus':[5,6,7],'data':'4 seeds per instance per epoch; fixed4 validation; full Z and original split retained'})
save(ROOT/'state.json',{'stage':'video_fast_preflight','status':'running','message':'快速版三卡梯度/显存预检；每实例每轮4个seed，global batch4。','time':stamp()})
shutil.copy2(ROOT/'README.md',ROOT/'dashboard/README.md')
shutil.copy2(ROOT/'dashboard.html',ROOT/'dashboard/index.html')
comparison=json.loads((ROOT/'detector_comparison.json').read_text());rows=comparison['results'];assert len(rows)==3
table='| Checkpoint | Original complete validation DET loss | Hard-negative validation DET loss | Negative200 any detection |\n|---|---:|---:|---:|\n'
for r in rows:table+=f'| {r["name"]}, epoch{r["epoch"]} | {r["old_validation"]["val_loss"]:.6f} | {r["negative_validation"]["val_loss"]:.6f} | {r["negative_detections"]["fraction_with_any_mito_or_lyso_detection"]:.1%} |\n'
improvement=100*(1-rows[1]['old_validation']['val_loss']/rows[0]['old_validation']['val_loss'])
report='# Detector-only fine-tuning evaluation\n\n'+table+f'\nSame evaluation implementation, FP32, old complete validation9582 crops and hardnegative validation3600 crops, rank batch22. Best improves old detector-only loss by {improvement:.2f}% relative to original epoch27. Original joint validation loss6.818 is a different objective and is not compared directly.\n\nBest is epoch3; last epoch10 is worse. Train loss falls2.5807→2.1528 while old validation worsens after epoch3, consistent with overfitting. The negative200 probe is4→2→13 detected crops; not enough to establish a statistically reliable reduction in false positives. Full hardnegative loss is slightly worse at best than original, so do not claim uniform negative suppression. These are pseudo-negative crops, not manually verified precision. No AP/recall benchmark has been computed.\n\nUse frozen epoch3 best for downstream SAM.\n\n'+str(ROOT/'detector_frozen.pt')+'\n'
(ROOT/'DETECTOR_RESULTS.md').write_text(report);(ROOT/'dashboard/DETECTOR_RESULTS.md').write_text(report)
