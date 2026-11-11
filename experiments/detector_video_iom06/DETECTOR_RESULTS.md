# Detector-only fine-tuning evaluation

| Checkpoint | Original complete validation DET loss | Hard-negative validation DET loss | Negative200 any detection |
|---|---:|---:|---:|
| original30_best, epoch27 | 3.565413 | 1.072109 | 2.0% |
| detonly_best, epoch3 | 3.453959 | 1.097474 | 1.0% |
| detonly_last, epoch10 | 3.571285 | 1.992226 | 6.5% |

Same evaluation implementation, FP32, old complete validation9582 crops and hardnegative validation3600 crops, rank batch22. Best improves old detector-only loss by 3.13% relative to original epoch27. Original joint validation loss6.818 is a different objective and is not compared directly.

Best is epoch3; last epoch10 is worse. Train loss falls2.5807→2.1528 while old validation worsens after epoch3, consistent with overfitting. The negative200 probe is4→2→13 detected crops; not enough to establish a statistically reliable reduction in false positives. Full hardnegative loss is slightly worse at best than original, so do not claim uniform negative suppression. These are pseudo-negative crops, not manually verified precision. No AP/recall benchmark has been computed.

Use frozen epoch3 best for downstream SAM.

/path/to/project_data/JaneliaEM_experiments/detector_video_iom06_20261031/detector_frozen.pt
