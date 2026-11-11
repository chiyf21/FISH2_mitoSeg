# flyEM 2500 s1500x2 det_sam experiment

This directory packages the run represented by train_det_sam_flyem2500_s1500x2_v1v2_100ep_eos0p2_retry3_resume_e6.log: 3D det_sam, slice2d boxes, RFF prompts, prompt jitter 0.1, EOS 0.2 and DFL in the shared YOLO detection loss. Run from ~/JaneliaEM in cellmap_organelle. Common model and data-loader modules remain under organelle_detect/models and organelle_detect/data.

## Data processing
The exporters include v1/v2 entry points and their shared crop exporter. They write NPZ crops, split and supervision metadata, and preview images. Edit paths and augmentation counts near each file's top. Setup is s1500 at native 8 nm/px, crop 10 x 188 x 188, 35 type1 and 17 type2 extra augmentations.
Run:
- bash organelle_detect/experiments/flyem2500_s1500x2_det_sam_dfl_jitter/run_export_v1.sh
- bash organelle_detect/experiments/flyem2500_s1500x2_det_sam_dfl_jitter/run_export_v2.sh

## Training and test
Training evaluates held-out split.json test regions each epoch. Run:
- bash organelle_detect/experiments/flyem2500_s1500x2_det_sam_dfl_jitter/run_train.sh
Change GPUS, BS, NPZ1, NPZ2, OUT at the top of run_train.sh. Loss options: --mask-loss tversky|ce (default tversky), --tversky-alpha 0.5, --tversky-beta 0.5. They select matched voxel mask terms, including prompt masks. Class-label CE and Hungarian matching costs are unchanged. --mask-loss ce restores prior sigmoid BCE. At alpha=beta=0.5 Tversky is Dice-like; the separate Dice term remains. Other parameters: --npz-root, --scales, --epochs, --batch-size, --effective-batch, --lr, --num-workers, --eval-every, --prompt-pe, --prompt-jitter, --eos-coef, --bf16, --resume.

Test a checkpoint without training:
PY=/home/cyf/miniconda3/envs/cellmap_organelle/bin/python
EXP=organelle_detect/experiments/flyem2500_s1500x2_det_sam_dfl_jitter
$PY $EXP/train_npz.py --mode det_sam --dim 3d --det-box slice2d --npz-root /path/v1 /path/v2 --resume /path/best_det_sam.pt --eval-only --num-workers 4 --prompt-pe rff --prompt-jitter 0.1 --eos-coef 0.2 --mask-loss tversky

## Inference
infer_f2500v2_det_sam_trial.py runs sliding windows, prompt masks, fragment stitching, edge-window trials, and nucleus exclusion:
PY=/home/cyf/miniconda3/envs/cellmap_organelle/bin/python
EXP=organelle_detect/experiments/flyem2500_s1500x2_det_sam_dfl_jitter
$PY $EXP/infer_f2500v2_det_sam_trial.py --ckpt /path/best_det_sam.pt --tif /path/raw.ome.tif --nuc /path/nucleus.npy --out-dir /path/infer --viz-dir /path/infer_viz --stride-z 3 --stride-xy 94 --gpus 0,1,2,3
Other options: --ckpt-nuc, --tag, --recompute-nuc, --nuc-skip-thr, --roi Z0 Z1 Y0 Y1 X0 X1, --workers-per-gpu, --merge-iou, --merge-cont, --new-free-frac, --order, --merge-region, --frag-min, --contact-thr. See --help. Crop is 10 x 188 x 188; default stride 3 x 94 x 94.

## Visualization
Plot training and test JSONL:
PY=/home/cyf/miniconda3/envs/cellmap_organelle/bin/python
EXP=organelle_detect/experiments/flyem2500_s1500x2_det_sam_dfl_jitter
$PY $EXP/visualization/plot_train_log.py --log outputs/my_run/train_log_det_sam.jsonl --out outputs/my_run/curves.png
Render held-out examples with raw, predictions and GT:
$PY $EXP/visualization/viz_det_sam10.py --npz-root /path/v1 /path/v2 --ckpt /path/best_det_sam.pt --out outputs/my_run/test_examples.png --n-picks 10
viz_val.py and viz_flyem1500x10_val.py are included helpers.
