#!/usr/bin/env bash
set -euo pipefail
cd /path/to/JaneliaEM
PY=/home/cyf/miniconda3/envs/cellmap_organelle/bin/python
EXP=organelle_detect/experiments/flyem2500_s1500x2_det_sam_dfl_jitter
NPZ1=/path/to/project_data/JaneliaEM_npz/npz_crops_flyem2500_s1500x2
NPZ2=/path/to/project_data/JaneliaEM_npz/npz_crops_flyem2500_s1500x2_v2
OUT=outputs/train_det_sam_flyem2500_s1500x2_v1v2_100ep_tversky
GPUS=0,2,3,4,5,6
BS=16
if [[ -e "$OUT" ]]; then echo "Refusing to overwrite existing output: $OUT" >&2; exit 2; fi
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
for path in "$NPZ1" "$NPZ2"; do [[ -f "$path/summary.json" && -f "$path/split.json" ]] || { echo "Missing data: $path" >&2; exit 2; }; done
mkdir -p outputs/logs
CUDA_VISIBLE_DEVICES="$GPUS" "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" "$EXP/train_npz.py" \
 --mode det_sam --dim 3d --det-box slice2d --prompt-pe rff --prompt-jitter 0.1 --eos-coef 0.2 --bf16 \
 --npz-root "$NPZ1" "$NPZ2" --batch-size "$BS" --effective-batch 96 --epochs 100 --eval-every 1 \
 --num-workers 4 --mask-loss tversky --tversky-alpha 0.5 --tversky-beta 0.5 --out "$OUT"
