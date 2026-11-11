#!/usr/bin/env bash
set -euo pipefail
cd /path/to/JaneliaEM
/home/cyf/miniconda3/envs/cellmap_organelle/bin/python organelle_detect/experiments/flyem2500_s1500x2_det_sam_dfl_jitter/exporters/export_flyem2500_v2_npz_s1500x2.py
