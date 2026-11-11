#!/usr/bin/env python3
"""Export the repaired v2 flyEM volume using the established s1500 crop pipeline."""
from __future__ import annotations

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import export_flyem2500_npz_ms1500_3000x10 as E  # noqa: E402

FLYEM_V2 = Path("/path/to/project_data/flyEM/2500x2500x250_v2")
E.DATASET = "fish2_2500_v2"
E.EM_PATH = FLYEM_V2 / "raw.ome_v2_repaired_prev_z.tif"
E.LABELS = [
    (E.MITO_ID, FLYEM_V2 / "annotation/mito_annotation.zarr"),
    (E.LYSO_ID, FLYEM_V2 / "annotation/lyso_annotation.zarr"),
]
E.OUT_DIR = Path(
    "/path/to/project_data/JaneliaEM_npz/npz_crops_flyem2500_s1500x2_v2"
)
E.VIZ_DIR = Path("/path/to/JaneliaEM/outputs/logs/flyem2500_s1500x2_v2")
E.SCALES = [dict(name="s1500", ds=1)]
E.N_AUG_TYPE1 = 35
E.N_AUG_TYPE2 = 17

if __name__ == "__main__":
    E.main()
    supervision_path = E.OUT_DIR / "class_supervision.json"
    supervision = json.loads(supervision_path.read_text())
    supervision["comment"] = (
        "fish2_2500_v2 uses the repaired v2 volume with existing mito and lyso instance IDs"
    )
    supervision_path.write_text(json.dumps(supervision, indent=2))

    summary_path = E.OUT_DIR / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["norm"] = (
        "per-volume clip(p0.5,p99.8)->[0,1], then per-crop z-score; "
        "s1500 native 8 nm/px, 10 x 188 x 188 (300 nm z-depth)"
    )
    summary_path.write_text(json.dumps(summary, indent=2))
