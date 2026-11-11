#!/usr/bin/env python3
"""flyEM fish2 2500^2x250 (mito + lyso), 1500 nm scale only, 2x the augmented copies of
export_flyem2500_npz_ms1500_3000x10.py (same rotation/translation/in-bounds logic, split, norm).

  type1  1 base + 35 aug per instance   (was 1 + 17)
  type2  1000 random windows x (1 + 17)  (was 1 + 8)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import export_flyem2500_npz_ms1500_3000x10 as E  # noqa: E402

E.OUT_DIR = Path("/path/to/project_data/JaneliaEM_npz/npz_crops_flyem2500_s1500x2")
E.VIZ_DIR = Path("/path/to/JaneliaEM/outputs/logs/flyem2500_s1500x2")
E.SCALES = [dict(name="s1500", ds=1)]
E.N_AUG_TYPE1 = 35
E.N_AUG_TYPE2 = 17

if __name__ == "__main__":
    E.main()
