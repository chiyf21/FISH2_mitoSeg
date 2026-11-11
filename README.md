# Mitochondria segmentation and block inference

Code-only research repository for mitochondrial detection, prompt-based segmentation, SAM2 Video inference, and 3D assembly and repair. It does not contain EM volumes, annotations, trained weights, run outputs, cached features, or generated dashboards.

## Scope

This repository groups segmentation and inference source snapshots. It is not one installable Python package, and scripts from different experiments are not interchangeable.

| Directory | Purpose |
|---|---|
| `organelle_detect/models/` | 2D/3D detectors, prompt-SAM heads, segmentation heads, losses, and box operations. |
| `organelle_detect/data/` | Dataset, NPZ, CellMap, and segmentation/detection loading code. |
| `organelle_detect/experiments/flyem2500_s1500x2_det_sam_dfl_jitter/` | Original DET-SAM DFL, RFF, and prompt-jitter training, export, and inference entry points. |
| `experiments/detector_video_iom06/` | Later detector-only fine-tuning and SAM2 Video training and audit entry points. |
| `block_inference/` | Large-block orchestration for nuclear inference, detection, Video inference, and instance assembly. |
| `video_repair/` | Experimental targeted repair, crop-shift, and detection-graph utilities. |
| `block_stitch/` | Block-level 3D stitching prototype and tests. |

`SOURCE_MAP.md` records source origins and omissions. The original DET-SAM and later detector plus SAM2 Video workflows remain separate because they use different model and training setups.

## Original DET-SAM workflow

The original experiment uses the shared 3D detector and prompt-segmentation model. Its experiment README documents the prepared NPZ format, DFL detection loss, RFF prompt encoding, box jitter of 0.1, type-1/type-2 data setup, and train/evaluation scripts. The exporter, trainer, and inference entry points are under:

`organelle_detect/experiments/flyem2500_s1500x2_det_sam_dfl_jitter/`

Read that directory's README before running. It contains experiment-specific commands and flags. Edit local data, output, GPU, and checkpoint paths before launching. Exporters and trainers can create substantial output directories.

## Later detector and SAM2 Video workflow

The later experiment code is in `experiments/detector_video_iom06/`. It contains detector fine-tuning, Video training, full-GT IoM filtering, and audit scripts. Its README records the experiment and historical settings. Required datasets, checkpoints, SAM2 source, and run outputs are external and are not included here. Treat that README as experiment provenance, not proof that those assets are present.

## Large-block inference

`block_inference/` contains Python entry points and runtime helpers for nuclear segmentation, detection, SAM2 Video prompting, score-based propagation, crop adjustment, and 3D connected-component assembly. Start with its README and `config.example.json`.

The example config is a template. Supply paths to the raw block, detector, nuclear model, SAM2 implementation and checkpoint, and output directory. This snapshot omits model weights and vendored model code, so it cannot run end-to-end until those dependencies are restored. Confirm input axis order, voxel spacing, origin, and coordinate conventions for each block; do not assume coordinates from a crop are global coordinates.

## Environment and dependencies

There is no single pinned environment for all subprojects. Install dependencies for the workflow you intend to use and keep CUDA and PyTorch versions compatible with the machine. `block_inference/requirements.txt` applies only to that inference subproject; it is not a repository-wide lockfile.

Recommended first steps:

1. Create a dedicated environment for the selected workflow.
2. Restore only required data and checkpoint assets from authorized storage.
3. Copy the relevant example config and replace machine-specific paths.
4. Inspect the script help and run its provided checks before processing a full block.
5. Write outputs outside the source checkout.

Tests are provided beside relevant code, including `block_inference/test_bundle.py` and `block_stitch/test_stitch_core.py`. They were not run while preparing this repository.

## Data, weights, and generated files

Keep large or private artifacts outside Git. `.gitignore` excludes common data, checkpoint, cache, and run-output formats. Review ignore rules before adding other file types. Do not add raw EM volumes, masks, model weights, credentials, per-instance result JSON, logs, or HTML panels to a public repository.

## GitHub handoff

This repository is initialized on the 113 server on branch `main`. Files are staged for the initial commit. No Git remote is configured and nothing has been pushed. Set your Git author identity, review `git status`, make the initial commit, add your GitHub remote, and push when ready.