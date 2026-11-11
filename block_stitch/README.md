# Overlapping block SAM2 video inference

This is a separate, non-destructive inference experiment. It does not train,
change model weights, overwrite the original 19,947 masks, or fill holes.

## Frozen models

On cyf@166.111.72.113, under
`/path/to/project_data/JaneliaEM_experiments/detector_video_iom06_20261031/`:

- Detector: `detector_frozen.pt`, detector-only fine-tuning best epoch 3.
- SAM2 Video: `video_fast_fullgt_ft10/best.pt`, full-GT IoM corrected best epoch 8,
  2,752 optimizer updates. Actual checkpoint SHA256s are written into every case.

## Algorithm

1. The requested bbox uses **global XYZ voxel coordinates, half-open upper bounds**.
   Read real native EM at 8 x 8 x 30 nm. Only true source boundaries are padded.
2. Run the frozen detector with its existing native 128 x 128 x 32 tiling,
   reflected XY context and overlapping 10 x 188 x 188 detector inputs. Keep the
   original detection postprocessing; deduplicate only near-identical same-Z boxes.
3. Separately build SAM video windows: XY 256 x 256, stride 128; Z depth 32,
   stride 16. Edge windows cover the requested block. Z is never resized or
   interpolated; each actual XY image uses the SAM training preprocessing to 1024.
4. For each real detection box, run **every video window that fully contains its
   XY box and seed Z**. The full-box requirement avoids clipped prompts. If no XY
   grid window fully contains the box, add a detection-centred window (still 256).
   A larger-than-window detection is an explicit error requiring larger `--xy`.
5. One actual box prompt, on one seed slice; propagate forward and backward within
   each video window. Never use the old mask as a SAM prompt or trim outputs by it.
   Reuse image-encoder features among different seeds in the same window.
6. Transform output masks back to global pixel coordinates. On the **same Z**,
   compare full-mask IoU, including mask portions outside the pair's overlapping
   crop region. Do not compute overlap-only IoU, which falsely merges partial masks.
7. Same-Z edges with **IoU > 0.8** define connected groups. Within each group,
   select the frame with the highest trained **SAM mask IoU-head prediction**.
   This is independent of actual reference-mask IoU. Detection confidence breaks
   ties only. Save discarded-candidate provenance in `stitch_audit.json`.
8. A video supplies within-clip 3D identity. Link two video fragments across crop
   or Z-window boundaries only if at least two shared Z planes independently
   pass IoU > 0.8; a one-plane fragment may use its single shared plane. Save
   weak one-plane links separately. Output unresolved fragments separately;
   never force them into the same 3D instance merely because their boxes touch.
9. Export each 3D object as an NPZ with `z_global`, tight-XY `masks`, and global
   `origin_xy`; sparse object files may overlap. A memory-mapped union mask is a
   convenience visualization, **not an instance-label volume**.

## Known-instance pilot

`audit_instances.py` reads every original mask and writes `audit/audit.jsonl`:

- Short: occupied Z span < 10.
- Multi-hole: at least two slices each containing >=3 enclosed 2D holes of >=4 px,
  with total holes >=max(12 px, 2% foreground area).
- Straight-edge: at least two slices with an axis-aligned exposed boundary run
  >=max(16 px, 25% minimum XY side). Runs on native 128-grid seams receive extra
  selection priority. **This is a broad heuristic, not proof of stitching failure.**

The first pilot selects four cases per group, distinct IDs, with manageable XY/Z
extent, at most two cases per source neuron. All audit results are preserved;
The initial 12 cases include four unmatched one-slice extremes. Four additional short cases (6/7/8/9 slices) are appended without changing any threshold. This is a 16-case qualitative pilot, not inference over all flagged instances.

Each case uses a larger 512 x 512 XY block and the old bbox Z plus 16 real slices
on either side. Matching uses **actual same-Z box vs old-mask IoU >=0.3**, chooses
the highest matching seed per overlapping Z interval, then includes each selected
seed in every containing video window. Old masks influence **which known target
to trial**, but never the SAM input, window geometry, final mask, score ranking,
or 0.8 same-Z duplicate test. No match remains an explicit `no_detection` case;
no old bbox/GT prompt is substituted.

The old masks are model predictions, not manual ground truth. New/old IoU is
agreement, not accuracy. Show old segmentation, new per-object colors, and
added/removed pixels across **every block Z plane**, including beyond the old bbox.
Track component counts and boundary-touch flags: high-quality segmentation cannot
be concluded just from fewer holes, a longer Z span, or higher old/new agreement.

## Run arbitrary block

Use the existing SAM venv Python on 113:

```bash
CUDA_VISIBLE_DEVICES=0 /path/to/project_data/EM/point-MAE/runs/detsam_sam2base_cohort5k_20261031/venv/bin/python infer_block.py \
  --bbox X0 Y0 Z0 X1 Y1 Z1 --out /absolute/new/output \
  --xy 256 --xy-stride 128 --depth 32 --z-stride 16
```

Without `--reference-dir`, all actual detections inside the block are eligible;
no old masks are needed. High-density blocks may create many prompted tracks.
For very large volumes, call blockwise and reconcile the exported overlapping
objects in the same global coordinate system; current track-graph assembly is
in-memory, so a whole-brain single call is not recommended.

## Limitations to check

- Axis-aligned cuts and many holes are screening clues, not verified annotation errors.
- The trained SAM saw box-centred 3x crops; fixed 256 windows introduce a scale/context
  shift. A poor result may reflect that shift or detector domain transfer.
- 0.8 is strict: partial views and changing masks may remain split. Lowering it
  silently is unsafe; inspect saved links before a controlled threshold comparison.
- Connected-component duplicate grouping is transitive: A-B and B-C may group even
  when A-C <=0.8. All direct edges remain in the audit file for this reason.
- The SAM score is not guaranteed calibrated across windows; boundary-touching
  winners are explicitly marked, not automatically repaired.
- No detections in a Z region can leave a gap. No hallucinated continuation or
  interpolation is introduced. Nonempty masks at the outer block boundary mean
  the reconstruction may need a larger block.

## Artifacts

Experiment: `/path/to/project_data/JaneliaEM_experiments/block_video_stitch_20261101/`

- `audit/audit_completed.json`, `audit/audit.jsonl`, `audit/selected.json`
- `cases/instance_XX/detections.json`, `selected_detections.json`, `stitch_audit.json`
- `cases/instance_XX/object_XXXX.npz`, `predicted_union.npy`, `completed.json`
- Panel: `http://127.0.0.1:18071/training_pipeline/block_video_stitch_pilot/`
