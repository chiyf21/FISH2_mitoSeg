# Large-block inference source bundle

This folder contains source code, runtime policy, tests, an example config, and a dependency list for the large-block mitochondria inference workflow. It is a source-only subset: detector, nuclear, and SAM2 weights, vendored model code, raw EM data, and generated validation outputs are not included.

## Pipeline

1. Read a ZYX raw block and validate its dimensions, voxel spacing, and optional XYZ subvolume.
2. Run the nuclear model and detector, or reuse explicitly supplied precomputed outputs.
3. Filter and associate detections, then use them as prompts for SAM2 Video propagation.
4. Adjust crops when needed, record per-frame masks and scores, and assemble 3D connected components.
5. Write labels and per-instance metadata to the selected output directory.

Exact thresholds and behavior are defined by the code and `config.example.json`. Review the selected configuration before use. The snapshot remains experimental until the required external model assets and compatible environment are supplied.

## Configure and run

Copy `config.example.json` outside the repository and edit all paths, GPU choices, and block coordinates. Model and checkpoint paths must refer to assets installed separately.

```bash
python infer_block.py --config /path/to/local_config.json --check
python -u infer_block.py --config /path/to/local_config.json
```

Individual stages are also exposed:

```bash
python -u nucleus_infer.py --config /path/to/local_config.json
python -u detection_infer.py --config /path/to/local_config.json
python -u video_infer.py --config /path/to/local_config.json
```

Use each script's help output to confirm the current CLI. Input arrays use ZYX storage, while optional block bounds and detection metadata use the documented XYZ convention. Check the config and code before mixing global and crop-local coordinates.

## External requirements

- A compatible CUDA/PyTorch environment.
- The detector, nuclear model, SAM2 implementation, and checkpoint referenced by the local config.
- Raw block input and sufficient temporary disk space for the output label volume.

`requirements.txt` lists Python packages for this bundle. It does not install CUDA, provide model code or weights, or validate checkpoint compatibility. Start with a small local test before processing production data.