# HSI-SC-NeRF

Codebase for **HSI-SC-NeRF**, a stationary-camera-based hyperspectral NeRF framework for 3D plant phenotyping and postharvest agricultural inspection.

## Links

- **Paper:** [arXiv:2602.16950](https://arxiv.org/abs/2602.16950)
- **Dataset:** [HSI-SC-NeRF on Hugging Face](https://huggingface.co/datasets/BGLab/HSI-SC-NeRF)

## Overview

HSI-SC-NeRF extends a stationary-camera NeRF pipeline to hyperspectral 3D reconstruction. Unlike conventional NeRF pipelines that require camera motion around a static object, this framework uses a **fixed camera and a rotating object**, enabling a simpler and more practical acquisition setup under controlled imaging conditions.

This repository provides code and commands for:

1. **Pose estimation** from pseudo-RGB images using COLMAP
2. **Hyperspectral NeRF training**
3. **Evaluation** of reconstruction quality
4. **Export** of hyperspectral 3D point clouds

## Pipeline

### 1. Pose Estimation (COLMAP) using pseudo-RGB images

```bash
time ns-process-data images     --data <INPUT_IMAGE_DIR>     --output-dir <PROCESSED_OUTPUT_DIR>     --sfm-tool colmap     --matching-method sequential     --feature-type any     --matcher-type any     --use-single-camera-mode     --same-dimensions     --no-refine-intrinsics     --camera-type simple_pinhole     --num-downscales 3
```

### 2. Train HSI NeRF

```bash
ns-train nerfacto     --data <PROCESSED_OUTPUT_DIR>     --output-dir <TRAIN_OUTPUT_DIR>     --pipeline.model.num-output-channels <NUM_HSI_CHANNELS>     --pipeline.model.predict-normals True     --viewer.quit-on-train-completion True     --pipeline.model.far_plane <FAR_PLANE>     --pipeline.model.near_plane <NEAR_PLANE>     --pipeline.datamanager.pixel-sampler.max-num-iterations <SAMPLER_MAX_ITERS>     --pipeline.model.camera-optimizer.mode <CAMERA_OPTIMIZER_MODE>     --pipeline.model.hsi_loss_mult <HSI_LOSS_WEIGHT>     --pipeline.model.angular_loss_mult <ANGULAR_LOSS_WEIGHT>     --max-num-iterations <MAX_ITERS>
```

### 3. Evaluation

```bash
ns-eval     --load-config <CONFIG_YML_PATH>     --output-path eval_metrics.json
```

### 4. Export Hyperspectral Point Cloud

```bash
ns-export hsi-pointcloud     --load-config <CONFIG_YML_PATH>     --output-dir <POINTCLOUD_OUTPUT_DIR>     --num-points <NUM_POINTS>
```

## Example Workflow

### Step 1. Preprocess and estimate poses

```bash
ns-process-data images\
--data data/pseudo_rgb     --output-dir data/processed\
--sfm-tool colmap     --matching-method sequential     --feature-type any     --matcher-type any\
--use-single-camera-mode     --same-dimensions     --no-refine-intrinsics     --camera-type simple_pinhole     --num-downscales 3
```

### Step 2. Train hyperspectral NeRF

```bash
ns-train nerfacto\
--data data/processed     --output-dir outputs\
--pipeline.model.num-output-channels <the number of channels>\
--pipeline.model.predict-normals True     --viewer.quit-on-train-completion True\
--pipeline.model.far_plane <0.6>     --pipeline.model.near_plane <0.02>\
--pipeline.datamanager.pixel-sampler.max-num-iterations <20000>\
--pipeline.model.camera-optimizer.mode <off>\
--pipeline.model.hsi_loss_mult <1.0>     --pipeline.model.angular_loss_mult <0.0>\
--max-num-iterations <20000>
```

### Step 3. Evaluate

```bash
ns-eval\
--load-config outputs/<experiment_name>/nerfacto/<timestamp>/config.yml\
--output-path eval_metrics.json
```

### Step 4. Export hyperspectral point cloud

```bash
ns-export hsi-pointcloud\
--load-config outputs/<experiment_name>/nerfacto/<timestamp>/config.yml\
--output-dir exports/hsi_pointcloud\
--num-points <1000000>
```

## Dataset

The dataset associated with this project is publicly available on Hugging Face:

[HSI-SC-NeRF Dataset](https://huggingface.co/datasets/BGLab/HSI-SC-NeRF)

Please refer to the dataset card for details on the imaging setup, spectral calibration workflow, directory structure, and released reconstruction outputs.

## Requirements

- Python
- Nerfstudio
- COLMAP
- Hyperspectral dataset prepared for training

## Notes

- Replace placeholder paths such as `<INPUT_IMAGE_DIR>` and `<CONFIG_YML_PATH>` before running.
- Make sure the processed dataset and config paths match your local setup.
- Adjust the number of output channels to match your hyperspectral data.

## Citation

If you use this code or dataset in your research, please cite the corresponding paper.

```bibtex
@article{ku2026hyperstationarynerf,
  title   = {HSI-SC-NeRF: NeRF-based Hyperspectral 3D Reconstruction using a Stationary Camera for Agricultural Applications},
  author  = {Ku, Kibon and Jubery, Talukder Z. and Krishnamurthy, Adarsh and Ganapathysubramanian, Baskar},
  year    = {2026},
  journal = {arXiv preprint arXiv:2602.16950}
}
```
