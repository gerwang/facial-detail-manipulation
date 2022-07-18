# Structure-aware Editable Morphable Model for 3D Facial Detail Animation and Manipulation

Code for our ECCV 2022 paper "Structure-aware Editable Morphable Model for 3D Facial Detail Animation and Manipulation".

![](./imgs/exp_inter.gif)

![](./imgs/age_progression.gif)

![](./imgs/wrinkle_line_edit.gif)

## Getting started

- Download `model_mse.pth` from here and put it into `./checkpoints`.
- Download pretrained model from here, unzip and put `SEMM` into `./checkpoints`.
- Download `facescape_bilinear_model_v1_3.zip` from https://facescape.nju.edu.cn/Page_Download/  and put `core_847_50_52.npy` into `./predef`.
- Download `dpmap_single_net_G.pth` from [here](https://drive.google.com/file/d/18j8bnj5IHP0u2jNuIrWh7dvQkfagBxsM/view?usp=sharing) and put it into `./checkpoints`.

- Install **Blender 3.0.1** and **FFmpeg**. Make sure `blender` and `ffmpeg` can be accessed in shell, or you can set the `blender_path` and `ffmpeg_path` in `./experiments/both_cond_launcher.py`.

## Usage

### Data preprocessing

Before performing the editing tasks below, we should first pre-process the data via

```shell
python -m detail_shape_process.detail_process --input ./demo/input --output ./demo/output
```

### Expression and age editing

```shell
python -m experiments both_cond exp_age_edit SEMM --gpu_id 0
```

The reconstruction results are located in `./demo/output/${filename}/recon/`, the expression editing results are in `./demo/output/${filename}/exp_edit/`, and the age editing results are in `./demo/output/${filename}/age_edit/`.

### Expression interpolation

```shell
python -m experiments both_cond exp_inter SEMM --gpu_id 0
```

The results will be in `./demo/output/${filename}/exp_inter_${target_exp_name}/`. You can set which data to process and the target expression via `file_paths` and `target_exps` in `exp_inter_options` in the file `./experiments/both_cond_launcher.py`.

### Blendshape animation

```shell
python -m experiments both_cond bs_anime SEMM --gpu_id 0
```

The results will be in `./demo/output/${filename}/bs_anime_${clip_name}/`.

### Age progression animation

```shell
python -m experiments both_cond age_progression SEMM --gpu_id 0
```

The results will be in `./demo/output/${filename}/age_progression/`.

### Interactive wrinkle line editing

```shell
python -m experiments both_cond demo SEMM --gpu_id 0
```

An GUI window will pop up. You can import a displacement map and edit it by drawing or erasing lines.

## Acknowledgements

The network to extract wrinkle lines is from **[sketch_simplification](https://github.com/bobbens/sketch_simplification)**. The 3DMM fitting part is adapted from **[Detailed3DFace](https://github.com/yanght321/Detailed3DFace)**. We use the facial landmark detector in **[face_alignment](https://github.com/hhj1897/face_alignment)**. The training code is based on **[swapping-autoencoder-pytorch](https://github.com/taesungp/swapping-autoencoder-pytorch)**. The texture extractor code is from **[DECA](https://github.com/YadiraF/DECA)**. The drawing GUI is based on **[DeepFaceDrawing-Jittor](https://github.com/IGLICT/DeepFaceDrawing-Jittor)**. We thank the above projects for making their code open-source.


## Citation

If you find our code or paper useful, please cite as:

```latex
@inproceedings{ling2022structure,
  title={Structure-aware Editable Morphable Model for 3D Facial Detail Animation and Manipulation},
  author={Ling, Jingwang and Wang, Zhibo and Lu, Ming and Wang, Quan and Qian, Chen and Xu, Feng},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year = {2022}
}
```

