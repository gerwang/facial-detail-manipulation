# Structure-aware Editable Morphable Model for 3D Facial Detail Animation and Manipulation

Code for our ECCV 2022 paper "Structure-aware Editable Morphable Model for 3D Facial Detail Animation and Manipulation".

### [Paper](https://arxiv.org/abs/2207.09019) | [Presentation Video](https://www.youtube.com/watch?v=tONe8QzR0u0) | [Supplementary Video](https://www.youtube.com/watch?v=HgoN8wM56AM) | [Poster](https://github.com/gerwang/facial-detail-manipulation/releases/download/v1.0/4426.pdf)

##### Expression editing

![](./imgs/exp_inter.gif)

##### Age editing

![](./imgs/age_progression.gif)

##### Wrinkle line editing

![](./imgs/wrinkle_line_edit.gif)

## Getting started

- Clone this repository and `cd facial-detail-manipulation`
- Run `conda create -n semm python=3.7 && conda activate semm`
- Run `pip install -r requirements.txt`
- Install `pytorch==1.9.0 torchvision==0.10.0` from [PyTorch](https://pytorch.org/get-started/previous-versions/#v190).
- Install `pytorch3d==0.6.2` from [PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md). The author uses `pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.6.2"`.
- Install [ibug-face-detection](https://github.com/hhj1897/face_detection) and [ibug-face-alignment](https://github.com/hhj1897/face_alignment).

- Download pretrained model from [here](https://drive.google.com/file/d/16g8zcvQXts9SuU5tgstHpWMgQ49vBmeY/view?usp=sharing), unzip and put `SEMM` into `./checkpoints`.
- Download `model_mse.pth` from [here](https://drive.google.com/file/d/1lc3GsP8XfIMDJfvamMmou2sOTG0ID02p/view?usp=sharing) and put it into `./checkpoints`.
- Download `facescape_bilinear_model_v1_3.zip` from https://facescape.nju.edu.cn/Page_Download/  and put `core_847_50_52.npy` into `./predef`.
- Download `dpmap_single_net_G.pth` from [here](https://drive.google.com/file/d/18j8bnj5IHP0u2jNuIrWh7dvQkfagBxsM/view?usp=sharing) and put it into `./checkpoints`.
- Install **Blender 3.2** and **FFmpeg**. Make sure `blender` and `ffmpeg` can be accessed in shell, or you can set the `blender_path` and `ffmpeg_path` in `./experiments/both_cond_launcher.py`.
- Install `scipy` for Blender. Typically, we should first download the [get-pip](https://github.com/pypa/get-pip) script, run `${blender_path}/3.2/python/bin/python3.10 get-pip.py` and then run `${blender_path}/3.2/python/bin/python3.10 -m pip install scipy`.

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

### Age progression animation

```shell
python -m experiments both_cond age_progression SEMM --gpu_id 0
```

The results will be in `./demo/output/${filename}/age_progression/`.

### Blendshape animation

```shell
python -m experiments both_cond bs_anime SEMM --gpu_id 0
```

The results will be in `./demo/output/${filename}/bs_anime_${clip_name}/`.

### Interactive wrinkle line editing

```shell
python -m experiments both_cond demo SEMM --gpu_id 0
```

An GUI window will pop up. You can import a displacement map and edit it by drawing or erasing lines.

<details>
    <summary><b>Common issues</b></summary>
If you encounter the problem "qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "${conda_path}/envs/semm/lib/python3.7/site-packages/cv2/qt/plugins" even though it was found", that's a conflict between opencv-python and pyqt5. Consider using
    
```shell
conda install -c anaconda py-opencv
conda install -c alges pyqt 
```
</details>

## Training

We provide a small sample dataset that comprises identities in FaceScape's [publishable_list_v1.txt](https://facescape.nju.edu.cn/static/publishable_list_v1.txt). You can commence training by utilizing the sample dataset through the following command:

```shell
python -m experiments both_cond train SEMM --gpu_id 01
```
The default configuration assumes running on two RTX 3090 GPUs.

### Training data preprocessing

Given that the sample dataset is insufficiently small to lead to favorable results, it is necessary to apply and download `Topologically Uniformed Model(TU-Model)` (`facescape_trainset_001_100.zip` ~ `facescape_trainset_801_847.zip`) in the FaceScape dataset. 

Once downloaded, extract the dataset to `/path/to/FaceScape`. The folder should comprise of 847 folders, labeled from `1` to `847`.

Next, use the following command to process the training dataset:

```shell
python -m detail_shape_process.train_detail_process --input /path/to/FaceScape --output /path/to/processed/dataset
```

Lastly, make sure to update the subsequent line:

```python
        opt.set(
            dataroot="./predef/sample_dataset/",  # just a small sample dataset
```

in `experiments/both_cond_launcher.py` to point to the processed dataset.


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

