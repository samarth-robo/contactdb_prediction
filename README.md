# [ContactDB: Analyzing and Predicting Grasp Contact via Thermal Imaging](https://contactdb.cc.gatech.edu)
This repository contains code to analyze and predict contact maps for human grasping, presented in the paper 

[ContactDB: Analyzing and Predicting Grasp Contact via Thermal Imaging](https://contactdb.cc.gatech.edu/contactdb_paper.pdf) - [Samarth Brahmbhatt](https://samarth-robo.github.io/), [Cusuh Ham](https://cusuh.github.io/), [Charles C. Kemp](http://ckemp.bme.gatech.edu/), and [James Hays](https://www.cc.gatech.edu/~hays/), CVPR 2019

[Paper (CVPR 2019 Oral)](https://contactdb.cc.gatech.edu/contactdb_paper.pdf) | [Supplementary Material](https://contactdb.cc.gatech.edu/contactdb_supp.pdf) | [Explore the dataset](https://contactdb.cc.gatech.edu/contactdb_explorer.html) | Poster | Slides

Please see [contactdb_utils](https://github.com/samarth-robo/contactdb_utils) for access to raw ContactDB data, and code to process it.

## Setup
1. Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (Python 3.x version).
2. Download this repository: `git clone https://github.com/samarth-robo/contactdb_prediction.git`. Commands for the following steps should be executed from the `contactdb_prediction` directory.
2. Create the `contactdb_prediction` environment: `conda create env -f environment.yml`, and activate it: `source activate contactdb_prediction`.
3. Download the preprocessed contact maps from [this Dropbox link](https://www.dropbox.com/sh/x5ivxw75tvf6tax/AADXw7KRWbH3eEofbbr6NQQga?dl=0) (17.9 GB). If the download location is `CONTACTDB_DATA_DIR`, make a symlink to it: `ln -s CONTACTDB_DATA_DIR data/voxelized_meshes`.
4. Download the trained models from [this Dropbox link](https://www.dropbox.com/sh/3kvyhin9030mdzo/AAC_eYOVAvXMRhsAJsDlL_soa?dl=0) (700 MB). If the download location is `CONTACTDB_MODELS_DIR`, make a symlink to it: `ln -s CONTACTDB_MODELS_DIR data/checkpoints`.
5. (Optional, for comparison purposes): Download the predicted contact maps from [this Dropbox link](https://www.dropbox.com/sh/zrpgtoycbik0iq3/AAAHMyzs9Lc2kH8UPZttRCmGa?dl=0).

## Predicting Contact Maps
We propose two methods to make diverse contact map predictions: [DiverseNet](http://openaccess.thecvf.com/content_cvpr_2018/papers/Firman_DiverseNet_When_One_CVPR_2018_paper.pdf) and [Stochastic Multiple Choice Learning (sMCL)](https://papers.nips.cc/paper/6270-stochastic-multiple-choice-learning-for-training-diverse-deep-ensembles). This branch has code for the **diversenet models**. Checkout the `smcl` branch for sMCL code.

Predict contact maps for the 'use' instruction, using the voxel grid 3D representation:

```
$ python eval.py --instruction use --config configs/voxnet.ini --checkpoint data/checkpoints/use_voxnet_diversenet_release/checkpoint_model_86_val_loss\=0.01107167.pth
pan loss = 0.0512
mug loss = 0.0706
wine_glass loss = 0.1398
```

You can add the `--show object <pan | mug | wine_glass>` flag to show the 10 diverse predictions:
```
$ python eval.py --instruction use --config configs/voxnet.ini --checkpoint data/checkpoints/use_voxnet_diversenet_release/checkpoint_model_86_val_loss\=0.01107167.pth --show_object mug
mug loss = 0.0706
```
<img alt="use-voxnet-diversenet-mug0" src="use_voxnet_diversenet_mug_prediction0.png" width="100">
<img alt="use-voxnet-diversenet-mug1" src="use_voxnet_diversenet_mug_prediction1.png" width="100">

In general, the command is

`python eval.py --instruction <use | handoff> --config <configs/voxnet.ini | configs/pointnet.ini> --checkpoint <checkpoint filename>`

Use the following checkpoints:

|      Method        |                                             Checkpoint                                            |
|:------------------:|:-------------------------------------------------------------------------------------------------:|
|   Use - VoxNet     | data/checkpoints/use_voxnet_diversenet_release/checkpoint_model_86_val_loss\=0.01107167.pth       |
|  Use - PointNet    | data/checkpoints/use_pointnet_diversenet_release/checkpoint_model_29_val_loss\=0.6979221.pth      |
| Handoff - VoxNet   | data/checkpoints/handoff_voxnet_diversenet_release/checkpoint_model_167_val_loss\=0.01268427.pth  |
| Handoff - PointNet | data/checkpoints/handoff_pointnet_diversenet_release/checkpoint_model_745_val_loss\=0.5969936.pth |

## Training your own models

The base command is

`python train_val.py --instruction <use | handoff> --config <configs/voxnet.ini | configs/pointnet.ini> [--device <GPU ID> --checkpoint_dir <directory where checkpints are saved> --data_dir <directory where data is downloaded>]`

## Citation
```
@inproceedings{brahmbhatt2018contactdb,
  title={{ContactDB: Analyzing and Predicting Grasp Contact via Thermal Imaging}},
  author={Samarth Brahmbhatt and Cusuh Ham and Charles C. Kemp and James Hays},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019},
  note={\url{https://contactdb.cc.gatech.edu}}
}
```
