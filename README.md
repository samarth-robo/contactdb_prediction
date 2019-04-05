# [ContactDB: Analyzing and Predicting Grasp Contact via Thermal Imaging](https://contactdb.cc.gatech.edu)
This repository contains code to analyze and predict contact maps for human grasping, presented in the paper 

[ContactDB: Analyzing and Predicting Grasp Contact via Thermal Imaging](https://contactdb.cc.gatech.edu/contactdb_paper.pdf) - [Samarth Brahmbhatt](https://samarth-robo.github.io/), [Cusuh Ham](https://cusuh.github.io/), [Charles C. Kemp](http://ckemp.bme.gatech.edu/), and [James Hays](https://www.cc.gatech.edu/~hays/), CVPR 2019

[Paper (CVPR 2019 Oral)](https://contactdb.cc.gatech.edu/contactdb_paper.pdf) | [Supplementary Material](https://contactdb.cc.gatech.edu/contactdb_supp.pdf) | [Explore the dataset](https://contactdb.cc.gatech.edu/contactdb_explorer.html) | Poster | Slides

Please see [contactdb_utils](https://github.com/samarth-robo/contactdb_utils) for access to raw ContactDB data, and code to process it.

## Setup
1. Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (Python 3.x version).
2. Download this repository: `git clone https://github.com/samarth-robo/contactdb_prediction.git`. Commands for the following steps should be executed from the `contactdb_prediction` directory.
2. Create the `contactdb_prediction` environment: `conda create env -f environment.yml`, and activate it: `source activate contactdb_prediction`.
3. Download the voxelized contact maps from [this Dropbox link](https://www.dropbox.com/sh/x5ivxw75tvf6tax/AADXw7KRWbH3eEofbbr6NQQga?dl=0) (17.9 GB). If the download location is `CONTACTDB_DATA_DIR`, make a symlink to it: `ln -s CONTACTDB_DATA_DIR data/voxlelized_meshes`.
4. Download the trained models from [this Dropbox link](https://www.dropbox.com/sh/3kvyhin9030mdzo/AAC_eYOVAvXMRhsAJsDlL_soa?dl=0) (700 MB). If the download location is `CONTACTDB_MODELS_DIR`, make a symlink to it: `ln -s CONTACTDB_MODELS_DIR data/checkpoints`.
5. (Optional, for comparison purposes): Download the predicted contact maps from [this Dropbox link](https://www.dropbox.com/sh/zrpgtoycbik0iq3/AAAHMyzs9Lc2kH8UPZttRCmGa?dl=0).

## Predicting Contact Maps
We propose two methods to make diverse contact map predictions: [DiverseNet](http://openaccess.thecvf.com/content_cvpr_2018/papers/Firman_DiverseNet_When_One_CVPR_2018_paper.pdf) and [Stochastic Multiple Choice Learning (sMCL)](https://papers.nips.cc/paper/6270-stochastic-multiple-choice-learning-for-training-diverse-deep-ensembles). Their code is in **separate branches**.

### DiverseNet Models
First, check out the correct branch: `git checkout diversenet`.

#### Voxel-grid 3D representation (VoxNet):
- `Use` grasping instruction: `python eval.py --instruction use --config configs/voxnet.ini --checkpoint data/checkpoints/use_voxnet_diversenet_release/checkpoint_model_86_val_loss\=0.01107167.pth`

- `Handoff` grasping instruction: `python eval.py --instruction handoff --config configs/voxnet.ini --checkpoint data/checkpoints/handoff_voxnet_diversenet_release/checkpoint_model_167_val_loss\=0.01268427.pth`

#### Pointcloud 3D representation (PointNet):
- `Use` grasping instruction: `python eval.py --instruction use --config configs/pointnet.ini --checkpoint data/checkpoints/use_pointnet_diversenet_release/checkpoint_model_29_val_loss\=0.6979221.pth`

- `Handoff` grasping instruction: `python eval.py --instruction handoff --config configs/pointnet.ini --checkpoint data/checkpoints/handoff_pointnet_diversenet_release/checkpoint_model_745_val_loss\=0.5969936.pth`

### sMCL Models
First, check out the correct branch: `git checkout smcl`.

#### Voxel-grid 3D representation (VoxNet):
- `Use` grasping instruction: `python eval.py --instruction use --config configs/voxnet.ini --checkpoint_dir data/checkpoints/use_voxnet_diversenet_release`

- `Handoff` grasping instruction: `python eval.py --instruction handoff --config configs/voxnet.ini --checkpoint_dir data/checkpoints/handoff_voxnet_diversenet_release`

#### Pointcloud 3D representation (PointNet):
- `Use` grasping instruction: `python eval.py --instruction use --config configs/pointnet.ini --checkpoint_dir data/checkpoints/use_pointnet_diversenet_release`

- `Handoff` grasping instruction: `python eval.py --instruction handoff --config configs/pointnet.ini --checkpoint_dir data/checkpoints/handoff_pointnet_diversenet_release`

## Analyzing Contact Maps
The analysis code is in the `master` branch: `git checkout master`. Analysis will require downloading the **full** ContactDB dataset, see [download instructions](https://github.com/samarth-robo/contactdb_utils). Let's say the dataset was downloaded at `CONTACTDB_DATASET_DIR`.

- Calculate average contact areas for each object: `python analyze_contact_area.py --instruction use`. This will save the information in `data/use_contact_areas.pkl`, which can be plotted with `plot_contact_areas.py` to generate Figures 5(a) and 5(b) of the paper.

- Analyze relationship between hand size and single-handed/bi-manual grasps: `python analyze_bimanual_grasps.py --instruction handoff --data_dir CONTACTDB_DATASET_DIR --object_names bowl,cube_large,cylinder_large,piggy_bank,pyramid_large,sphere_large,utah_teapot`. This should produce Figure 6 of the paper.

- Clustering the contact maps: `python cluster_contact_maps.py --object_name camera --instruction use`. Add the `--symmetric` switch for symmetric objects like `wine_glass`. This should cluster the contact maps for that object and print out the cluster center and assignments. Useful for making Figure 3 of the paper.

- Analyzing contact frequency of active areas: The cropped meshes of the active areas mentioned in Table 2 of the paper are saved in `data/active_areas`. To crop meshes yourself, see this [Open3D tutorial](http://www.open3d.org/docs/tutorial/Advanced/interactive_visualization.html#crop-geometry). After this, run e.g. for `camera`: `python analyze_active_areas.py --object camera --instruction use`. This will print out contact frequency for all all active areas of that object, and their union (frequency of touching area A `or` area B). `frequency(A and B) = frequency(A) + frequency(B) - frequency(A or B)`.

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
