# [ContactDB: Analyzing and Predicting Grasp Contact via Thermal Imaging](https://contactdb.cc.gatech.edu)
This repository contains code to analyze and predict contact maps for human grasping, presented in the paper 

[ContactDB: Analyzing and Predicting Grasp Contact via Thermal Imaging](https://contactdb.cc.gatech.edu/contactdb_paper.pdf) - [Samarth Brahmbhatt](https://samarth-robo.github.io/), [Cusuh Ham](https://cusuh.github.io/), [Charles C. Kemp](http://ckemp.bme.gatech.edu/), and [James Hays](https://www.cc.gatech.edu/~hays/), CVPR 2019

[Paper (CVPR 2019 Oral)](https://contactdb.cc.gatech.edu/contactdb_paper.pdf) | [Supplementary Material](https://contactdb.cc.gatech.edu/contactdb_supp.pdf) | [Explore the dataset](https://contactdb.cc.gatech.edu/contactdb_explorer.html) | Poster | Slides

Please see [contactdb_utils](https://github.com/samarth-robo/contactdb_utils) for access to raw ContactDB data, and code to process it.

## Setup
1. Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (Python 3.x version).
2. Download this repository: `git clone https://github.com/samarth-robo/contactdb_prediction.git`. Commands for the following steps should be executed from the `contactdb_prediction` directory.
2. Create the `contactdb_prediction` environment: `conda create env -f environment.yml`, and activate it: `source activate contactdb_prediction`.
This branch has code for **analyzing contact maps** (Section 4 of the paper), and for preprocessing them for ML experiments. For PyTorch code to predict contact maps, see the `diversenet` and `smcl` branches.

## Analyzing Contact Maps

Download the contact maps from [this Dropbox link](https://www.dropbox.com/sh/gzwk21ssod63xdl/AAAJ5StPMS2eid2MnZddBGsca?dl=0) (11.2 GB). If the download location is `CONTACTDB_DATA_DIR`, make a symlink to it: `ln -s CONTACTDB_DATA_DIR data/contactmaps`.

- Calculate average contact areas for each object: `python analyze_contact_area.py --instruction use`. This will save the information in `data/use_contact_areas.pkl`, which can be plotted with `plot_contact_areas.py` to generate Figures 5(a) and 5(b) of the paper.

- Analyze relationship between hand size and single-handed/bi-manual grasps: `python analyze_bimanual_grasps.py --instruction handoff --data_dir CONTACTDB_DATASET_DIR --object_names bowl,cube_large,cylinder_large,piggy_bank,pyramid_large,sphere_large,utah_teapot`. This should produce Figure 6 of the paper.

- Clustering the contact maps: `python cluster_contact_maps.py --object_name camera --instruction use`. Add the `--symmetric` switch for symmetric objects like `wine_glass`. This should cluster the contact maps for that object and print out the cluster center and assignments. Useful for making Figure 3 of the paper.

- Analyzing contact frequency of active areas: The cropped meshes of the active areas mentioned in Table 2 of the paper are saved in `data/active_areas`. To crop meshes yourself, see this [Open3D tutorial](http://www.open3d.org/docs/tutorial/Advanced/interactive_visualization.html#crop-geometry). After this, run e.g. for `camera`: `python analyze_active_areas.py --object camera --instruction use`. This will print out contact frequency for all all active areas of that object, and their union (frequency of touching area A `or` area B). `frequency(A and B) = frequency(A) + frequency(B) - frequency(A or B)`.

## Preprocessing Data
This section describes how we preprocessed ContactDB data for the contactmap prediction models.

- First, download the 3D models of the objects from [this Dropbox link](https://www.dropbox.com/sh/5rnxri7dzh9ciy3/AABXgwqpmBtlXgQc8aWBVl8aa?dl=0) (82 MB). If the download location is `CONTACTDB_OBJ_MODELS_DIR`, make a symlink to it: `ln -s CONTACTDB_OBJ_MODELS_DIR data/object_models`.

- We use Patrick Min's [binvox](http://www.patrickmin.com/binvox/) to voxelize the object meshes for VoxNet models. Generate the voxelgrids using `python generate_binvoxes.py`. The default parameters should work fine.

- Next, we need to generate the voxelgrids and pointclouds with associated contact textures for the prediction models. Create the directory for storing them: `mkdir data/voxelized_meshes` (stores both the pointclouds and voxelgrids).

- Generate the voxelgrids using: `python voxelize_mesh.py --instruction <use | handoff>`

- Generate the pointclouds using: `python voxelize_mesh.py --instruction <use | handoff> --hollow`.

We make the preprocessed data available for direct download at [this Dropbox link](https://www.dropbox.com/sh/x5ivxw75tvf6tax/AADXw7KRWbH3eEofbbr6NQQga?dl=0) (17.9 GB).


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
