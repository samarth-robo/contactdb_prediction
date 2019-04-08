import torch.utils.data as tdata
import os
import numpy as np
import utils
import transforms3d.euler as txe
from collections import OrderedDict
from IPython.core.debugger import set_trace
osp = os.path


class PointCloudDataset(tdata.Dataset):
  def __init__(self, data_dir, instruction, train, n_points=2500,
      include_sessions=None, exclude_sessions=None,
      random_rotation=180, random_scale=0.1,
      n_ensemble=20, color_thresh=0.4, test_only=False):
    super(PointCloudDataset, self).__init__()
    data_dir = osp.expanduser(data_dir)
    self.n_points = n_points
    self.random_rotation = random_rotation
    self.random_scale = random_scale
    self.n_ensemble = n_ensemble
    self.color_thresh = color_thresh

    # list the voxel grids
    self.filenames = OrderedDict()
    for filename in next(os.walk(data_dir))[-1]:
      if '_hollow.npy' not in filename:
        continue
      if test_only:
        if 'testonly' not in filename:
          continue
      else:
        if '_{:s}_'.format(instruction) not in filename:
          continue
      session_name = filename.split('_')[0]
      if include_sessions is not None:
        if session_name not in include_sessions:
          continue
      if exclude_sessions is not None:
        if session_name in exclude_sessions:
          continue
      offset = 1 if test_only else 2
      object_name = '_'.join(filename.split('.')[0].split('_')[offset:-1])
      if not test_only:
        if train:
          if object_name in utils.test_objects:
            continue
        else:
          if object_name not in utils.test_objects:
            continue
      filename = osp.join(data_dir, filename)
      if object_name not in self.filenames:
        self.filenames[object_name] = [filename]
      else:
        self.filenames[object_name].append(filename)

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, index):
    # load geometry
    object_name = list(self.filenames.keys())[index]
    _, _, _, _, xx, yy, zz = np.load(self.filenames[object_name][0])
    pts = np.vstack((xx, yy, zz))
    offset = (pts.max(1, keepdims=True) + pts.min(1, keepdims=True)) / 2
    pts -= offset
    scale = max(pts.max(1) - pts.min(1)) / 2
    pts /= scale
    pts = np.vstack((pts, scale*np.ones(pts.shape[1])))

    # resample
    pts_choice = np.random.choice(pts.shape[1], size=self.n_points, replace=True)
    pts = pts[:, pts_choice]

    # random perturbations
    # rotation
    if abs(self.random_rotation) > 0:
      theta = np.random.uniform(-np.pi*self.random_rotation/180,
        np.pi*self.random_rotation/180)
      R = txe.euler2mat(0, 0, theta)
      pts[:3] = R @ pts[:3]
    # scale
    if abs(self.random_scale) > 0:
      axis = np.random.choice(2)
      T = np.eye(3)
      T[axis, axis] = np.random.uniform(1-self.random_scale, 1+self.random_scale)
      pts[:3] = T @ pts[:3]

    # load textures
    N = len(self.filenames[object_name])
    filename_choice = np.arange(N)
    if self.n_ensemble > 0 and self.n_ensemble < N:
      filename_choice = np.random.choice(N, size=self.n_ensemble, replace=False)
    cs = []
    filenames = [self.filenames[object_name][c] for c in filename_choice]
    for filename in filenames:
      _, _, _, c, _, _, _ = np.load(filename)
      c = utils.discretize_texture(c, thresh=self.color_thresh)
      c = c[pts_choice]
      cs.append(c)
    cs = np.vstack(cs)

    return pts.astype(np.float32), cs.astype(np.int)

if __name__ == '__main__':
  n_ensemble = 1
  N_show = 30
  dset = PointCloudDataset(osp.join('data', 'voxelized_meshes'), 'use',
    train=True, random_rotation=180, n_ensemble=n_ensemble)
  for idx in np.random.choice(len(dset), N_show):
    geom, tex = dset[idx]
    utils.show_pointcloud(geom[:3].T, tex[0])
