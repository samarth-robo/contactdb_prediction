import torch.utils.data as tdata
import os
import numpy as np
import utils
import transforms3d.euler as txe
from IPython.core.debugger import set_trace
osp = os.path


class PointCloudDataset(tdata.Dataset):
  def __init__(self, data_dir, instruction, train, n_points=2500,
      include_sessions=None, exclude_sessions=None,
      random_rotation=180, random_scale=0.1, color_thresh=0.4):
    super(PointCloudDataset, self).__init__()
    data_dir = osp.expanduser(data_dir)
    self.n_points = n_points
    self.random_rotation = random_rotation
    self.random_scale = random_scale
    self.color_thresh = color_thresh

    # list the voxel grids
    self.filenames = []
    for filename in next(os.walk(data_dir))[-1]:
      if '_hollow.npy' not in filename:
        continue
      if '_{:s}_'.format(instruction) not in filename:
        continue
      session_name = filename.split('_')[0]
      if include_sessions is not None:
        if session_name not in include_sessions:
          continue
      if exclude_sessions is not None:
        if session_name in exclude_sessions:
          continue
      object_name = '_'.join(filename.split('.')[0].split('_')[2:-1])
      if train:
        if object_name in utils.test_objects:
          continue
      else:
        if object_name not in utils.test_objects:
          continue
      self.filenames.append(osp.join(data_dir, filename))

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, index):
    _, _, _, c, xx, yy, zz = np.load(self.filenames[index])
    pts = np.vstack((xx, yy, zz))
    offset = (pts.max(1, keepdims=True) + pts.min(1, keepdims=True)) / 2
    pts -= offset
    scale = max(pts.max(1) - pts.min(1)) / 2
    pts /= scale
    pts = np.vstack((pts, scale * np.ones(pts.shape[1])))
    c = utils.discretize_texture(c, thresh=self.color_thresh)

    # resample
    choice = np.random.choice(pts.shape[1], size=self.n_points, replace=True)
    pts = pts[:, choice]
    c   = c[choice]

    # random perturbations
    # rotation
    if self.random_rotation > 0:
      theta = np.random.uniform(-np.pi*self.random_rotation/180,
        np.pi*self.random_rotation/180)
      R = txe.euler2mat(0, 0, theta)
      pts[:3] = R @ pts[:3]
    # scale
    if self.random_scale > 0:
      axis = np.random.choice(2)
      T = np.eye(3)
      T[axis, axis] = np.random.uniform(1-self.random_scale, 1+self.random_scale)
      pts[:3] = T[:3] @ pts[:3]

    return pts.astype(np.float32), c.astype(np.int)


if __name__ == '__main__':
  import open3d
  from utils import show_pointcloud
  dset = PointCloudDataset(osp.join('data', 'voxelized_meshes'), 'use',
    train=True, random_rotation=180)
  for idx in np.random.choice(len(dset), 10):
    geom, tex = dset[idx]
    show_pointcloud(geom[:3].T, tex)
