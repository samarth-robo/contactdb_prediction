import torch.utils.data as tdata
import os
import numpy as np
import transforms3d.euler as txe
import utils
from IPython.core.debugger import set_trace
osp = os.path


class VoxelDataset(tdata.Dataset):
  def __init__(self, data_dir, instruction, train,
      grid_size=64, include_sessions=None, exclude_sessions=None,
      random_rotation=180, color_thresh=0.4):
    super(VoxelDataset, self).__init__()

    data_dir = osp.expanduser(data_dir)
    self.grid_size = grid_size
    self.random_rotation = random_rotation
    self.color_thresh = color_thresh

    # list the voxel grids
    self.filenames = []
    for filename in next(os.walk(data_dir))[-1]:
      if '_solid.npy' not in filename:
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
    x, y, z, c, xx, yy, zz = np.load(self.filenames[index])
    x, y, z = x.astype(int), y.astype(int), z.astype(int)
    pts = np.vstack((xx, yy, zz))
    offset = (pts.max(1, keepdims=True) + pts.min(1, keepdims=True)) / 2
    pts -= offset
    scale = max(pts.max(1) - pts.min(1)) / 2
    pts /= scale
    pts = np.vstack((np.ones(pts.shape[1]), pts, scale*np.ones(pts.shape[1])))

    # center the object
    offset_x = (self.grid_size - x.max() - 1) // 2
    offset_y = (self.grid_size - y.max() - 1) // 2
    offset_z = (self.grid_size - z.max() - 1) // 2
    x += offset_x
    y += offset_y
    z += offset_z

    # random rotation
    if abs(self.random_rotation) > 0:
      theta = np.random.uniform(-np.pi*self.random_rotation/180,
        np.pi*self.random_rotation/180)
      R = txe.euler2mat(0, 0, theta)
      p = np.vstack((x, y, z)) + 0.5
      p = p - self.grid_size / 2.0
      p = R @ p
      s = max(p.max(1) - p.min(1))
      p = p * (self.grid_size-1) / s
      s = (p.max(1, keepdims=True) + p.min(1, keepdims=True)) / 2.0
      p = p + self.grid_size/2.0 - s
      x, y, z = (p - 0.5).astype(int)

    # create occupancy grid
    geom = np.zeros((5, self.grid_size, self.grid_size, self.grid_size))
    geom[:, z, y, x] = pts

    # process texture
    c = utils.discretize_texture(c, thresh=self.color_thresh)
    tex = 2 * np.ones((self.grid_size, self.grid_size, self.grid_size))
    tex[z, y, x] = c

    return geom.astype(np.float32), tex.astype(np.int)


if __name__ == '__main__':
  import open3d
  from utils import show_pointcloud
  dset = VoxelDataset(osp.join('data', 'voxelized_meshes'), 'use',
    train=True, random_rotation=180)
  for idx in np.random.choice(len(dset), 10):
    geom, tex = dset[idx]
    z, y, x = np.nonzero(geom[0])
    c = tex[z, y, x]
    x3d = geom[1, z, y, x]
    y3d = geom[2, z, y, x]
    z3d = geom[3, z, y, x]
    show_pointcloud(np.vstack((x3d, y3d, z3d)).T, c)
    show_pointcloud(np.vstack((x,y,z)).T, c)
