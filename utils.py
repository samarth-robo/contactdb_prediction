import numpy as np
import open3d
import os.path as osp

test_objects = ['mug', 'pan', 'wine_glass']

base_dir = osp.expanduser(osp.join('~', 'deepgrasp_data'))
use_data_dirs = \
    [osp.join(base_dir, 'data')]*28 + \
    [osp.join(base_dir, 'data3')] + \
    [osp.join(base_dir, 'data2')]*4 + \
    [osp.join(base_dir, 'data3')]*17
handoff_data_dirs = \
    [osp.join(base_dir, 'data')]*24 + \
    [osp.join(base_dir, 'data3')]*5 +\
    [osp.join(base_dir, 'data2')]*4 + \
    [osp.join(base_dir, 'data3')]*17

def texture_proc(c, k, max_frac=0.75, invert=False):
  idx = c > 0
  if sum(idx) == 0:
    return c
  ci = c[idx]
  if invert:
    ci = 1 - ci
  m = max_frac * np.max(ci)
  ci = np.exp(k*(ci-m)) / (1 + np.exp(k*(ci-m)))
  c[idx] = ci
  return c

def discretize_texture(c, thresh=0.4):
  idx = c > 0
  if sum(idx) == 0:
    return c
  ci = c[idx]
  c[:] = 2
  ci = ci > thresh
  c[idx] = ci
  return c

def show_pointcloud(pts, colors, cmap=np.asarray([[0,0,1],[1,0,0],[0,0,1]])):
  colors = np.asarray(colors)
  if (colors.dtype == int) and (colors.ndim == 1) and (cmap is not None):
    colors = cmap[colors]
  if colors.ndim == 1:
    colors = np.tile(colors, (3, 1)).T

  pc = open3d.PointCloud()
  pc.points = open3d.Vector3dVector(np.asarray(pts))
  pc.colors = open3d.Vector3dVector(colors)

  open3d.draw_geometries([pc])
