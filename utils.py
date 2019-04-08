import numpy as np
import open3d
import os
osp = os.path

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


def texture_proc(colors, a=0.05, invert=False):
  idx = colors > 0
  ci = colors[idx]
  if len(ci) == 0:
    return colors
  if invert:
    ci = 1 - ci
  # fit a sigmoid
  x1 = min(ci); y1 = a
  x2 = max(ci); y2 = 1-a
  lna = np.log((1 - y1) / y1)
  lnb = np.log((1 - y2) / y2)
  k = (lnb - lna) / (x1 - x2)
  mu = (x2*lna - x1*lnb) / (lna - lnb)
  # apply the sigmoid
  ci = np.exp(k * (ci-mu)) / (1 + np.exp(k * (ci-mu)))
  colors[idx] = ci
  return colors


def discretize_texture(c, thresh=0.4):
  idx = c > 0
  if sum(idx) == 0:
    return c
  ci = c[idx]
  c[:] = 2
  ci = ci > thresh
  c[idx] = ci
  return c


def show_pointcloud(pts, colors,
    cmap=np.asarray([[0,0,1],[1,0,0],[0,0,1]])):
  colors = np.asarray(colors)
  if (colors.dtype == int) and (colors.ndim == 1) and (cmap is not None):
    colors = cmap[colors]
  if colors.ndim == 1:
    colors = np.tile(colors, (3, 1)).T

  pc = open3d.PointCloud()
  pc.points = open3d.Vector3dVector(np.asarray(pts))
  pc.colors = open3d.Vector3dVector(colors)

  open3d.draw_geometries([pc])
