import numpy as np
import os
import logging
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
  """
  output: 0: no contact, 1: contact, 2: unknown
  """
  idx = c > 0
  ci = c[idx]
  c[:] = 2
  ci = ci > thresh
  c[idx] = ci
  return c


def get_session_mesh_filenames(session_name, data_dir):
  logger = logging.getLogger(__name__)
  base_dir = osp.join(data_dir, session_name)

  # determine the directories from which to take meshes
  object_dirs = {}
  for object_dir in next(os.walk(base_dir))[1]:
    object_dir = osp.join(base_dir, object_dir)
    object_name_filename = osp.join(object_dir, 'object_name.txt')
    try:
      with open(object_name_filename, 'r') as f:
        object_name = f.readline().strip()
    except IOError:
      # logger.warning('Skipping {:s}'.format(object_dir))
      continue
    if object_name not in object_dirs:
      object_dirs[object_name] = [object_dir]
    else:
      object_dirs[object_name].append(object_dir)

  mesh_filenames = {}
  for oname, odirs in object_dirs.items():
    if len(odirs) == 1:
      mesh_filename = osp.join(odirs[0], 'thermal_images',
        '{:s}_textured.ply'.format(oname))
    else:  # find which one has merge.txt
      merge_dirs = [od for od in odirs if osp.isfile(osp.join(od, 'merge.txt'))]
      # determine which directory has the merged mesh
      for d in merge_dirs:
        with open(osp.join(d, 'merge.txt'), 'r') as f:
          m = f.readline()
        if len(m):
          merge_dir = d
          break
      else:
        logger.error('No directory is marked as destination for {:s} {:s}'.
            format(session_name, oname))
        raise IOError
      mesh_filename = osp.join(merge_dir, 'thermal_images',
        '{:s}_textured.ply'.format(oname))
    mesh_filenames[oname] = mesh_filename
  return mesh_filenames

handoff_objects = [
		'airplane',
		'alarm_clock',
		'apple',
		'banana',
		'binoculars',
		'bowl',
		'camera',
		'cell_phone',
		'cube_small',
		'cube_medium',
		'cube_large',
		'cup',
		'cylinder_small',
		'cylinder_medium',
		'cylinder_large',
		'elephant',
		'eyeglasses',
		'flashlight',
		'flute',
		'hammer',
		'headphones',
		'knife',
		'light_bulb',
		'mouse',
		'mug',
		'pan',
		'piggy_bank',
		'ps_controller',
		'pyramid_small',
		'pyramid_medium',
		'pyramid_large',
		'rubber_duck',
		'scissors',
		'sphere_small',
		'sphere_medium',
		'sphere_large',
		'stanford_bunny',
		'stapler',
		'toothbrush',
		'toothpaste',
		'torus_small',
		'torus_medium',
		'torus_large',
		'train',
		'utah_teapot',
		'water_bottle',
		'wine_glass',
		'wristwatch',
]

use_objects = [
		'apple',
		'banana',
		'binoculars',
		'bowl',
		'camera',
		'cell_phone',
		'cup',
		'door_knob',
		'eyeglasses',
		'flashlight',
		'flute',
		'hammer',
		'hand',
		'headphones',
		'knife',
		'light_bulb',
		'mouse',
		'mug',
		'pan',
		'ps_controller',
		'scissors',
		'stapler',
		'toothbrush',
		'toothpaste',
		'utah_teapot',
		'water_bottle',
		'wine_glass',
]

object_names = list(set(handoff_objects) | set(use_objects))

handoff_bimanual_objects = [
		'binoculars',
		'bowl',
		'cube_large',
		'cylinder_large',
		'eyeglasses',
		'headphones',
		'piggy_bank',
		'ps_controller',
		'pyramid_large',
		'sphere_large',
		'utah_teapot',
]

use_bimanual_objects = [
		'banana',
		'binoculars',
		'bowl',
		'camera',
		'eyeglasses',
		'flute',
		'headphones',
		'ps_controller',
    'utah_teapot',
		'water_bottle',
]
