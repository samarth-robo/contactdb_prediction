import open3d
import subprocess
import numpy as np
import os
import logging
import argparse
from thirdparty import binvox_rw
osp = os.path


def get_3d_points(mv, scale, translate):
  i,j,k = np.nonzero(mv)
  x_n = (i + .5) / mv.shape[0]
  y_n = (j + .5) / mv.shape[1]
  z_n = (k + .5) / mv.shape[2]
  x = scale * x_n + translate[0]
  y = scale * y_n + translate[1]
  z = scale * z_n + translate[2]
  return np.vstack((x, y, z)).T


def generate_binvoxes(models_dir, N_voxels, hollow, debug_mode=False):
  logger = logging.getLogger(__name__)

  mesh_filenames = [osp.join(models_dir, fn)
    for fn in next(os.walk(models_dir))[-1] if '.ply' in fn]
  output_filenames = []
  suffix = 'hollow' if hollow else 'solid'
  for fn in mesh_filenames:
    output_filename = '{:s}_{:s}.binvox'.\
      format(fn.split('/')[-1].split('.')[0], suffix)
    output_filenames.append(osp.join(models_dir, output_filename))

  for mesh_filename, output_filename in zip(mesh_filenames, output_filenames):
    # voxelize mesh
    args = osp.join('thirdparty', 'binvox')
    if hollow:
      args += ' -ri -e'
    args += ' -pb -d {:d} '.format(N_voxels)
    args += mesh_filename

    vox_filename = mesh_filename.split('.')[0] + '.binvox'
    if osp.isfile(vox_filename):
      os.remove(vox_filename)
    try:
      subprocess.check_call(args, shell=True)
    except subprocess.CalledProcessError as e:
      print(e)

    os.rename(vox_filename, output_filename)
    logger.info('Written {:s}'.format(output_filename))

    if debug_mode:
      with open(output_filename, 'rb') as f:
        mv = binvox_rw.read_as_3d_array(f)
      x, y, z = np.where(mv.data)
      pc = open3d.PointCloud()
      pc.points = open3d.Vector3dVector(np.vstack((x, y, z)).T)
      open3d.draw_geometries([pc])


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--models_dir',
    default=osp.join('~', 'deepgrasp_data', 'models'))
  parser.add_argument('--hollow', action='store_true')
  parser.add_argument('--N_voxels', default=64, type=int,
      help='Size of voxelgrid used by the ML algorithm')
  parser.add_argument('--debug', action='store_true')
  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO)
  generate_binvoxes(osp.expanduser(args.models_dir), N_voxels=args.N_voxels,
      hollow=hollow, debug_mode=args.debug)
