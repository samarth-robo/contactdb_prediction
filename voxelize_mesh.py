import open3d
import numpy as np
import os
import logging
import argparse
import utils
from thirdparty import binvox_rw
from IPython.core.debugger import set_trace
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


def voxelize_mesh(session_name, data_dir, output_dir, models_dir,
    hollow=False, sigmoid_a=0.05, debug_mode=False, test_only=False):
  logger = logging.getLogger(__name__)
  base_dir = osp.join(data_dir, session_name)

  # determine the directories from which to take meshes
  object_dirs = {}
  for object_dir in next(os.walk(base_dir))[1]:
    object_dir = osp.join(base_dir, object_dir)
    if test_only:  # basically skip this (and the next) whole loop
      if 'v2' not in object_dir:
        continue
    object_name_filename = osp.join(object_dir, 'object_name.txt')
    try:
      with open(object_name_filename, 'r') as f:
        object_name = f.readline().strip()
    except IOError:
      logger.warning('Skipping {:s}'.format(object_dir))
      continue
    if object_name == 'palm_print':
      continue
    if object_name not in object_dirs:
      object_dirs[object_name] = [object_dir]
    else:
      object_dirs[object_name].append(object_dir)

  mesh_filenames = []
  output_filenames = []
  vox_filenames = []
  suffix = 'hollow' if hollow else 'solid'
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
    mesh_filenames.append(mesh_filename)
    output_filename = osp.join(output_dir,
    '{:s}_{:s}_{:s}.npy'.format(session_name, oname, suffix))
    output_filenames.append(output_filename)
    vox_filename = osp.join(models_dir, '{:s}.binvox'.format(oname))
    vox_filenames.append(vox_filename)

  if test_only:
    for vox_filename in next(os.walk(models_dir))[-1]:
      if ('v2' not in vox_filename) or ('binvox' not in vox_filename):
        continue
      if 'old'in vox_filename:
        continue
      vox_filename = osp.join(models_dir, vox_filename)
      vox_filenames.append(vox_filename)
      mesh_filenames.append(vox_filename.replace('binvox', 'ply'))
      object_name = vox_filename.split('.')[0].split('/')[-1]
      output_filename = osp.join(output_dir, 'testonly_{:s}_{:s}.npy'.
          format(object_name, suffix))
      output_filenames.append(output_filename)

  for mesh_filename, output_filename, vox_filename in \
      zip(mesh_filenames, output_filenames, vox_filenames):
    m = open3d.read_triangle_mesh(mesh_filename)
    if hollow:  # pointcloud -- can directly take points and colors
      ps_vox = np.asarray(m.vertices)
      x = y = z = -np.ones(len(ps_vox))
      if test_only:
        colors_vox = np.zeros(len(ps_vox))
      else:
        colors_vox = np.asarray(m.vertex_colors)[:, 0]
        colors_vox = utils.texture_proc(colors_vox, sigmoid_a,
            invert=('full14' in mesh_filename))
    else:  # voxelgrid
      with open(vox_filename, 'rb') as f:
        mv = binvox_rw.read_as_3d_array(f)
      mv_np = mv.data.copy()
      x, y, z = np.where(mv_np)
      ps_vox = get_3d_points(mv_np, mv.scale, mv.translate)
      if test_only:
        colors_vox = np.zeros(len(ps_vox))
      else:
        # need to match voxels to points on the mesh and transfer colors
        # read texture of mesh
        colors_m = np.asarray(m.vertex_colors)[:, 0]
        colors_m = utils.texture_proc(colors_m, sigmoid_a,
            invert=('full14' in mesh_filename))
        tree = open3d.KDTreeFlann(m)

        # find correspondences
        pitch = mv.scale / mv.dims[0]
        nbr_idxs = []
        for p_vox in ps_vox:
          _, nidx, _ = tree.search_radius_vector_3d(p_vox, pitch)
          nbr_idxs.append(np.asarray(nidx))

        # transfer colors
        colors_vox = np.zeros(len(ps_vox))
        for idx, nbrs in enumerate(nbr_idxs):
          cs = colors_m[nbrs]
          ci = cs > 0
          if sum(ci) > 0:
            colors_vox[idx] = np.mean(cs[ci])

    if debug_mode:
      if hollow:
        utils.show_pointcloud(ps_vox, colors_vox)
      else:
        utils.show_pointcloud(np.vstack((x, y, z)).T, colors_vox)
    else:
      try:
        np.save(output_filename, np.vstack((x, y, z, colors_vox, ps_vox.T)))
        logger.info('Written {:s}'.format(output_filename))
      except:
        logger.error('Could not write {:s}'.format(output_filename))
        raise IOError


def voxelize_all_meshes(session_nums, instruction, output_dir, models_dir, hollow,
    sigmoid_a, debug_mode, test_only):
  data_dirs = getattr(utils, '{:s}_data_dirs'.format(instruction))
  for session_num in session_nums:
    session_name = 'full{:s}_{:s}'.format(session_num, instruction)
    data_dir = data_dirs[int(session_num)-1]
    voxelize_mesh(session_name, data_dir, output_dir, models_dir, hollow,
        sigmoid_a, debug_mode, test_only)
    if test_only:
      break


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--session_nums', default=None)
  parser.add_argument('--instruction', required=True)
  parser.add_argument('--output_dir',
      default=osp.join('data', 'voxelized_meshes'))
  parser.add_argument('--models_dir',
    default=osp.join('data', 'object_models'))
  parser.add_argument('--hollow', action='store_true')
  parser.add_argument('--sigmoid_a', default=0.05, type=float)
  parser.add_argument('--test_only', action='store_true',
      help='use for objects without textured meshes, that only need to be tested')
  parser.add_argument('--debug', action='store_true')
  args = parser.parse_args()

  session_nums = args.session_nums
  if session_nums is not None:
    if '-' in session_nums:
      start, end = session_nums.split('-')
      session_nums = ['{:d}'.format(i) for i in range(int(start), int(end)+1)]
    else:
      session_nums = session_nums.split(',')
  else:
    session_nums = ['{:d}'.format(i) for i in range(1, 51)]
  voxelize_all_meshes(session_nums, args.instruction, osp.expanduser(args.output_dir),
      osp.expanduser(args.models_dir), args.hollow, args.sigmoid_a, args.debug,
      args.test_only)
