import numpy as np
import os
import logging
import argparse
import utils
from scipy.spatial import KDTree
import open3d
from IPython.core.debugger import set_trace
osp = os.path


def query_touch(mesh_filename, mesh, tree, query_pts, sigmoid_a, color_thresh):
  """
  returns whether the query region is touched in the mesh
  """
  tex = np.asarray(mesh.vertex_colors)[:, 0]
  tex = utils.texture_proc(tex, a=sigmoid_a, invert=('full14' in mesh_filename))
  nbr_texs = []
  for query_pt in query_pts:
    _, nbr_idx, _ = tree.search_radius_vector_3d(query_pt, 5e-3)
    nbr_tex = tex[nbr_idx]
    valid_idx = nbr_tex > 0
    nbr_tex = np.mean(nbr_tex[valid_idx]) if sum(valid_idx) > 0 else 0
    # nbr_tex = np.max(nbr_tex)
    nbr_texs.append(nbr_tex)
  nbr_texs = np.asarray(nbr_texs)
  touched = nbr_texs.max() > color_thresh
  return touched


def analyze_active_areas(object_name, instruction, active_areas_dir,
    sigmoid_a=0.05, color_thresh=0.6):
  logger = logging.getLogger(__name__)
  logger.info('Processing {:s}'.format(object_name))
  
  # get mesh filenames
  data_dirs = utils.handoff_data_dirs if instruction=='handoff'\
      else utils.use_data_dirs
  mesh_filenames = []
  for session_idx, data_dir in enumerate(data_dirs):
    session_name = 'full{:d}_{:s}'.format(session_idx+1, instruction)
    mfs = utils.get_session_mesh_filenames(session_name, data_dir)
    if object_name in mfs:
      mesh_filenames.append(mfs[object_name])
  logger.info('Found {:d} meshes for {:s}'.format(len(mesh_filenames), object_name))
  meshes = [open3d.read_triangle_mesh(mfn) for mfn in mesh_filenames]
  print('Computing KD-Trees...')
  trees  = [open3d.KDTreeFlann(mesh) for mesh in meshes]
  print('Done')

  # read query points
  query_filenames = [fn for fn in next(os.walk(active_areas_dir))[-1]
      if ('.ply' in fn) and (object_name in fn)]
  if len(query_filenames) > 1:
    query_filenames.append('+'.join(query_filenames))
  for query_filename in query_filenames:
    if '+' in query_filename:
      query_pts = []
      for qfn in query_filename.split('+'):
        qm = open3d.read_triangle_mesh(osp.join(active_areas_dir, qfn))
        query_pts.append(np.asarray(qm.vertices))
      query_pts = np.vstack(query_pts)
    else:
      qm = open3d.read_triangle_mesh(osp.join(active_areas_dir, query_filename))
      query_pts = np.asarray(qm.vertices)
    touched = []
    for idx, (mesh_filename, mesh, tree) in enumerate(zip(mesh_filenames,
      meshes, trees)):
      print('{:d} / {:d}'.format(idx, len(meshes)))
      touched.append(query_touch(mesh_filename, mesh, tree, query_pts, sigmoid_a,
        color_thresh))
    touched = np.asarray(touched, dtype=float)
    logger.info(' {:.2f} % touched {:s}'.format(100.0 * np.mean(touched), query_filename))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--object_name', required=True)
  parser.add_argument('--instruction', required=True)
  parser.add_argument('--sigmoid_a', default=0.05, type=float)
  parser.add_argument('--active_areas_dir',
      default=osp.join('data', 'active_areas'))
  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO)
  analyze_active_areas(args.object_name, args.instruction,
      osp.expanduser(args.active_areas_dir), sigmoid_a=args.sigmoid_a)
