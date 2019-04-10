import numpy as np
import transforms3d.euler as txe
import os
import logging
import argparse
import utils
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import open3d
import pickle
from IPython.core.debugger import set_trace
from thirdparty import kmedoids
osp = os.path


def pointcloud_distance_matrix(pts, symmetric=False, max_corr_dist=5e-3):
  N = len(pts)
  D = np.zeros((N, N))
  yaw_step = 40 if symmetric else 359
  max_yaw = 360-yaw_step 

  # create KDTrees
  trees = [KDTree(p) for p in pts]

  for idx1 in range(N):
    pts1 = pts[idx1]
    tree1 = trees[idx1]
    print('Distance matrix {:d} / {:d}'.format(idx1*N, N*N))
    for idx2 in range(idx1):
      pts2 = pts[idx2]
      tree2 = trees[idx2]
      min_d = float('inf')
      for yaw in range(0, max_yaw, yaw_step):
        T = np.eye(4)
        T[:3, :3] = txe.euler2mat(0, 0, np.deg2rad(yaw))
        p2 = np.vstack((pts2.T, np.ones(len(pts2))))
        p2 = np.dot(T, p2)[:3].T
        if symmetric:
          tree2 = KDTree(p2)
        d12, _ = tree2.query(pts1, k=1)
        d21, _ = tree1.query(p2, k=1)
        d = np.sum(d12) + np.sum(d21)
        if d < min_d:
          min_d = d
      D[idx1, idx2] = d / (len(pts1) + len(pts2))
  D = D + D.T
  return D


def cluster(object_name, instruction, symmetric,
    color_thresh=0.4, max_points=150, compute_distances=True):
  logger = logging.getLogger(__name__)
  logger.info('Processing {:s}'.format(object_name))
  if symmetric:
    logger.info('{:s} is symmetric'.format(object_name))
  
  dmatrix_filename = osp.join('data', 'distance_matrices', '{:s}_{:s}_distances.pkl'.
      format(object_name, instruction))
  
  if compute_distances:
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
    
    # read mesh files and extract 3D points
    pts = []
    meshes = []
    for mesh_filename in mesh_filenames:
      m = open3d.read_triangle_mesh(mesh_filename)
      meshes.append(m)
      c = np.asarray(m.vertex_colors)[:, 0]
      c = utils.texture_proc(c, invert=('full14' in mesh_filename))
      idx = c > color_thresh
      p = np.asarray(m.vertices)[idx]
      if len(p) > max_points:
        choice = np.random.choice(len(p), size=max_points, replace=False)
        p = p[choice]
      pts.append(p)
    
    # compute distance matrix
    D = pointcloud_distance_matrix(pts, symmetric)
    with open(dmatrix_filename, 'wb') as f:
      pickle.dump({'D': D, 'mesh_filenames': mesh_filenames}, f)
    logger.info('{:s} saved'.format(dmatrix_filename))
  else:
    # k medoids clustering 
    with open(dmatrix_filename, 'rb') as f:
      d = pickle.load(f)
    D = d['D']
    mesh_filenames = d['mesh_filenames']
    M, C = kmedoids.cluster(D, 3)
    print('Cluster centers = ', C)
    print('Assignments = ', M)
    for label, c_center_idx in enumerate(C):
      print('### Cluster {:d} center: {:s}'.format(label, mesh_filenames[c_center_idx]))
      for member_idx in np.where(M==c_center_idx)[0]:
        print('Member: {:s}'.format(mesh_filenames[member_idx]))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--object_names', required=True)
  parser.add_argument('--symmetric', action='store_true')
  parser.add_argument('--instruction', required=True)
  parser.add_argument('--compute_distances', help='Compute distance matrices for clustering, and save them. If flag is not set, program uses those files to perform clustering and print cluster centers and membership')
  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO)
  object_names = args.object_names.split(',')
  for object_name in object_names:
    cluster(object_name, args.instruction, args.symmetric,
        compute_distances=args.compute_distances)
