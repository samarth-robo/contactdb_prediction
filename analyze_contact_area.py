import cv2
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import open3d
from IPython.core.debugger import set_trace
import argparse
import logging
import utils
osp = os.path


def calculate_contact_area_object(object_name, instruction, sigmoid_a=0.05,
    color_thresh=0.4):
  logger = logging.getLogger(__name__)
	# get mesh filenames
  data_dirs = utils.handoff_data_dirs if instruction=='handoff'\
      else utils.use_data_dirs
  mesh_filenames = []
  session_names = []
  for session_idx, data_dir in enumerate(data_dirs):
    session_name = 'full{:d}_{:s}'.format(session_idx+1, instruction)
    mfs = utils.get_session_mesh_filenames(session_name, data_dir)
    if object_name in mfs:
      mesh_filenames.append(mfs[object_name])
      session_names.append(session_name)

  # uncomment for calculating fingertip area
  # mesh_filenames = []
  # data_dir = osp.join('data', 'palm_prints')
  # for mfn in next(os.walk(data_dir))[-1]:
  #   mesh_filenames.append(osp.join(data_dir, mfn))
  logger.info('Found {:d} meshes for {:s}'.format(len(mesh_filenames), object_name))
 
  contact_areas = []
  for idx, mesh_filename in enumerate(mesh_filenames):
    if idx % 10 == 0:
      logger.info('{:d} / {:d}'.format(idx+1, len(mesh_filenames)))
    # read mesh
    m = open3d.read_triangle_mesh(mesh_filename)
    tris = np.asarray(m.triangles)
    verts = np.asarray(m.vertices)
    tex = np.asarray(m.vertex_colors)[:, 0]
    tex = utils.texture_proc(tex, a=sigmoid_a, invert=('full14' in mesh_filename))
    tex = utils.discretize_texture(tex, thresh=color_thresh).astype(int)
    tex[tex == 2] = 0 

    # filter triangles by texture
    tris_tex = tex[tris.ravel()].reshape(tris.shape)
    tris_tex = np.max(tris_tex, 1)
    tris = tris[tris_tex == 1]

    # calculate area of triangles
    A = verts[tris[:, 0]] * 100
    B = verts[tris[:, 1]] * 100
    C = verts[tris[:, 2]] * 100
    areas = np.cross(B-A, C-A, axis=1)
    areas = 0.5 * np.linalg.norm(areas, axis=1).sum()
    # logger.info('{:s}: {:f}'.format(mesh_filename, areas))
    contact_areas.append(areas)
  contact_area = np.mean(contact_areas)
  logger.info('{:s} contact_area = {:f} cm^2'.format(object_name,
    contact_area))
  return contact_area


def calculate_contact_area_objects(object_names, instruction, sigmoid_a,
    color_thresh):
  logger = logging.getLogger(__name__)
  contact_areas = \
      [calculate_contact_area_object(object_name, instruction,
        sigmoid_a=sigmoid_a, color_thresh=color_thresh)
        for object_name in object_names]
  object_names.append('fingertips')
  contact_areas.append(16.745952)
  output_filename = osp.join('data',
      '{:s}_contact_areas.pkl'.format(instruction))
  with open(output_filename, 'wb') as f:
    pickle.dump({'object_names': object_names, 'contact_areas': contact_areas},
        f)
  logger.info('{:s} written'.format(output_filename))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--object_name', default=None)
  parser.add_argument('--instruction', required=True)
  parser.add_argument('--sigmoid_a', default=0.05)
  parser.add_argument('--color_thresh', default=0.4)
  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO)
  sigmoid_a = float(args.sigmoid_a)
  color_thresh = float(args.color_thresh)
  if args.object_name is not None:
    calculate_contact_area_object(args.object_name, args.instruction,
      sigmoid_a=sigmoid_a, color_thresh=color_thresh)
  else:
    objects = getattr(utils, '{:s}_objects'.format(args.instruction))
    calculate_contact_area_objects(objects, args.instruction,
      sigmoid_a=sigmoid_a, color_thresh=color_thresh)
