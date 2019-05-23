import os
import open3d
import numpy as np
from utils import texture_proc
import transforms3d.euler as txe
from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt
import pickle
osp = os.path


def animate(geomi, suffix=None):
  T = np.eye(4)
  T[:3, :3] = txe.euler2mat(np.deg2rad(-90), np.deg2rad(0), np.deg2rad(0))
  geom.transform(T)

  animate.count = -1
  animate.step = 50.0
  animate.radian_per_pixel = 0.003

  def move_forward(vis):
    glb = animate
    ctr = vis.get_view_control()
    ro = vis.get_render_option()
    ro.point_size = 25.0
    if glb.count >= 0:
      image = vis.capture_screen_float_buffer(False)
      im_filename = osp.join('animation_images',
          'image_{:03d}'.format(glb.count))
      if suffix is not None:
        im_filename = '{:s}_{:s}'.format(im_filename, str(suffix))
      im_filename += '.png'
      plt.imsave(im_filename, np.asarray(image))

      if np.rad2deg(glb.radian_per_pixel * glb.step * glb.count) >= 360.0:
        vis.register_animation_callback(None)

      ctr.rotate(glb.step, 0)
    else:
      ctr.scale(10)
    glb.count += 1

  open3d.draw_geometries_with_animation_callback([geom], move_forward)


if __name__ == '__main__':
  if True:
    filename = 'data/contactdb_contactmaps/42_handoff_cylinder_large.ply'
    filename = osp.expanduser(filename)
    geom = open3d.read_triangle_mesh(filename)
    geom.compute_vertex_normals()
    geom.compute_triangle_normals()

    c = np.asarray(geom.vertex_colors)[:, 0]
    c = texture_proc(c)
    c = plt.cm.inferno(c)[:, :3]
    geom.vertex_colors = open3d.Vector3dVector(c)
    animate(geom)
  else:
    filename = 'data/contactdb_predictions/camerav2_use_voxnet_diversenet_preds.pkl'
    filename = osp.expanduser(filename)
    with open(filename, 'rb') as f:
      d = pickle.load(f)
    cmap = np.asarray([[0, 0, 1], [1, 0, 0]])
    z, y, x = np.nonzero(d['geom'][0])
    pts = np.vstack((x, y, z)).T
    pred_idxs = [0, 4, 9]
    #pred_idxs = range(len(d['tex_preds']))
    for pred_idx in pred_idxs:
      print(pred_idx)
      tex_pred = np.argmax(d['tex_preds'][pred_idx], axis=0)
      tex_pred = tex_pred[z, y, x]
      tex_pred = cmap[tex_pred]
      geom = open3d.PointCloud()
      geom.points = open3d.Vector3dVector(pts)
      geom.colors = open3d.Vector3dVector(tex_pred)
      animate(geom, pred_idx)
      #open3d.draw_geometries([geom])
