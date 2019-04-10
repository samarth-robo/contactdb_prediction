import numpy as np
import os
import matplotlib.pyplot as plt
import open3d
from IPython.core.debugger import set_trace
import argparse
import logging
import utils
osp = os.path


def calculate_hand_lengths(sigmoid_a=0.05, color_thresh=0.4):
  DEBUG = False
  ignore_sessions = ['16', '19', '26']  # bad palm prints
  logger = logging.getLogger(__name__)
  data_dirs = utils.handoff_data_dirs
  # only use the sessions with the large blue plate
  session_nums = ['{:d}'.format(s) for s in range(8, 51)]
  
  # collect palm_print mesh filenames
  mesh_filenames = {}
  for session_num in session_nums:
    if session_num in ignore_sessions:
      continue
    session_name = 'full{:s}_handoff'.format(session_num)
    data_dir = data_dirs[int(session_num)-1]
    mfs = utils.get_session_mesh_filenames(session_name, data_dir)
    if 'palm_print' in mfs:
      mesh_filenames[session_num] = mfs['palm_print']

  hand_lengths = {}
  for session_num, mesh_filename in mesh_filenames.items():
    m = open3d.read_triangle_mesh(mesh_filename)
    tex = np.asarray(m.vertex_colors)[:, 0]
    zs  = np.asarray(m.vertices)[:, 2]
    tex = utils.texture_proc(tex, a=sigmoid_a, invert=(session_num == '14'))
    tex = utils.discretize_texture(tex, thresh=color_thresh).astype(int)
    tex[tex == 2] = 0
    if DEBUG:
      print(session_num)
      tex_show = np.tile(tex, (3, 1)).T
      tex_show = open3d.Vector3dVector(tex_show)
      m.vertex_colors = tex_show
      open3d.draw_geometries([m])

    zs = zs[tex > 0]
    hand_length = max(zs) - min(zs)
    hand_lengths[session_num] = hand_length * 100  # in cm

  return hand_lengths


def analyze_object(object_name, instruction, hand_lengths, data_dir):
  logger = logging.getLogger(__name__)

  # read list of bimanual grasps
  list_filename = osp.join(data_dir,
      '{:s}_bimanual_grasps.txt'.format(instruction))
  
  # sessions for which a bimanual grasp occurred for this object
  bimanual_grasps = []
  with open(list_filename, 'r') as f:
    for line in f:
      line = line.strip()
      o_name = '_'.join(line.split('_')[:-2])
      session_num = line.split('_')[-2]
      session_num = session_num.replace('full', '')
      if object_name in o_name:
        bimanual_grasps.append(session_num)
  bimanual_grasps = list(set(bimanual_grasps))

  # match with hand lengths
  out = []
  for session_num, hand_length in hand_lengths.items():
    bimanual = session_num in bimanual_grasps
    out.append((hand_length, bimanual))
  return out


def analyze_objects(object_names, instruction, hand_lengths, data_dir):
  plot_data = []
  for object_name in object_names:
    pdata = analyze_object(object_name, instruction, hand_lengths, data_dir)
    plot_data.append(pdata)
  object_disp_names = [n.replace('_large', '') for n in object_names]

  ebar_kwargs = {'elinewidth': 3, 'capsize': 5, 'capthick': 5}
  # create plot
  for y, x_data in enumerate(plot_data):
    x = [xx[0] for xx in x_data if not xx[1]]
    m = np.mean(x)
    s = np.std(x)
    if y == 0:
      plt.errorbar(m, y-0.1, xerr=s, fmt='rs', label='single-handed', **ebar_kwargs)
    else:
      plt.errorbar(m, y-0.1, xerr=s, fmt='rs', **ebar_kwargs)
    x = [xx[0] for xx in x_data if xx[1]]
    m = np.mean(x)
    s = np.std(x)
    if y == 0:
      plt.errorbar(m, y+0.1, xerr=s, fmt='go', label='bimanual', **ebar_kwargs)
    else:
      plt.errorbar(m, y+0.1, xerr=s, fmt='go', **ebar_kwargs)
  plt.ylim(-1, len(object_names))
  plt.yticks(np.arange(len(object_names)), object_disp_names, rotation=0,
      fontsize='x-large')
  plt.xticks(fontsize='x-large')
  plt.xlabel('hand length (cm)', fontsize='x-large')
  plt.legend(fontsize='x-large')
  plt.gca().invert_yaxis()
  plt.show()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--object_names', required=True, help='comma separated')
  parser.add_argument('--instruction', required=True)
  parser.add_argument('--data_dir', default=osp.join('~', 'deepgrasp_data'),
      help='directory containing lists of bimanual grasps')
  parser.add_argument('--sigmoid_a', default=0.05, type=float)
  parser.add_argument('--color_thresh', default=0.3)
  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO)
  color_thresh = float(args.color_thresh)

  # calculate the hand lengths
  hand_lengths = calculate_hand_lengths(args.sigmoid_a, color_thresh)

  # analyze objects
  object_names = args.object_names.split(',')
  analyze_objects(object_names, args.instruction, hand_lengths,
    osp.expanduser(args.data_dir))
