from models.voxnet import DiverseVoxNet as VoxNet
from models.pointnet import DiversePointNet as PointNet
from voxel_dataset import VoxelDataset
from pointcloud_dataset import PointCloudDataset
from models.losses import DiverseLoss

import numpy as np
import open3d
import os
import torch
from torch.utils.data import DataLoader
import argparse
import configparser
import pickle
from IPython.core.debugger import set_trace
osp = os.path

def show_pointcloud_texture(geom, tex_preds):
  cmap = np.asarray([[0, 0, 1], [1, 0, 0]])
  x, y, z, scale = geom
  pts = np.vstack((x, y, z)).T * scale[0]
  for tex_pred in tex_preds:
    pc = open3d.PointCloud()
    pc.points = open3d.Vector3dVector(pts)
    tex_pred = np.argmax(tex_pred, axis=0)
    tex_pred = cmap[tex_pred]
    pc.colors = open3d.Vector3dVector(tex_pred)
    open3d.draw_geometries([pc])


def show_voxel_texture(geom, tex_preds):
  cmap = np.asarray([[0, 0, 1], [1, 0, 0]])
  z, y, x = np.nonzero(geom[0])
  pts = np.vstack((x, y, z)).T
  for tex_pred in tex_preds:
    tex_pred = np.argmax(tex_pred, axis=0)
    tex_pred = tex_pred[z, y, x]
    tex_pred = cmap[tex_pred]
    pc = open3d.PointCloud()
    pc.points = open3d.Vector3dVector(pts)
    pc.colors = open3d.Vector3dVector(tex_pred)
    open3d.draw_geometries([pc])


def eval(data_dir, instruction, checkpoint_filename, config_filename, device_id,
    test_only=False, show_object=None, save_preds=False):
  # config
  config = configparser.ConfigParser()
  config.read(config_filename)
  droprate  = config['hyperparams'].getfloat('droprate')

  # cuda
  if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
  else:
    devices = os.environ['CUDA_VISIBLE_DEVICES']
    devices = devices.split(',')[device_id]
    os.environ['CUDA_VISIBLE_DEVICES'] = devices
  device = 'cuda:0'

  # load checkpoint
  checkpoint = torch.load(checkpoint_filename)

  # create model
  model_name = osp.split(config_filename)[1].split('.')[0]
  kwargs = dict(data_dir=data_dir, instruction=instruction, train=False,
    random_rotation=0, n_ensemble=-1, test_only=test_only)
  if 'voxnet' in model_name:
    model = VoxNet(n_ensemble=checkpoint.n_ensemble, droprate=droprate)
    model.voxnet.load_state_dict(checkpoint.voxnet.state_dict())
    grid_size = config['hyperparams'].getint('grid_size')
    dset = VoxelDataset(grid_size=grid_size, **kwargs)
  elif 'pointnet' in model_name:
    model = PointNet(n_ensemble=checkpoint.n_ensemble, droprate=droprate)
    model.pointnet.load_state_dict(checkpoint.pointnet.state_dict())
    n_points = config['hyperparams'].getint('n_points')
    dset = PointCloudDataset(n_points=n_points, random_scale=0, **kwargs)
  else:
    raise NotImplementedError
  if 'pointnet' not in model_name:
    model.eval()
  model.to(device=device)

  loss_fn = DiverseLoss(train=False, eval_mode=True)

  # eval loop!
  dloader = DataLoader(dset)
  for batch_idx, batch in enumerate(dloader):
    object_name = list(dset.filenames.keys())[batch_idx]
    if show_object is not None:
      if object_name != show_object:
        continue
    geom, tex_targs = batch
    geom = geom.to(device=device)
    tex_targs = tex_targs.to(device=device)
    with torch.no_grad():
      tex_preds = model(geom)

    loss, match_indices = loss_fn(tex_preds, tex_targs)
    print('{:s} loss = {:.4f}'.format(object_name, loss.item()))

    geom      = geom.cpu().numpy().squeeze()
    tex_preds = tex_preds.cpu().numpy().squeeze()
    match_indices = match_indices.cpu().numpy().squeeze()
    tex_targs = tex_targs.cpu().numpy().squeeze()

    if (save_preds):
      output_data = {
          'checkpoint_filename': checkpoint_filename,
          'geom': geom,
          'tex_preds': tex_preds,
          'match_indices': match_indices,
          'tex_targs': tex_targs}
      output_filename = '{:s}_{:s}_{:s}_diversenet_preds.pkl'.format(object_name,
          instruction, model_name)
      with open(output_filename, 'wb') as f:
        pickle.dump(output_data, f)
      print('{:s} saved'.format(output_filename))

    if show_object is not None:
      if 'pointnet' in model_name:
        show_pointcloud_texture(geom, tex_preds)
      elif 'voxnet' in model_name:
        show_voxel_texture(geom, tex_preds)
        break
      else:
        raise NotImplementedError


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', default=osp.join('data', 'voxelized_meshes'))
  parser.add_argument('--instruction', required=True)
  parser.add_argument('--checkpoint_filename', required=True)
  parser.add_argument('--config_filename', required=True)
  parser.add_argument('--test_only', action='store_true')
  parser.add_argument('--device_id', default=0)
  parser.add_argument('--show_object', default=None)
  args = parser.parse_args()

  eval(osp.expanduser(args.data_dir), args.instruction,
    osp.expanduser(args.checkpoint_filename),
    osp.expanduser(args.config_filename), args.device_id,
    test_only=args.test_only, show_object=args.show_object)
