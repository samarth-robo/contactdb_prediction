from models.voxnet import VoxNet
from models.pointnet import PointNetDenseCls as PointNet
from voxel_dataset import VoxelDataset
from pointcloud_dataset import PointCloudDataset
from models.losses import sMCLLoss
from utils import show_pointcloud

import numpy as np
import open3d
import matplotlib.pyplot as plt
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
    tex_pred = np.argmax(tex_pred, axis=0)
    tex_pred = cmap[tex_pred]
    show_pointcloud(pts, tex_pred)


def show_voxel_texture(geom, tex_preds):
  cmap = np.asarray([[0, 0, 1], [1, 0, 0]])
  z, y, x = np.nonzero(geom[0])
  pts = np.vstack((x, y, z)).T
  for tex_pred in tex_preds:
    tex_pred = np.argmax(tex_pred, axis=0)
    tex_pred = tex_pred[z, y, x]
    tex_pred = cmap[tex_pred]
    show_pointcloud(pts, tex_pred)


def eval(data_dir, instruction, checkpoint_dir, config_filename, device_id,
    show_object=None, save_preds=False):
  # config
  config = configparser.ConfigParser()
  config.read(config_filename)
  droprate = config['hyperparams'].getfloat('droprate')

  # cuda
  if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
  else:
    devices = os.environ['CUDA_VISIBLE_DEVICES']
    devices = devices.split(',')[device_id]
    os.environ['CUDA_VISIBLE_DEVICES'] = devices
  device = 'cuda:0'

  if checkpoint_dir[-1] == '/':
    checkpoint_dir = checkpoint_dir[:-1]
  checkpoint_filenames = []
  for filename in next(os.walk(checkpoint_dir))[-1]:
    if 'val_loss' not in filename:
      continue
    checkpoint_filenames.append(osp.join(checkpoint_dir, filename))
  n_ensemble = len(checkpoint_filenames)
  
  model_name = osp.split(config_filename)[1].split('.')[0]
  kwargs = dict(data_dir=data_dir, instruction=instruction, train=False,
      random_rotation=0)
  if 'voxnet' in model_name:
    models = [VoxNet(droprate=droprate) for _ in range(n_ensemble)]
    grid_size = config['hyperparams'].getint('grid_size')
    dset = VoxelDataset(grid_size=grid_size, **kwargs)
  elif 'pointnet' in model_name:
    models = [PointNet(droprate=droprate) for _ in range(n_ensemble)]
    n_points = config['hyperparams'].getint('n_points')
    dset = PointCloudDataset(n_points=n_points, random_scale=0, **kwargs)
  else:
    raise NotImplementedError

  # load checkpoints
  for model, checkpoint_filename in zip(models, checkpoint_filenames):
    checkpoint = torch.load(checkpoint_filename)
    model.load_state_dict(checkpoint.state_dict())
    print('Loaded model from {:s}'.format(checkpoint_filename))
    model.to(device=device)

  loss_fn = sMCLLoss(train=False, eval_mode=True)

  # eval loop!
  dloader = DataLoader(dset)
  losses = {}
  geoms_all = {}
  tex_preds_all = {}
  tex_targs_all = {}
  for batch_idx, batch in enumerate(dloader):
    object_name = '_'.join(dset.filenames[batch_idx].split('/')[-1].split('_')[2:-1])
    if show_object is not None:
      if object_name != show_object:
        continue
    if batch_idx % 10 == 0:
      print('{:d} / {:d}'.format(batch_idx+1, len(dloader)))
    geom, tex_targs = batch
    geom      = geom.to(device=device)
    tex_targs = tex_targs.to(device=device)
    tex_preds = []
    for model in models:
      model.eval()
      with torch.no_grad():
        tex_preds.append(model(geom))
    loss, min_idx = loss_fn(tex_preds, tex_targs)
    if object_name in losses:
      losses[object_name].append(loss.item())
    else:
      losses[object_name] = [loss.item()]

    geom      = geom.cpu().numpy().squeeze()
    tex_preds = [tex_pred.cpu().numpy().squeeze() for tex_pred in tex_preds]
    tex_targs = tex_targs.cpu().numpy().squeeze()

    if object_name not in geoms_all:
      geoms_all[object_name] = geom
    if object_name not in tex_preds_all:
      tex_preds_all[object_name] = tex_preds
    if object_name not in tex_targs_all:
      tex_targs_all[object_name] = [tex_targs]
    else:
      tex_targs_all[object_name].append(tex_targs)
	
    if show_object is not None:
      if 'pointnet' in model_name:
        show_pointcloud_texture(geom, tex_preds)
      elif 'voxnet' in model_name:
        show_voxel_texture(geom, tex_preds)
      else:
        raise NotImplementedError
      break
  
  for object_name, object_losses in losses.items():
    print('{:s} loss = {:.4f}'.format(object_name, np.mean(object_losses)))
  
  if save_preds:
    output_data = {
        'checkpoint_dir': checkpoint_dir,
        'geoms': geoms_all,
        'tex_preds': tex_preds_all,
        'tex_targs': tex_targs_all}
    output_filename = '{:s}_{:s}_smcl_preds.pkl'.format(instruction, model_name)
    with open(output_filename, 'wb') as f:
      pickle.dump(output_data, f)
    print('{:s} saved'.format(output_filename))
	

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', default=osp.join('data', 'voxelized_meshes'))
  parser.add_argument('--instruction', required=True)
  parser.add_argument('--checkpoint_dir', required=True)
  parser.add_argument('--config_filename', required=True)
  parser.add_argument('--show_object', default=None)
  parser.add_argument('--device_id', default=0)
  args = parser.parse_args()

  eval(osp.expanduser(args.data_dir), args.instruction,
    osp.expanduser(args.checkpoint_dir), osp.expanduser(args.config_filename),
    args.device_id, show_object=args.show_object)
