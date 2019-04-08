from models.voxnet import VoxNet
from models.pointnet import PointNetDenseCls as PointNet
from models.losses import sMCLLoss
from voxel_dataset import VoxelDataset
from pointcloud_dataset import PointCloudDataset

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
import visdom
from torch.utils.data import DataLoader
from torch import optim as toptim
import numpy as np
import os
import torch
import configparser
import argparse
import logging
from IPython.core.debugger import set_trace
osp = os.path

def create_plot_window(vis, xlabel, ylabel, title, win, env, trace_name):
  if not isinstance(trace_name, list):
    trace_name = [trace_name]

  vis.line(X=np.array([1]), Y=np.array([np.nan]), win=win, env=env,
    name=trace_name[0],
    opts=dict(xlabel=xlabel, ylabel=ylabel, title=title))
  for name in trace_name[1:]:
    vis.line(X=np.array([1]), Y=np.array([np.nan]), win=win, env=env,
    name=name)


def train(data_dir, instruction, config_file, experiment_suffix=None,
    checkpoint_dir='.', device_id=0, weights_filename=None,
    include_sessions=None, exclude_sessions=None):
  # config
  config = configparser.ConfigParser()
  config.read(config_file)

  section = config['optim']
  batch_size = section.getint('batch_size')
  max_epochs = section.getint('max_epochs')
  base_lr = section.getfloat('base_lr')
  momentum = section.getfloat('momentum')
  weight_decay = section.getfloat('weight_decay', 5e-3)
  val_interval = section.getint('val_interval')
  do_val = val_interval > 0

  section = config['misc']
  log_interval = section.getint('log_interval')
  shuffle = section.getboolean('shuffle')
  num_workers = section.getint('num_workers')

  section = config['hyperparams']
  n_ensemble = section.getint('n_ensemble', 1)
  random_rotation = section.getfloat('random_rotation')
  pos_weight = section.getfloat('pos_weight')
  lr_step_size = section.getint('lr_step_size', 10000)
  lr_gamma = section.getfloat('lr_gamma', 1.0)
  droprate = section.getfloat('droprate')

  # cuda
  if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
  else:
    devices = os.environ['CUDA_VISIBLE_DEVICES']
    devices = devices.split(',')[device_id]
    os.environ['CUDA_VISIBLE_DEVICES'] = devices
  device = 'cuda:0'

  # create dataset and model
  model_name = config_file.split('/')[-1].split('.')[0]
  kwargs = dict(data_dir=data_dir, instruction=instruction,
    include_sessions=include_sessions, exclude_sessions=exclude_sessions)
  if 'voxnet' in model_name:
    models = [VoxNet(droprate=droprate) for _ in range(n_ensemble)]
    grid_size = config['hyperparams'].getint('grid_size')
    train_dset = VoxelDataset(grid_size=grid_size, train=True,
      random_rotation=random_rotation, **kwargs)
    val_dset = VoxelDataset(grid_size=grid_size, random_rotation=0,
      train=False, **kwargs)
  elif 'pointnet' in model_name:
    models = [PointNet(droprate=droprate) for _ in range(n_ensemble)]
    n_points = section.getint('n_points')
    random_scale = config['hyperparams'].getfloat('random_scale')
    train_dset = PointCloudDataset(n_points=n_points, train=True,
      random_rotation=random_rotation, random_scale=random_scale, **kwargs)
    val_dset   = PointCloudDataset(n_points=n_points, train=False,
      random_rotation=0, random_scale=0, **kwargs)
  else:
    raise NotImplementedError

  # checkpointing
  exp_name = '{:s}_{:s}_smcl'.format(instruction, model_name)
  if experiment_suffix:
    exp_name += '_{:s}'.format(experiment_suffix)

  def checkpoint_fn(engine: Engine):
    return -engine.state.avg_loss

  checkpoint_dir = osp.join(checkpoint_dir, exp_name)
  checkpoint_kwargs = dict(dirname=checkpoint_dir, filename_prefix='checkpoint',
      score_function=checkpoint_fn, create_dir=True, require_empty=False,
      save_as_state_dict=False)
  checkpoint_dict = {'model_{:d}'.format(idx): models[idx] 
      for idx in range(len(models))}

  # logging
  if not osp.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
  log_filename = osp.join(checkpoint_dir, 'training_log.txt')
  logging.basicConfig(level=logging.INFO,
      handlers=[logging.FileHandler(log_filename, mode='w'), logging.StreamHandler()])
  logger = logging.getLogger()
  logger.info('Config from {:s}:'.format(config_file))
  with open(config_file, 'r') as f:
    for line in f:
      logger.info(line.strip())

  # load weights
  if weights_filename is not None:
    checkpoint = torch.load(osp.expanduser(weights_filename))
    checkpoint = {k:v for k,v in checkpoint.items() if 'conv4' not in k}
    for model in models:
      model.load_state_dict(checkpoint, strict=False)
    logger.info('Loaded weights from {:s}'.format(weights_filename))

  # loss function
  loss_fn = sMCLLoss(pos_weight=pos_weight)
  loss_fn.to(device=device)
  if do_val:
    val_loss_fn = sMCLLoss(train=False, pos_weight=pos_weight)
    val_loss_fn.to(device=device)

  # optimizer
  optims = []
  lr_schedulers = []
  for model in models:
    model.to(device=device)
    optim = toptim.SGD(model.parameters(), lr=base_lr, momentum=momentum,
      weight_decay=weight_decay)
    optims.append(optim)
    lr_scheduler = toptim.lr_scheduler.StepLR(optim, step_size=lr_step_size,
        gamma=lr_gamma)
    lr_schedulers.append(lr_scheduler)
  
  # dataloader 
  train_dloader = DataLoader(train_dset, batch_size=batch_size, shuffle=shuffle,
    pin_memory=True, num_workers=num_workers, drop_last=True)
  if do_val:
    val_dloader = DataLoader(val_dset, batch_size=batch_size, shuffle=shuffle,
      pin_memory=True, num_workers=num_workers, drop_last=True)

  # train and val loops
  def train_loop(engine: Engine, batch):
    geom, tex_targ = batch
    geom     = geom.to(device=device, non_blocking=True)
    tex_targ = tex_targ.to(device=device, non_blocking=True)
    tex_preds = []
    for model, optim in zip(models, optims):
      model.train()
      optim.zero_grad()
      tex_preds.append(model(geom))
    loss, _ = loss_fn(tex_preds, tex_targ)
    loss.backward()
    for optim in optims:
      optim.step()
    engine.state.train_loss = loss.item()
    return loss.item()
  trainer = Engine(train_loop)
  train_checkpoint_handler = ModelCheckpoint(score_name='train_loss',
      **checkpoint_kwargs)
  trainer.add_event_handler(Events.EPOCH_COMPLETED, train_checkpoint_handler,
      checkpoint_dict)

  if do_val:
    def val_loop(engine: Engine, batch):
      geom, tex_targ = batch
      geom     = geom.to(device=device, non_blocking=True)
      tex_targ = tex_targ.to(device=device, non_blocking=True)
      tex_preds = []
      for model in models:
        model.eval()
        with torch.no_grad():
          tex_preds.append(model(geom))
      loss, _ = val_loss_fn(tex_preds, tex_targ)
      engine.state.val_loss = loss.item()
      return loss.item()
    valer = Engine(val_loop)
    val_checkpoint_handler = ModelCheckpoint(score_name='val_loss',
        **checkpoint_kwargs)
    valer.add_event_handler(Events.EPOCH_COMPLETED, val_checkpoint_handler,
        checkpoint_dict)

  # callbacks
  vis = visdom.Visdom()
  loss_win = 'loss'
  create_plot_window(vis, '#Epochs', 'Loss', 'Training and Validation Loss',
    win=loss_win, env=exp_name, trace_name=['train_loss', 'val_loss'])

  @trainer.on(Events.ITERATION_COMPLETED)
  def log_training_loss(engine):
    it = (engine.state.iteration - 1) % len(train_dloader)
    engine.state.avg_loss = (engine.state.avg_loss*it + engine.state.output) / \
                            (it + 1)

    if it % log_interval == 0:
      logger.info("{:s} train Epoch[{:03d}/{:03d}] Iteration[{:04d}/{:04d}] "
            "Loss: {:02.4f} lr: {:.4f}".
        format(exp_name, engine.state.epoch, max_epochs, it+1, len(train_dloader),
        engine.state.output, lr_schedulers[0].get_lr()[0]))
      epoch = engine.state.epoch - 1 +\
              float(it)/(len(train_dloader)-1)

      vis.line(X=np.array([epoch]), Y=np.array([engine.state.output]),
        update='append', win=loss_win, env=exp_name, name='train_loss')

  if do_val:
    @valer.on(Events.ITERATION_COMPLETED)
    def avg_loss_callback(engine: Engine):
      it = (engine.state.iteration - 1) % len(train_dloader)
      engine.state.avg_loss = (engine.state.avg_loss*it + engine.state.output) / \
                              (it + 1)
      if it % log_interval == 0:
        logger.info("{:s} val Iteration[{:04d}/{:04d}] Loss: {:02.4f}"
          .format(exp_name, it+1, len(val_dloader), engine.state.output))

    @valer.on(Events.EPOCH_COMPLETED)
    def log_val_loss(engine: Engine):
      vis.line(X=np.array([trainer.state.epoch]),
        Y=np.array([engine.state.avg_loss]), update='append', win=loss_win,
        env=exp_name, name='val_loss')

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_val(engine: Engine):
      vis.save([exp_name])
      if val_interval < 0:  # don't do validation
        return
      if engine.state.epoch % val_interval != 0:
        return
      valer.run(val_dloader)

    @trainer.on(Events.EPOCH_STARTED)
    def step_lr_schedulers(engine: Engine):
      for lr_scheduler in lr_schedulers:
        lr_scheduler.step()

  def reset_avg_loss(engine: Engine):
    engine.state.avg_loss = 0
  trainer.add_event_handler(Events.EPOCH_STARTED, reset_avg_loss)
  if do_val:
    valer.add_event_handler(Events.EPOCH_STARTED, reset_avg_loss)

  # Ignite the torch!
  trainer.run(train_dloader, max_epochs)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir',
    default=osp.join('data', 'voxelized_meshes'))
  parser.add_argument('--checkpoint_dir',
    default=osp.join('data', 'checkpoints'))
  parser.add_argument('--instruction', required=True)
  parser.add_argument('--config_file', required=True)
  parser.add_argument('--weights_file', default=None)
  parser.add_argument('--suffix', default=None)
  parser.add_argument('--device_id', default=0)
  parser.add_argument('--include_sessions', default=None)
  parser.add_argument('--exclude_sessions', default=None)
  args = parser.parse_args()

  include_sessions = None
  if args.include_sessions is not None:
    include_sessions = args.include_sessions.split(',')
  exclude_sessions = None
  if args.exclude_sessions is not None:
    exclude_sessions = args.exclude_sessions.split(',')
  train(osp.expanduser(args.data_dir), args.instruction, args.config_file,
    experiment_suffix=args.suffix, device_id=args.device_id,
    checkpoint_dir=osp.expanduser(args.checkpoint_dir),
    weights_filename=args.weights_file, include_sessions=include_sessions,
    exclude_sessions=exclude_sessions)
