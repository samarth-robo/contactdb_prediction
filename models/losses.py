import torch
import torch.nn as tnn
import numpy as np
from IPython.core.debugger import set_trace

# only count the loss where target is non-zero
class TextureLoss(tnn.Module):
  def __init__(self, pos_weight=10):
    super(TextureLoss, self).__init__()
    self.loss = tnn.CrossEntropyLoss(weight=torch.Tensor([1, pos_weight]),
        ignore_index=2, reduction='none')

  def forward(self, preds, targs):
    loss = self.loss(preds, targs)
    loss = torch.mean(loss, 1)
    return loss


def classification_error(preds, targs):
  _, pred_class = torch.max(preds, dim=1)
  errors = []
  for pred, targ in zip(pred_class, targs):
    mask = targ != 2
    masked_pred = pred.masked_select(mask)
    masked_targ = targ.masked_select(mask)
    if torch.sum(masked_pred) == 0:  # ignore degenerate masks
      error = torch.tensor(1000).to(device=preds.device, dtype=preds.dtype)
    else:
      error = masked_pred != masked_targ
      error = torch.mean(error.to(dtype=preds.dtype))
    errors.append(error)
  errors = torch.stack(errors)
  return errors


class sMCLLoss(tnn.Module):
  def __init__(self, pos_weight=10, eps=0.05, droprate=0.1, train=True,
			eval_mode=False):
    super(sMCLLoss, self).__init__()
    self.loss = classification_error if eval_mode else \
        TextureLoss(pos_weight=pos_weight)
    self.eps = eps
    self.droprate = droprate
    self.train = train
    self.eval_mode = eval_mode
    if eval_mode:
      self.train = False
      self.droprate = 0
      self.eps = 0

  def __call__(self, preds, targ):
    losses = []
    for pred in preds:
      pred = pred.view(*pred.shape[:2], -1)
      targ = targ.view(targ.shape[0], -1)
      if self.train:
        w = np.random.choice(2, p=[self.droprate, 1-self.droprate])
        pred = pred * w
      loss = self.loss(pred, targ)
      losses.append(loss)
    losses = torch.stack(losses)
    _, min_idx = torch.min(losses, 0)

    mults = torch.zeros_like(losses)
    for i,m in enumerate(min_idx):
      if self.train and (len(preds) > 1):
        mults[:, i] = self.eps / (len(preds)-1.0) * torch.ones(len(preds))
        mults[m, i] = 1 - self.eps
      else:
        mults[m, i] = 1

    # mults = []
    # for i in range(len(min_idx)):
    #   if len(preds) > 1:
    #   else:
    #     mult = torch.ones(1)
    #   mults.append(mult)
    # mults = torch.stack(mults).transpose(1, 2)
    # mults = mults.to(device=losses.device, dtype=losses.dtype)

    loss = torch.sum(mults * losses)

    return loss, min_idx
