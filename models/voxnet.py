import torch
import torch.nn as tnn
import torch.nn.functional as tnnF
from IPython.core.debugger import set_trace

class VoxNet(tnn.Module):
  def __init__(self, inplanes=5, outplanes=2, droprate=0):
    super(VoxNet, self).__init__()
    self.droprate = droprate
    self.drop = tnn.Dropout(p=droprate)
    nc = inplanes
    nc *= 4
    self.conv1 = tnn.Conv3d(inplanes, nc, kernel_size=3, padding=1, bias=False)
    self.bn1   = tnn.BatchNorm3d(nc)
    self.pool1 = tnn.MaxPool3d(2)

    nc *= 4
    self.conv2 = tnn.Conv3d(self.conv1.out_channels, nc,
      kernel_size=3, padding=1, bias=False)
    self.bn2   = tnn.BatchNorm3d(nc)
    self.pool2 = tnn.MaxPool3d(2)

    nc *= 4
    self.conv3 = tnn.Conv3d(self.conv2.out_channels, nc, kernel_size=3,
      padding=1, bias=False)
    self.bn3   = tnn.BatchNorm3d(nc)
    self.pool3 = tnn.MaxPool3d(4)

    inplanes = nc
    nc = nc // 4
    self.upconv1 = tnn.Conv3d(inplanes, nc, kernel_size=3, padding=1, bias=False)
    self.upbn1 = tnn.BatchNorm3d(nc)
    nc = nc // 4
    self.upconv2 = tnn.Conv3d(self.upconv1.out_channels, nc, kernel_size=3,
      padding=1, bias=False)
    self.upbn2 = tnn.BatchNorm3d(nc)
    nc = nc // 4
    self.upconv3 = tnn.Conv3d(self.upconv2.out_channels, nc,
      kernel_size=3, padding=1, bias=False)
    self.upbn3 = tnn.BatchNorm3d(nc)
    self.upconv4 = tnn.Conv3d(self.upconv3.out_channels, outplanes, kernel_size=3,
      padding=1)

    for m in self.modules():
      if isinstance(m, tnn.BatchNorm3d):
        tnn.init.constant_(m.weight, 1)
        tnn.init.constant_(m.bias, 0)

  def forward(self, x):
    if self.droprate > 0:
      x = self.drop(x)
    x = self.pool1(self.bn1(tnnF.relu(self.conv1(x))))
    x = self.pool2(self.bn2(tnnF.relu(self.conv2(x))))
    x = self.pool3(self.bn3(tnnF.relu(self.conv3(x))))
    x = self.upbn1(tnnF.relu(self.upconv1(x.view(x.shape[0], -1, 4, 4, 4))))
    x = tnnF.interpolate(x, scale_factor=4)
    x = self.upbn2(tnnF.relu(self.upconv2(x)))
    x = tnnF.interpolate(x, scale_factor=2)
    x = self.upbn3(tnnF.relu(self.upconv3(x)))
    x = tnnF.interpolate(x, scale_factor=2)
    x = self.upconv4(x)
    return x
