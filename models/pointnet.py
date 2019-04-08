import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from IPython.core.debugger import set_trace


class STN3d(nn.Module):
    def __init__(self, n=4):
        super(STN3d, self).__init__()
        self.n = n
        self.conv1 = torch.nn.Conv1d(n, 64, 1, bias=False)
        self.conv2 = torch.nn.Conv1d(64, 128, 1, bias=False)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1, bias=False)
        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.fc3 = nn.Linear(256, n*n)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = np.eye(self.n).ravel().astype(np.float32)
        iden = torch.from_numpy(iden).view(1,self.n*self.n).repeat(batchsize,1)
        iden = iden.to(device=x.device)
        x = x + iden
        x = x.view(-1, self.n, self.n)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, n=4, n_ensemble=20):
        super(PointNetfeat, self).__init__()
        self.n = n
        self.n_ensemble = n_ensemble
        self.global_feat = global_feat
        self.stn = STN3d(n=n)
        self.conv1 = torch.nn.Conv1d(n+self.n_ensemble, 64, 1, bias=False)
        self.conv2 = torch.nn.Conv1d(64, 128, 1, bias=False)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, c):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2,1)
        x = torch.bmm(x, trans)
        x = x.transpose(2,1)
        x = torch.cat((x, c), dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans

class PointNetCls(nn.Module):
    def __init__(self, k = 2):
        super(PointNetCls, self).__init__()
        self.feat = PointNetfeat(global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
    def forward(self, x):
        x, trans = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x, dim=0), trans

class PointNetDenseCls(nn.Module):
    def __init__(self, k=2, n=4, n_ensemble=20, droprate=0):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.n = n
        self.n_ensemble = n_ensemble
        self.droprate = droprate
        self.drop  = torch.nn.Dropout(p=droprate)
        self.feat = PointNetfeat(global_feat=False, n=n, n_ensemble=n_ensemble)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1, bias=False)
        self.conv2 = torch.nn.Conv1d(512, 256, 1, bias=False)
        self.conv3 = torch.nn.Conv1d(256+self.n_ensemble, 128, 1, bias=False)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, c):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        if self.droprate > 0:
          x = self.drop(x)
        x, _ = self.feat(x, c)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.cat((x, c), dim=1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x


class DiversePointNet(nn.Module):
  def __init__(self, n_ensemble, n=4, droprate=0):
    super(DiversePointNet, self).__init__()
    self.n_ensemble = n_ensemble
    self.pointnet = PointNetDenseCls(n_ensemble=n_ensemble, n=n,
        droprate=droprate)

  def forward(self, xs):
    """
    :param xs: N x 3 x P
    :return: N x E x P
    """
    N, _, P = xs.shape
    preds = []
    for x in xs:
      x = x.view(1, *x.shape).expand(self.n_ensemble, -1, -1) 
      c = torch.eye(self.n_ensemble, dtype=x.dtype, device=x.device)
      c = c.view(self.n_ensemble, self.n_ensemble, 1).expand(-1, -1, P)
      pred = self.pointnet(x, c)
      preds.append(pred)
    preds = torch.stack(preds)
    return preds


if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())

    pointfeat = PointNetfeat(global_feat=True)
    out, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _ = seg(sim_data)
    print('seg', out.size())
