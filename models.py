import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import utils

class Twist_net_one_phase(nn.Module):
    def __init__(self, nFrame):
        super().__init__()

        self.conv0 = nn.Conv2d(nFrame,64,3,1,1)
        self.conv11 = nn.Conv2d(64,64,3,1,1)
        self.conv12 = nn.Conv2d(64,64,3,1,1)
        self.conv13 = nn.Conv2d(64,64,3,1,1)
        self.conv14 = nn.Conv2d(64,64,3,1,1)
        self.conv6 = nn.Conv2d(64,nFrame,3,1,1)
        self.last = nn.Sigmoid()

        self.nFrame = nFrame
        self.lambdaStep = torch.tensor(0.5, dtype=torch.float32, requires_grad=True)
        self.softThr = torch.tensor(0.01, dtype=torch.float32, requires_grad=True)
        self.t = torch.tensor(1, dtype=torch.float32, requires_grad=True)

    def forward(self, layerxk, layerzk, Phi, PhiT, Yinput):
        rk = utils.shift(Phi*layerzk[-1])
        rk = torch.sum(rk, dim=1)/self.nFrame*2
        rk = utils.shift_back(rk)
        rk = rk - Yinput
        rk = PhiT * rk
        rk = layerzk[-1] - self.lambdaStep * rk

        Frk = self.conv0(rk)
        tmp = Frk.clone()
        Frk = F.relu(self.conv11(Frk))
        Frk = self.conv12(Frk)
        
        softFrk = torch.sign(Frk) * F.relu(torch.abs(Frk)-self.softThr)

        FFrk = F.relu(self.conv13(softFrk))
        FFrk = self.conv14(FFrk)
        FFrk = self.conv6(FFrk)

        xk = rk + FFrk
        # xk = self.last(xk)
        zk = (1 + self.t) * xk - self.t * layerxk[-1]

        sFFrk = self.conv14(F.relu(self.conv13(Frk)))
        symmetric = sFFrk - tmp
        return xk, zk, symmetric, Frk


def compute_loss(prediction, predictionSymmetric, Xoutput, transField):
    lossMean = 0
    nPhase = len(predictionSymmetric)
    lossMean += torch.mean((prediction[-1] - Xoutput) ** 2)
    lossSymmetric = 0
    for k in range(nPhase):
        lossSymmetric += torch.mean(predictionSymmetric[k] ** 2)
    lossSparsity = 0
    for k in range(nPhase):
        lossSparsity += torch.mean(torch.abs(transField[k]))
    return lossMean, lossSymmetric, lossSparsity


class Twist_net(nn.Module):
    def __init__(self, nFrame=28, nPhase=5):
        super().__init__()
        phase_list = []
        for i in range(nPhase):
            phase = Twist_net_one_phase(nFrame)
            phase_list.append(phase)
        self.phase_list = nn.ModuleList(phase_list)

    def forward(self, Xinput, Xoutput, Phi, PhiT, Yinput):
        layerxk = []
        layerzk = []
        layerSymmetric = []
        transField = []
        layerxk.append(Xinput) 
        layerzk.append(Xinput)
        for phase in self.phase_list:
            xk, zk, convSymmetric, field = phase(layerxk, layerzk, Phi, PhiT, Yinput)
            layerxk.append(xk)
            layerzk.append(zk)
            layerSymmetric.append(convSymmetric)
            transField.append(field)

        lossMean, lossSymmetric, lossSparsity = compute_loss(layerxk, layerSymmetric, Xoutput, transField)
        lossAll = lossMean + 0.01*lossSymmetric + 0.001*lossSparsity
        return layerxk[-1], lossAll
