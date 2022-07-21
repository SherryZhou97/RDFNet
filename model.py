import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import utils

class Shrinkage_CW(nn.Module):
    def __init__(self, channel):
        super(Shrinkage_CW, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1)
        )

    def forward(self, x):
        x_raw = x
        x = torch.abs(x)
        x_abs = x # 2*64*256*256
        x = self.fc(x)
        x = F.softmax(x, 1)
        x = torch.mul(x_abs, x)
        # soft_thresholding
        sub = x_abs - x
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x

class attention(nn.Module):
    def __init__(self):
        super(attention, self).__init__()
        self.avgpool = nn.AvgPool2d(5,1,3)
        self.fc1 = nn.Conv2d(64, 64, 1, bias=False)
        self.fc2 = nn.Conv2d(64, 3, 1, bias=True)
    
    def forward(self, x):
        x_raw = x
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x = x.view(x.size(0), -1)
        return F.softmax(x, 1)

class FF_block(nn.Module):
    def __init__(self):
        super(FF_block, self).__init__()
        self.shrinkage = Shrinkage_CW(64)
        self.conv11 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv12 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv13 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv14 = nn.Conv2d(64, 64, 3, 1, 1)
    
    def forward(self, x):
        x_raw = x.clone()
        x = F.relu(self.conv11(x))
        x = self.conv12(x)
        Fx = F.relu(self.conv13(self.shrinkage(x)))
        Fx = self.conv14(Fx)
        s = self.conv14(F.relu(self.conv13(x)))
        sym2 = s - x_raw
        return Fx, sym2

        
class Dynamic_block(nn.Module):
    def __init__(self):
        super(Dynamic_block, self).__init__()
        self.attention = attention()
        # self.block1 = F_block()
        self.block1 = FF_block()
        self.block2 = FF_block()
        self.block3 = FF_block()
    
    def forward(self, x):
        softmax_attention = self.attention(x) # 2*4*256*256
        # print(softmax_attention.shape)
        
        aggregate_weight1 = softmax_attention[:,0,:,:]
        aggregate_weight2 = softmax_attention[:,1,:,:]
        aggregate_weight3 = softmax_attention[:,2,:,:]
        
        aggregate_weight1 = aggregate_weight1.unsqueeze(dim = 1)
        aggregate_weight2 = aggregate_weight2.unsqueeze(dim = 1) # 2*1*256*256
        aggregate_weight3 = aggregate_weight3.unsqueeze(dim = 1) 

        block1, sym1 = self.block1(x)
        block2, sym2 = self.block2(x)
        block3, sym3 = self.block3(x)

        losssym1 = torch.mean(sym1 ** 2)
        losssym2 = torch.mean(sym2 ** 2)
        losssym3 = torch.mean(sym3 ** 2)
        
        output = block1*aggregate_weight1 + block2*aggregate_weight2 + block3*aggregate_weight3

        return output, losssym1, losssym2, losssym3
        
class Twist_net_one_phase(nn.Module):
    def __init__(self, nFrame):
        super().__init__()

        self.Dyblock = Dynamic_block()
        self.conv0 = nn.Conv2d(nFrame, 64, 3, 1, 1)
        self.conv6 = nn.Conv2d(64, nFrame, 3, 1, 1)

        self.nFrame = nFrame
        self.lambdaStep = nn.Parameter(torch.tensor(0.5, dtype=torch.float32, requires_grad=True))
        self.t = nn.Parameter(torch.tensor(1, dtype=torch.float32, requires_grad=True))

    def forward(self, layerxk, layerzk, Phi, PhiT, Yinput):
        rk = utils.shift(Phi * layerzk[-1])
        rk = torch.sum(rk, dim=1) / self.nFrame * 2
        rk = utils.shift_back(rk)
        rk = rk - Yinput
        rk = PhiT * rk
        rk = layerzk[-1] - self.lambdaStep * rk

        Frk = self.conv0(rk) # 2*64*256*256
        
        FFrk, losssym1, losssym2, losssym3 = self.Dyblock(Frk)
        
        FFrk = self.conv6(FFrk) # 2*28*256*256

        xk = rk + FFrk
        zk = (1 + self.t) * xk - self.t * layerxk[-1]

        return xk, zk, Frk, losssym1, losssym2, losssym3


def compute_loss(prediction, Xoutput, transField, losssym1, losssym2, losssym3):
    lossMean = 0
    lossMean += torch.mean((prediction[-1] - Xoutput) ** 2)
    loss_sym1_all = 0
    loss_sym2_all = 0
    loss_sym3_all =0
    for k in range(5):
        loss_sym1_all += torch.mean(losssym1[k] ** 2)
        loss_sym2_all += torch.mean(losssym2[k] ** 2)
        loss_sym3_all += torch.mean(losssym3[k] ** 2)
    lossSparsity = 0
    for k in range(5):
        lossSparsity += torch.mean(torch.abs(transField[k]))
    return lossMean, lossSparsity, loss_sym1_all, loss_sym2_all, loss_sym3_all


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
        transField = []
        layerxk.append(Xinput)  # x0 = x0
        layerzk.append(Xinput)
        losssym1_list = []
        losssym2_list = []
        losssym3_list = []
        for phase in self.phase_list:
            xk, zk, field, losssym1, losssym2, losssym3 = phase(layerxk, layerzk, Phi, PhiT, Yinput)
            layerxk.append(xk)
            layerzk.append(zk)
            transField.append(field)
            losssym1_list.append(losssym1)
            losssym2_list.append(losssym2)
            losssym3_list.append(losssym3)

        lossMean, lossSparsity, loss_sym1, loss_sym2, loss_sym3 = compute_loss(layerxk, Xoutput, transField, losssym1_list, losssym2_list, losssym3_list)
        lossAll =  lossMean + 0.001 * lossSparsity + 0.001 * loss_sym1 + 0.001 * loss_sym2 + 0.001 * loss_sym3
        return layerxk[-1], lossAll

if __name__ == '__main__':
    Xoutput = torch.rand((2,28,256,256),dtype=torch.float32).cuda()
    Xinput = torch.rand((2,28,256,256),dtype=torch.float32).cuda()
    import scipy.io as sio
    Phi = sio.loadmat('./mask.mat')['mask']
    Phi = Phi[np.newaxis, np.newaxis, ...]
    Phi = torch.from_numpy(Phi).cuda()
    PhiT = Phi
    Yinput = torch.sum(Xoutput * Phi, dim=1, keepdim=True).repeat(1,28,1,1)

    net = Twist_net().cuda()
    layerxk, lossAll = net(Xinput, Xoutput, Phi, PhiT, Yinput)
    print(lossAll.item())

   
