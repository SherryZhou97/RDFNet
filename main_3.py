import torch
import torch.distributed as dist
import argparse
import torch.nn as nn
import torch.optim as optim
from models import Twist_net

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=3)
Flags = parser.parse_args()

local_rank = Flags.local_rank()

torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl') 

device = torch.device("cuda", local_rank)
# model = Twist_net(nFrame=28, nPhase=5).to(device)

model = nn.Linear(10, 10).to(device)

model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)


# 前向传播
outputs = model(torch.randn(20, 10).to(rank))
labels = torch.randn(20, 10).to(rank)
loss_fn = nn.MSELoss()
loss_fn(outputs, labels).backward()
# 后向传播
optimizer = optim.SGD(model.parameters(), lr=0.001)
optimizer.step()