# Thesis: https://arxiv.org/abs/2102.05095
# Usage: https://github.com/lucidrains/TimeSformer-pytorch

import torch
from timesformer_pytorch import TimeSformer

model = TimeSformer(
  dim = 512,
  image_size = 224,
  patch_size = 16,
  num_frames = 8,
  num_classes = 10,
  depth = 12,
  heads = 8,
  dim_head =  64,
  attn_dropout = 0.1,
  ff_dropout = 0.1
)

video = torch.randn(2, 8, 3, 224, 224) # (batch x frames x channels x height x width)

pred = model(video) # -inf ~ inf
print(pred) # (2, 10)

pred = torch.sigmoid(pred).to('cpu') # 0 ~ 1
print(pred) # (2, 10)