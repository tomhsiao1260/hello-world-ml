import torch
import numpy as np

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

print('Hello ML World!')
print('Device: ', device_type)

# https://pytorch.org/get-started/locally/
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
