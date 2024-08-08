import torch
import numpy as np

if __name__ == "__main__":
  device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

  print('Hello ML World!')
  print('Device: ', device_type)
