# PyTorch Lightning: https://pypi.org/project/pytorch-lightning

import torch
import numpy as np
import scipy.stats as st
import pytorch_lightning as pl
import torch.nn.functional as F
from timesformer_pytorch import TimeSformer

def gkern(kernlen=21, nsig=3):
  """Returns a 2D Gaussian kernel."""
  x = np.linspace(-nsig, nsig, kernlen+1)
  kern1d = np.diff(st.norm.cdf(x))
  kern2d = np.outer(kern1d, kern1d)
  return kern2d / kern2d.sum()

class RegressionPLModel(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.backbone = TimeSformer(
      dim = 512,
      image_size = 64,
      patch_size = 16,
      num_frames = 30,
      num_classes = 16,
      channels=1,
      depth = 8,
      heads = 6,
      dim_head =  64,
      attn_dropout = 0.1,
      ff_dropout = 0.1
    )

  def forward(self, x):
    x = self.backbone(torch.permute(x, (0, 2, 1, 3, 4)))
    x = x.view(-1, 1, 4, 4)      
    return x

if __name__ == "__main__":
  device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
  device = torch.device(device_type)

  model_path = './model/timesformer_wild15_20230702185753_0_fr_i3depoch=12.ckpt'
  model = RegressionPLModel.load_from_checkpoint(model_path, map_location=device, strict=False)
  model.eval()

  # (3, 1, 26, 64, 64) -> (3, 26, 1, 64, 64) -> (3, 16) -> (3, 1, 4, 4)
  images = torch.randn(3, 1, 26, 64, 64)
  with torch.no_grad():
    with torch.autocast(device_type=device_type):
      preds = model(images)
  preds = torch.sigmoid(preds).to('cpu')

  # (4, 4) -> (64, 64)
  for pred in preds:
    print('predict: ', pred.shape, pred)
    kernel = gkern(64, 1)
    kernel = kernel / kernel.max()
    pred = np.multiply(F.interpolate(pred.unsqueeze(0).float(), scale_factor=16, mode='bilinear').squeeze(0).squeeze(0).numpy(), kernel)
    print('kernel predict: ', pred.shape)



