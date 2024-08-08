## PyTorch Lightning: https://pypi.org/project/pytorch-lightning

import torch
import pytorch_lightning as pl
from timesformer_pytorch import TimeSformer

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
  pred = model(images)
  pred = torch.sigmoid(pred).to('cpu')
  print(pred)


