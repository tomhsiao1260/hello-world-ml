# agreement: https://forms.gle/HV1J6dJbmCB2z5QL8
# casey pi: https://dl.ash2txt.org/community-uploads/yao/casey_pi/

import os
import cv2
import torch
import numpy as np
import scipy.stats as st
import pytorch_lightning as pl
import torch.nn.functional as F
from timesformer_pytorch import TimeSformer
import albumentations as A
from albumentations.pytorch import ToTensorV2

def gkern(kernlen=21, nsig=3):
  """Returns a 2D Gaussian kernel."""
  x = np.linspace(-nsig, nsig, kernlen+1)
  kern1d = np.diff(st.norm.cdf(x))
  kern2d = np.outer(kern1d, kern1d)
  return kern2d / kern2d.sum()

def read_image_mask(data_path, start_idx, end_idx, rotation=0):
  image_stack = []
  mid = 65 // 2

  for i in range(start_idx, end_idx):
    filename = os.path.join(data_path, f"{i:02}.tif")
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) # 0 ~ 255
    pad0 = (256 - image.shape[0] % 256)
    pad1 = (256 - image.shape[1] % 256)
    image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
    image = np.clip(image, 0, 200) # 0 ~ 200
    image_stack.append(image)
  
  image_stack = np.stack(image_stack, axis=2)
  return image_stack

def get_img_splits(data_path, start_idx, end_idx, rotation=0):
  # (768, 768, 26), 0 ~ 200
  image_stack = read_image_mask(data_path, start_idx, end_idx)

  # (64, 64, 26), 0 ~ 200
  image_stack = image_stack[:64, :64, :]
  print('stack: ', image_stack.shape, np.max(image_stack))

  transform = A.Compose([
    A.Resize(64, 64),
    A.Normalize(mean= [0] * 26, std= [1] * 26),
    ToTensorV2(transpose_mask=True),
  ])

  # (26, 64, 64), 0 ~ 200/255
  image_stack = transform(image=image_stack)
  image_stack = image_stack["image"]
  print('transform stack: ', image_stack.numpy().shape, np.max(image_stack.numpy()))

  return image_stack

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

def predict_fn(model, images, device_type):
  # (26, 64, 64) -> (1, 1, 26, 64, 64)
  images = images.unsqueeze(0).unsqueeze(0)

  # (1, 26, 1, 64, 64) -> (1, 16) -> (1, 1, 4, 4) 
  with torch.no_grad():
    with torch.autocast(device_type=device_type):
      pred = model(images)
  pred = torch.sigmoid(pred).to('cpu')

  # (4, 4) -> (64, 64)
  kernel = gkern(64, 1)
  kernel = kernel / kernel.max()
  pred = np.multiply(F.interpolate(pred[0].unsqueeze(0).float(), scale_factor=16, mode='bilinear').squeeze(0).squeeze(0).numpy(), kernel)

  return pred

if __name__ == "__main__":
  device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
  device = torch.device(device_type)

  model_path = './model/timesformer_wild15_20230702185753_0_fr_i3depoch=12.ckpt'
  model = RegressionPLModel.load_from_checkpoint(model_path, map_location=device, strict=False)
  model.eval()

  data_path = './data/casey_pi/layers/'
  start_idx = 17
  end_idx = start_idx + 26
  images = get_img_splits(data_path, start_idx, end_idx)

  pred = predict_fn(model, images, device_type)

  # see raw data
  data = (images * 255).numpy().astype(np.uint8)
  for layer in range(data.shape[0]):
    cv2.imshow('data', data[layer, :, :])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

  # see prediction
  cv2.imshow('predict', (pred * 255).astype(np.uint8))
  cv2.waitKey(0)
  cv2.destroyAllWindows()

