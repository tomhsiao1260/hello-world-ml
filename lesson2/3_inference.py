# agreement: https://forms.gle/HV1J6dJbmCB2z5QL8
# casey pi: https://dl.ash2txt.org/community-uploads/yao/casey_pi/

import os
import sys
import cv2
import torch
import numpy as np
import scipy.stats as st
import pytorch_lightning as pl
import torch.nn.functional as F
from timesformer_pytorch import TimeSformer
import albumentations as A
from albumentations.pytorch import ToTensorV2

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_type)

segment_path = './data/'
segment_id = 'casey_pi'
model_path = './model/timesformer_wild15_20230702185753_0_fr_i3depoch=12.ckpt'

uv_min = [0, 0]
uv_max = [1, 1]

def gkern(kernlen=21, nsig=3):
  """Returns a 2D Gaussian kernel."""
  x = np.linspace(-nsig, nsig, kernlen+1)
  kern1d = np.diff(st.norm.cdf(x))
  kern2d = np.outer(kern1d, kern1d)
  return kern2d / kern2d.sum()

def read_image_mask(segment_id, start_idx, end_idx, rotation=0):
  image_stack = []
  mid = 65 // 2

  # mask processing
  fragment_mask = cv2.imread(f"{segment_path}/{segment_id}/{segment_id}_mask.png", 0)

  h, w = fragment_mask.shape
  xs, xe, ys, ye = w * uv_min[0], w * uv_max[0], h * (1 - uv_max[1]), h * (1 - uv_min[1])
  xs, xe, ys, ye = int(xs), int(xe), int(ys), int(ye)
  fragment_mask = fragment_mask[ys: ye, xs: xe]

  # image stack processing
  for i in range(start_idx, end_idx):
    image = cv2.imread(f"{segment_path}/{segment_id}/layers/{i:02}.tif", 0) # 0 ~ 255
    image = image[ys: ye, xs: xe]
    image = np.clip(image, 0, 200) # 0 ~ 200
    image_stack.append(image)

  # (256 * n, 256 * m, 26), 0 ~ 200
  image_stack = np.stack(image_stack, axis=2)
  image_shape = (h, w)
  image_coord = (xs, ys, xe, ye)

  return image_stack, fragment_mask, image_shape, image_coord

def get_img_splits(segment_id, start_idx, end_idx, rotation=0):
  image_stack, fragment_mask, image_shape, (xs, ys, xe, ye) = read_image_mask(segment_id, start_idx, end_idx)

  tile_size = 64
  stride = tile_size // 3

  images = []
  coords = []
  (h, w) = image_shape
  x_list = list(range(xs, xe-tile_size+1, stride))
  y_list = list(range(ys, ye-tile_size+1, stride))

  transform = A.Compose([
    A.Resize(64, 64),
    A.Normalize(mean= [0] * 26, std= [1] * 26),
    ToTensorV2(transpose_mask=True),
  ])

  for ymin in y_list:
    for xmin in x_list:
      if (ymin > h-tile_size): ymin = h-tile_size
      if (xmin > w-tile_size): xmin = w-tile_size
      ymax = ymin + tile_size
      xmax = xmin + tile_size

      if not np.any(fragment_mask[ymin-ys:ymax-ys, xmin-xs:xmax-xs]==0):
        # (tile_size, tile_size, 26) 0 ~ 200
        image = image_stack[ymin-ys:ymax-ys, xmin-xs:xmax-xs]
        image = transform(image=image)
        # (26, tile_size, tile_size) 0 ~ 200/255
        image = image["image"]
        images.append(image)
        coords.append([xmin, ymin, xmax, ymax])

  if len(coords) == 0: return None, None, None, None
  coords = np.stack(coords)

  return images, coords, image_shape, fragment_mask

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

def predict_fn(image_stack, coords, model, image_shape):
  mask_pred = np.zeros(image_shape)

  # (26, 64, 64) -> (1, 1, 26, 64, 64)
  ind = 0
  images = image_stack[ind]
  images = images.unsqueeze(0).unsqueeze(0)
  images = images.to(device)
  (x1, y1, x2, y2) = coords[ind]

  # (1, 26, 1, 64, 64) -> (1, 16) -> (1, 1, 4, 4) 
  with torch.no_grad():
    with torch.autocast(device_type=device_type):
      y_preds = model(images)
  y_preds = torch.sigmoid(y_preds).to('cpu')

  # (4, 4) -> (64, 64)
  kernel = gkern(64, 1)
  kernel = kernel / kernel.max()
  mask_pred[y1:y2, x1:x2] += np.multiply(F.interpolate(y_preds[0].unsqueeze(0).float(), scale_factor=16, mode='bilinear').squeeze(0).squeeze(0).numpy(), kernel)

  os.makedirs(f"./predict/{segment_id}/", exist_ok=True)

  # see cropped data 16.tif
  filename = f"./predict/{segment_id}/data.png"
  data = (image_stack[ind] * 255).numpy().astype(np.uint8)
  cv2.imwrite(filename, data[0])

  # see prediction
  filename = f"./predict/{segment_id}/prediction.png"
  data = (mask_pred * 255).astype(np.uint8)
  cv2.imwrite(filename, data)

if __name__ == "__main__":
  start_idx = 17
  end_idx = start_idx + 26

  images, coords, image_shape, fragment_mask = get_img_splits(segment_id, start_idx, end_idx)
  if (images == None): sys.exit()

  model = RegressionPLModel.load_from_checkpoint(model_path, map_location=device, strict=False)
  model.eval()

  predict_fn(images, coords, model, image_shape)

