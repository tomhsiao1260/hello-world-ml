import os
import sys
import cv2
import gc
import torch
import numpy as np
import scipy.stats as st
from tqdm.auto import tqdm
import pytorch_lightning as pl
import torch.nn.functional as F
from timesformer_pytorch import TimeSformer
from torch.utils.data import Dataset, DataLoader
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

class CFG:
  in_chans = 26 # 65

  size = 64
  tile_size = 64
  stride = tile_size // 3

  # batch_size = 8
  batch_size = 256

  num_workers = 4
  # num_workers = 16

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
    image = cv2.imread(f"{segment_path}/{segment_id}/layers/{i:02}.tif", 0)
    image = image[ys: ye, xs: xe]
    image = np.clip(image, 0, 200)
    image_stack.append(image)

  image_stack = np.stack(image_stack, axis=2)
  image_shape = (h, w)
  image_coord = (xs, ys, xe, ye)

  return image_stack, fragment_mask, image_shape, image_coord

def get_img_splits(segment_id, start_idx, end_idx, rotation=0):
  image_stack, fragment_mask, image_shape, (xs, ys, xe, ye) = read_image_mask(segment_id, start_idx, end_idx)

  images = []
  coords = []
  (h, w) = image_shape
  x_list = list(range(xs, xe-CFG.tile_size+1, CFG.stride))
  y_list = list(range(ys, ye-CFG.tile_size+1, CFG.stride))

  for ymin in y_list:
    for xmin in x_list:
      if (ymin > h-CFG.tile_size): ymin = h-CFG.tile_size
      if (xmin > w-CFG.tile_size): xmin = w-CFG.tile_size
      ymax = ymin + CFG.tile_size
      xmax = xmin + CFG.tile_size

      if not np.any(fragment_mask[ymin-ys:ymax-ys, xmin-xs:xmax-xs]==0):
        images.append(image_stack[ymin-ys:ymax-ys, xmin-xs:xmax-xs])
        coords.append([xmin, ymin, xmax, ymax])

  if len(coords) == 0: return None, None, None, None

  transform = A.Compose([
    A.Resize(CFG.size, CFG.size),
    A.Normalize(mean= [0] * CFG.in_chans, std= [1] * CFG.in_chans),
    ToTensorV2(transpose_mask=True),
  ])

  coords = np.stack(coords)
  dataset = CustomDatasetTest(images, coords, CFG, transform=transform)
  loader = DataLoader(dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

  return loader, coords, image_shape, fragment_mask

class CustomDatasetTest(Dataset):
  def __init__(self, images, coords, cfg, transform=None):
    self.images = images
    self.coords = coords
    self.cfg = cfg
    self.transform = transform

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    image = self.images[idx]   
    coord = self.coords[idx]

    if self.transform:
      data = self.transform(image=image)    
      image = data['image'].unsqueeze(0)   

    return image, coord

class RegressionPLModel(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.save_hyperparameters()
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

def predict_fn(loader, model, image_shape):
  kernel = gkern(CFG.size, 1)
  kernel = kernel / kernel.max()

  mask_pred = np.zeros(image_shape)
  mask_count = np.zeros(image_shape)

  predict_folder = f"./predict/{segment_id}/"
  os.makedirs(predict_folder, exist_ok=True)

  for step, (images, coords) in tqdm(enumerate(loader), total=len(loader)):
    images = images.to(device)

    with torch.no_grad():
      with torch.autocast(device_type=device_type):
        y_preds = model(images)
    y_preds = torch.sigmoid(y_preds).to('cpu')

    for i, (x1, y1, x2, y2) in enumerate(coords):
      mask_pred[y1:y2, x1:x2] += np.multiply(F.interpolate(y_preds[i].unsqueeze(0).float(), scale_factor=16, mode='bilinear').squeeze(0).squeeze(0).numpy(), kernel)
      mask_count[y1:y2, x1:x2] += np.ones((CFG.size, CFG.size))

  data = process_image(mask_pred.copy(), mask_count.copy())

  filename = f"./predict/{segment_id}/{segment_id}.png"
  if os.path.exists(filename):
    prev = cv2.imread(filename, 0)
    data = np.where((data != 0) & (prev != 0), data // 2 + prev // 2, data + prev)

  cv2.imwrite(filename, data)

def process_image(mask_pred, mask_count):
  mask_count[mask_count == 0] = 1

  data = mask_pred / mask_count
  data = np.clip(np.nan_to_num(data), a_min=0, a_max=1)
  data /= data.max()
  data = (data * 255).astype(np.uint8)
  return data

if __name__ == "__main__":
  start_idx = 17
  end_idx = start_idx + 26

  loader, coords, image_shape, fragment_mask = get_img_splits(segment_id, start_idx, end_idx)
  if (loader == None): sys.exit()

  model = RegressionPLModel.load_from_checkpoint(model_path, map_location=device, strict=False)
  model.eval()

  predict_fn(loader, model, image_shape)

  del loader, model
  torch.cuda.empty_cache()
  gc.collect()
