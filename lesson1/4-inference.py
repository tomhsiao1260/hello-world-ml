import os
import cv2
import gc
import torch
import shutil
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
fragment_id = 'casey_pi'
model_path = './model/timesformer_wild15_20230702185753_0_fr_i3depoch=12.ckpt'

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

def read_image_mask(fragment_id, start_idx=18, end_idx=38, rotation=0):
  image_stack = []
  mid = 65 // 2
  start = mid - CFG.in_chans // 2
  end = mid + CFG.in_chans // 2

  for i in range(start_idx, end_idx):
    image = cv2.imread(f"{segment_path}/{fragment_id}/layers/{i:02}.tif", 0)
    pad0 = (256 - image.shape[0] % 256)
    pad1 = (256 - image.shape[1] % 256)
    image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
    image = np.clip(image, 0, 200)
    image_stack.append(image)
  
  image_stack = np.stack(image_stack, axis=2)
  fragment_mask = cv2.imread(f"{segment_path}/{fragment_id}/{fragment_id}_mask.png", 0)
  fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)

  return image_stack, fragment_mask

def get_img_splits(fragment_id, start_idx, end_idx, rotation=0):
  image_stack, fragment_mask = read_image_mask(fragment_id, start_idx, end_idx)

  images = []
  coords = []
  x_list = list(range(0, image_stack.shape[1]-CFG.tile_size+1, CFG.stride))
  y_list = list(range(0, image_stack.shape[0]-CFG.tile_size+1, CFG.stride))

  for ymin in y_list:
    for xmin in x_list:
      ymax = ymin + CFG.tile_size
      xmax = xmin + CFG.tile_size
      if not np.any(fragment_mask[ymin:ymax, xmin:xmax]==0):
        images.append(image_stack[ymin:ymax, xmin:xmax])
        coords.append([xmin, ymin, xmax, ymax])

  transform = A.Compose([
    A.Resize(CFG.size, CFG.size),
    A.Normalize(mean= [0] * CFG.in_chans, std= [1] * CFG.in_chans),
    ToTensorV2(transpose_mask=True),
  ])

  coords = np.stack(coords)
  dataset = CustomDatasetTest(images, coords, CFG, transform=transform)
  # dataset = CustomDatasetTest(images[:1000], coords[:1000], CFG, transform=transform)
  loader = DataLoader(dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
  image_shape = (image_stack.shape[0], image_stack.shape[1])

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

def predict_fn(loader, model, device, image_shape):
  kernel = gkern(CFG.size, 1)
  kernel = kernel / kernel.max()

  mask_pred = np.zeros(image_shape)
  mask_count = np.zeros(image_shape)

  predict_folder = f"./predict/{fragment_id}/"
  if os.path.exists(predict_folder): shutil.rmtree(predict_folder)
  os.makedirs(predict_folder, exist_ok=True)

  for step, (images, coords) in tqdm(enumerate(loader), total=len(loader)):
    images = images.to(device)
    batch_size = images.size(0)

    with torch.no_grad():
      with torch.autocast(device_type=device_type):
        y_preds = model(images)
    y_preds = torch.sigmoid(y_preds).to('cpu')

    for i, (x1, y1, x2, y2) in enumerate(coords):
      mask_pred[y1:y2, x1:x2] += np.multiply(F.interpolate(y_preds[i].unsqueeze(0).float(), scale_factor=16, mode='bilinear').squeeze(0).squeeze(0).numpy(), kernel)
      mask_count[y1:y2, x1:x2] += np.ones((CFG.size, CFG.size))

    filename = f"./predict/{fragment_id}/{fragment_id}_{step}.png"
    image_save(filename, mask_pred.copy(), mask_count.copy())

  filename = f"./predict/{fragment_id}/{fragment_id}.png"
  image_save(filename, mask_pred.copy(), mask_count.copy())

def image_save(filename, mask_pred, mask_count):
  mask_count[mask_count == 0] = 1

  data = mask_pred / mask_count
  data = np.clip(np.nan_to_num(data), a_min=0, a_max=1)
  data /= data.max()
  data = (data * 255).astype(np.uint8)
  cv2.imwrite(filename, data)

if __name__ == "__main__":
  model = RegressionPLModel.load_from_checkpoint(model_path, map_location=device, strict=False)
  model.eval()

  start_idx = 17
  end_idx = start_idx + 26
  loader, coords, image_shape, fragment_mask = get_img_splits(fragment_id, start_idx, end_idx)

  predict_fn(loader, model, device, image_shape)

  del loader, model
  torch.cuda.empty_cache()
  gc.collect()
