## Setup

Virtual environment

```bash
conda activate hello-world-ml
```

Install dependency

```bash
pip install -r requirements.txt
```

## Model

Download the model [here](https://drive.google.com/drive/folders/1rn3GMOvtJRMBHOxVhWFVSY6IVI6xUnYp) and put it into `./model` folder.

## Run App

```bash
python inference.py
```

## Note

- `1_inference`: use TimeSformer
- `2_inference`: use our checkpoint
- `3_inference`: use real data
- `4_inference`: use DataLoader

Inference smaller region if you have memory issue.

Homework: Record your observations in other areas.


