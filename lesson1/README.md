## Setup

Virtual environment

```bash
conda create --name hello-world
conda activate hello-world
```

Install dependency

```bash
conda install --yes --file requirements.txt
```

## Model

Download the model [here](https://drive.google.com/drive/folders/1rn3GMOvtJRMBHOxVhWFVSY6IVI6xUnYp) and put it into `./model` folder.

[TimeSformer](https://arxiv.org/abs/2102.05095)

## Run App

```bash
python inference.py
```
