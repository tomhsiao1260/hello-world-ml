## Setup

Virtual environment

```bash
conda create --name hello-world-ml
conda activate hello-world-ml
```

Use the latest pip

```bash
conda install pip
pip install --upgrade pip
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
