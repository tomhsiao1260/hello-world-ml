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

Use Cuda. Checkout [here](https://pytorch.org/get-started/locally/).

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Install dependency

```bash
pip install -r requirements.txt
```

## Run App

```bash
python app.py
```

## Syntax

```bash
conda env list

conda activate hello-world-ml
conda deactivate

conda create --name hello-world-ml
conda env remove --name hello-world-ml
```

## Data Agreement

To access the data in Vesuvius Challenge, please fill in [this form](https://forms.gle/HV1J6dJbmCB2z5QL8) in advance.

