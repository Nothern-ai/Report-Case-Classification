# Installation

We provide installation instructions for report case classification experiments here.

## Dependency Setup
Create an new conda virtual environment
```
conda create -n repocase python=3.9.7 -y
conda activate repocase
```

Install [Pytorch](https://pytorch.org/)>=1.8.0, [torchvision](https://pytorch.org/vision/stable/index.html)>=0.9.0 following official instructions. For example:
```
pip install torch==1.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Clone this repo and install required packages:
```
git clone https://github.com/NothernSJTU/12345_Report_Case-Predicter
cd myproject
pip install -r requirements.txt
```

The fine-tuned model are produced with `torch==1.12.1`.

## Dataset 

The 2023 “12345” Dataset 