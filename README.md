# ml_steels_repo
Repository for "Machine learning assisted design of reactor steels with high long-term strength and toughness" article.

Here we provide trained models, subsets of data and inference script. Training scripts and whole database are available upon reasonable request to the authors.

## Install
We recommend using python 3.9 or higher

Install dependencies using:
```
pip install -r requirements.txt
```

## Run inference
```
python infer.py \
--target <target name> \
--model_type <specific or general> \
--data-path <path to data.csv> 