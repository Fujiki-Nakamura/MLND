## Capstone project
### How to Start
```shell
git clone https://github.com/Fujiki-Nakamura/MLND.git
```
```shell
cd MLND/capstone/
```
```shell
mkdir data
```

## Requirements
This project requires [Allstate Claims Severity competition](https://www.kaggle.com/c/allstate-claims-severity/) data.
<br>Download data from [here](https://www.kaggle.com/c/allstate-claims-severity/data) and place them under the `data/` directry.
<br>If you have data correctly, it should look like:
<br>`data/train.csv`
<br>`data/test.csv`

## Preprocess
```shell
cd capstone
```
```shell
python preprocess.py
```
This creates preprocessed data in `data/` directory.
