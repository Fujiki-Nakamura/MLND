## Capstone project
### How to Start
```shell
$ git clone https://github.com/Fujiki-Nakamura/MLND.git
$ cd MLND/capstone/
$ mkdir data
```


### Requirements
#### Data
This project requires [Allstate Claims Severity competition](https://www.kaggle.com/c/allstate-claims-severity/) data.
<br>Download data from [here](https://www.kaggle.com/c/allstate-claims-severity/data) and place them under the `data/` directry.
<br>If you have data correctly, it should look like:
```
data/train.csv
data/test.csv
```

#### Environment
```shell
$ conda env create --file conda_env.yaml
```


### Data Preparation
```shell
$ python preprocess.py
```
Based on the train and test data you got above, this creates preprocessed data in `data/` directory.
<br>They are necessary for Machine Learning models.
