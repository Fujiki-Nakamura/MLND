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


### Reproduction
#### XGBoost models
- Hyper parameter tuning and cross validation
```shell
$ cd /xgb/xgb_cv/
```
Tune the hyper parameters in `parameters.py` and then
```
$ python xgb_cv.py
```

- The best XGBoost model
```shell
$ cd /xgb/xgb_v3/
$ python stacking.py xgb_v3
```

- Another XGBoost model for stacking
```shell
$ cd /xgb/xgb_v2/
$ python stacking.py xgb_v2
```

#### Neural Network models
- 2-layer Neural Network for a benchmark
```shell
$ cd 2_layer_v1/
$ python stacking.py 2_layer_v1
```

- 3-layer Neural Network
```shell
$ cd 3_layer_v1/
$ python stacking.py 3_layer_v1
```

- 4-layer Neural Network
```shell
$ cd 4_layer_v1/
$ python stacking.py 4_layer_v1
```

#### Linear Regression for Level 2 model in stacking
Execute cells in `level_2_model/linear_regression.ipynb`.
