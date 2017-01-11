# coding: UTF-8
fair_obj_constant = 2


def load_data(path_to_train, path_to_test_data):
    train = pd.read_csv(path_to_train_data)
    test = pd.read_csv(path_to_test_data)

    return train, test


def create_dtrain_dtest(train, test):
    y = train['loss']
    X = train.drop(['loss', 'id'], axis=1)
    X_test = test.drop(['loss', 'id'], axis=1)

    dtrain = xgb.DMatrix(X, label=y)
    dtest = xgb.DMatrix(X_test)

    return dtrain, dtest


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(preds), np.exp(labels))


def fair_objective(preds, dtrain):
    labels = dtrain.get_label()
    con = fair_obj_constant
    x = preds - labels
    grad = con * x / (np.abs(x) + con)
    hess = con**2 / (np.abs(x) + con)**2
    return grad, hess
