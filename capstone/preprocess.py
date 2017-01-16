import numpy as np
import pandas as pd


def make_data_removed_one_hot():
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')
    test['loss'] = np.nan
    train_test = pd.concat([train, test])

    for column in list(train.select_dtypes(include=['object']).columns):

        # remove categorical values existing only in train data or test data
        if train[column].nunique() != test[column].nunique():
            set_train = set(train[column].unique())
            set_test = set(test[column].unique())
            remove_train = set_train - set_test
            remove_test = set_test - set_train
            remove = remove_train.union(remove_test)

            def filter_cat(x):
                if x in remove:
                    return np.nan
                return x

            train_test[column] = train_test[column].apply(lambda x: filter_cat(x), 1)

    # one-hot encode
    train_test_removed_one_hot = pd.get_dummies(train_test)

    train_removed_one_hot = \
        train_test_removed_one_hot[train_test_removed_one_hot['loss'].notnull()]
    test_removed_one_hot = \
        train_test_removed_one_hot[train_test_removed_one_hot['loss'].isnull()]

    train_removed_one_hot.to_csv('./data/train_removed_one_hot.csv', index=False)
    test_removed_one_hot.to_csv('./data/test_removed_one_hot.csv', index=False)


def make_data_removed_and_factorized():
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')
    test['loss'] = np.nan
    train_test = pd.concat([train, test])

    for column in list(train.select_dtypes(include=['object']).columns):

        # remove categorical values existing only in train data or test data
        if train[column].nunique() != test[column].nunique():
            set_train = set(train[column].unique())
            set_test = set(test[column].unique())
            remove_train = set_train - set_test
            remove_test = set_test - set_train
            remove = remove_train.union(remove_test)

            def filter_cat(x):
                if x in remove:
                    return np.nan
                return x

            train_test[column] = train_test[column].apply(lambda x: filter_cat(x), 1)

        # Factorize the categorical features in lexicographical order
        train_test[column] = pd.factorize(train_test[column].values, sort=True)[0]

    train = train_test[train_test['loss'].notnull()]
    test = train_test[train_test['loss'].isnull()]

    # log(target + shift) transform for the train data
    shift = 200
    train['loss'] = np.log(train['loss'] + shift)

    # Create preprocessed data
    train.to_csv('data/train_preprocessed.csv', index=False)
    test.to_csv('data/test_preprocessed.csv', index=False)


if __name__ == '__main__':
    make_data_removed_and_factorized()
    make_data_removed_one_hot()
