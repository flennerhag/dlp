"""
Test run on the Zillow competition
"""
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error

from deeplearn.train import Trainer
from deeplearn.opts import Nesterov
from deeplearn.networks import Sequential
from deeplearn.viz import plot_train_scores


def build_train():
    """Read in training data and return input, output, columns tuple."""

    df = pd.read_csv('../kaggle/zillow/input/train_2016.csv')
    prop = pd.read_csv('../kaggle/zillow/input/properties_2016.csv')
    convert = prop.dtypes == 'float64'
    prop.loc[:, convert] = \
        prop.loc[:, convert].apply(lambda x: x.astype(np.float32))

    df = df.merge(prop, how='left', on='parcelid')

    y = df['logerror'].values

    df = df.drop(['parcelid',
                  'logerror',
                  'transactiondate',
                  'taxdelinquencyflag',
                  'propertyzoningdesc',
                  'propertycountylandusecode'], axis=1)

    convert = df.dtypes == 'object'
    df.loc[:, convert] = \
        df.loc[:, convert].apply(lambda x: x == True)

    cols = df.columns

    df.fillna(0, inplace=True)

    return df, y, cols


def build_model(drop, activation, normalize):
    """Build Neural Network."""

    net = Sequential()
    net.add_fc(54, 200,
               dropout=drop,
               activation=activation)


    net.add_fc(200, 100,
               normalize=normalize,
               dropout=drop,
               activation=activation)


    net.add_fc(100, 50,
               normalize=normalize,
               dropout=drop,
               activation=activation)

    net.add_fc(50, 20,
               normalize=normalize,
               dropout=drop,
               activation=activation)

    net.add_fc(20, 100,
               normalize=normalize,
               dropout=drop,
               activation=activation)

    net.add_fc(100, 1, bias=False,
               activation=activation)

    net.add_cost("norm")

    return net

def mae(y, p):
    try:
        return mean_absolute_error(y, p) * 100
    except ValueError:
        return -100



def train_model(m, xtrain, ytrain):
    opt = Nesterov(m, lr=1e-5, u=0.99)
    trainer = Trainer(m,
                      opt, shuffle=False,
                      batch_size=50,
                      eval_size=50,
                      eval_ival=100,
                      eval_metric=mae,
                      )

    xtrain, ytrain = shuffle(xtrain, ytrain)

    trainer.train(xtrain, ytrain, 200000)
    plot_train_scores(trainer)


def shuffle(X, y):
    idx = np.random.permutation(X.shape[0])
    X = X[idx]
    y = y[idx]
    return X, y


def build_test(train_columns):
    """Read in test data and return input array for prediction."""
    prop = pd.read_csv('../input/properties_2016.csv')
    convert = prop.dtypes == 'float64'
    prop.loc[:, convert] = \
        prop.loc[:, convert].apply(lambda x: x.astype(np.float32))

    df = pd.read_csv("../input/sample_submission.csv")
    df['parcelid'] = df['ParcelId']

    df = df.loc[:, ['parcelid']].merge(prop, on='parcelid', how='left')
    df = df.loc[:, train_columns]

    convert = df.dtypes == 'object'
    df.loc[:, convert] = \
        df.loc[:, convert].apply(lambda x: x == True)

    df.fillna(0, inplace=True)

    return df.values

def gen_predictions(model, train_columns):
    """Wrapper around data generation to build test data and generate preds."""
    xtest = build_test(train_columns)

    pred = model.predict(xtest)

    sub = pd.read_csv('input/sample_submission.csv')
    sub = sub.apply(lambda x: pred if x.name != "ParcelId" else x.values)

    sub.to_csv('../output/ens_starter.csv', index=False, float_format='%.4f')

###############################################################################
if __name__ == "__main__":
    X, y, cols = build_train()

    X = X.values * 1
    X = X.astype(np.float32)
#    X_m = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_m = (X - X.mean(axis=0)) / X.std(axis=0)
    X_m, y = shuffle(X_m, y)

    m = build_model(False, "relu", False)
    train_model(m, X_m[:100], y[:100])
