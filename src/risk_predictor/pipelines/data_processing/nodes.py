import numpy as np
import pandas as pd
from scipy.stats import skew

from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


def remove_outliers(train, parameters):
    train = train.copy()
    train = train.drop(train[(train['GrLivArea'] > parameters['outliers']['GrLivArea']) &
                             (train['SalePrice'] < parameters['outliers']['SalePrice'])]
                       .index)
    return train


def fill_na(train, parameters):
    train = train.copy()
    train[parameters['none_cols']] = train[parameters['none_cols']].fillna("None")
    train[parameters['zero_cols']] = train[parameters['zero_cols']].fillna(0)

    train["LotFrontage"] = train.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))

    impute_int = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    impute_str = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    int_cols = train.select_dtypes(include='number').columns
    str_cols = train.select_dtypes(exclude='number').columns

    train[int_cols] = impute_int.fit_transform(train[int_cols])
    train[str_cols] = impute_str.fit_transform(train[str_cols])

    return train


def total_sf(train):
    train = train.copy()
    train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
    return train


def to_str(train, parameters):
    train = train.copy()
    train[parameters['str_cols']]=train[parameters['str_cols']].astype(str)
    return train


def create_skew_table(train):
    numeric_feats = train.dtypes[train.dtypes != "object"].index

    # Check the skew of all numerical features
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew' :skewed_feats})
    skewness = skewness.drop('SalePrice',axis=0)
    return skewness


def fix_skew(train):
    skew_table = create_skew_table(train)
    skewed_feats = skew_table[abs(skew_table) > 0.75].dropna().index
    skewed_feats = [i for i in skewed_feats if i not in list(train.dtypes[train.dtypes == 'object'].index)]
    skew_yj = PowerTransformer(method='yeo-johnson')
    train[skewed_feats] = skew_yj.fit_transform(train[skewed_feats])

    return train


def one_hot(train, parameters):
    ohe_cols = [i for i in list(train.dtypes[train.dtypes == 'object'].index) if i not in parameters['label_enc_cols']]
    train[ohe_cols] = train[ohe_cols].astype(str)

    column_trans = ColumnTransformer(
        [('one_hot', OneHotEncoder(), ohe_cols)]
        , remainder='passthrough', sparse_threshold=0)

    train = pd.DataFrame(column_trans.fit_transform(train), columns=column_trans.get_feature_names())

    return train


def to_float(train):
    train = train.copy()
    train = train.astype(float)
    return train


def house_prices_clean(train, parameters):
    train = train.copy()
    train = to_str(train, parameters)
    train = fix_skew(train)
    train = one_hot(train, parameters)
    train = train.astype(float)
    return train


