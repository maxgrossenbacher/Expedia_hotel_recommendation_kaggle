import pandas as pd 
import numpy as np
import random
import pickle
import xgboost as xgb
import matplotlib.pyplot as plot
from google.cloud import storage
from io import BytesIO
from sklearn.decomposition import PCA
try:
    from StringIO import StringIO
except:
    from io import BytesIO as StringIO


def read_data_local(filename):
    df = pd.read_csv(filename)
    if "date_time" in df.columns.tolist():
        df["date_time"] = pd.to_datetime(df["date_time"])
        df["year"] = df["date_time"].dt.year
        df["month"] = df["date_time"].dt.month
        return df
    else:
        print('\t[helper:read_data_local]: no data_time column')
        return df

def read_data_cloud(filename):
    # df = pd.read_csv(filename)
    if "date_time" in df.columns.tolist():
        df["date_time"] = pd.to_datetime(df["date_time"])
        df["year"] = df["date_time"].dt.year
        df["month"] = df["date_time"].dt.month
        return df
    else:
        print('\t[helper:read_data_cloud]: no data_time column')
        return df

def downsample_training_data(df, sample_size=10000):
    print('\t[helper:downsample_training_data]: downsample to {} samples'.format(
        sample_size))
    unique_users = set(df['user_id'].unique())
    sel_user_ids = random.sample(unique_users, sample_size)
    df = df[df.user_id.isin(sel_user_ids)]
    df.to_csv('../data/sampled_data.csv')
    return df

def pca_features(df, components=4):
    print('\t[helper:pca_features]: fitting PCA features')
    pca = PCA(n_components=components)
    d_pca = pca.fit_transform(
        df[['d{0}'.format(i+1) for i in range(149)]])
    pca_df = pd.DataFrame(d_pca)
    pca_df['srch_destination_id'] = df['srch_destination_id']
    return pca_df

def feature_generation(df, pca_df):
    print('\t[helper:feature_generation]: generating features and merging dfs')
    df["date_time"] = pd.to_datetime(df["date_time"])
    df["srch_ci"] = pd.to_datetime(
        df["srch_ci"], format='%Y-%m-%d', errors="coerce")
    df["srch_co"] = pd.to_datetime(
        df["srch_co"], format='%Y-%m-%d', errors="coerce")

    props = {}
    for prop in ["month", "day", "hour", "minute", "dayofweek", "quarter"]:
        props[prop] = getattr(df["date_time"].dt, prop)

    carryover = [p for p in df.columns if p not in [
        "date_time", "srch_ci", "srch_co"]]
    for prop in carryover:
        props[prop] = df[prop]

    date_props = ["month", "day", "dayofweek", "quarter"]
    for prop in date_props:
        props["ci_{0}".format(prop)] = getattr(df["srch_ci"].dt, prop)
        props["co_{0}".format(prop)] = getattr(df["srch_co"].dt, prop)
    props["stay_span"] = (df["srch_co"] - df["srch_ci"]
                          ).astype('timedelta64[h]')
    ret = pd.DataFrame(props)
    ret = ret.join(pca_df, on="srch_destination_id",
                   how='left', rsuffix="dest")
    ret = ret.drop("srch_destination_iddest", axis=1)
    return ret

def train_validation_split(df):
    print('\t[helper:train_validation_split]: creating train-validation split')
    train_df = df[((df.year == 2013) | ((df.year == 2014) & (df.month < 7)))]
        # t2 is new test set for validation of model
    validation_df = df[((df.year == 2014) & (df.month >= 7))]
    validation_df = validation_df[validation_df['is_booking'] == True]
    return train_df, validation_df

def get_top_n_class_predictions(model, preds_probability, n=5):
    results_df = pd.DataFrame(
        columns=model.classes_, data=preds_probability)
    sorted_probabilities = np.flip(results_df.values.argsort(), 1)
    return sorted_probabilities[:,:n]

# functions for evaluation metrics (mean absolute precision)
def apk(actual, predicted, k=5):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=5):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])
