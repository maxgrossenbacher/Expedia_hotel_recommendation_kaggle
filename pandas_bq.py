from google.datalab import Context
import google.datalab.bigquery as bq
import google.datalab.storage as storage
import pandas as pd
try:
  from StringIO import StringIO
except ImportError:
  from io import BytesIO as StringIO

# Reading from Cloud Storage Bucket Destinations
content = storage.Object(
    'expedia_kaggle_max', 'destinations.csv').read_stream()
destination_df = pd.read_csv(StringIO(content))
destination_df.head()

# Reading from Cloud Storage Bucket Test
content = storage.Object(
    'expedia_kaggle_max', 'test.csv').read_stream()
test_df = pd.read_csv(StringIO(content))
test_df.head()

test_df["date_time"] = pd.to_datetime(test_df["date_time"])
test_df["year"] = test_df["date_time"].dt.year
test_df["month"] = test_df["date_time"].dt.month

# Reading from Cloud Storage Bucket Train
content = storage.Object(
    'expedia_kaggle_max', 'train.csv').read_stream()
train_df = pd.read_csv(StringIO(content))
train_df.head()

train_df["date_time"] = pd.to_datetime(train_df["date_time"])
train_df["year"] = train_df["date_time"].dt.year
train_df["month"] = train_df["date_time"].dt.month

print('{} unique users training data'.format(train_df['user_id'].nunique()))

#check is user_ids in test is the same as user_ids in train
train_ids = set(train_df['user_id'].unique())
test_ids = set(test_df['user_id'].unique())
intersection_count = len(train_ids & test_ids)
print('{}: the user_id in train set the same as the user_id in test set.' .format(
    intersection_count == len(test_ids)))

# check the date in train vs test set
print('The min date in the train set is {}.'.format(train_df['year'].min()))
print('The max date in the train set is {}.'.format(train_df['year'].max()))
print('The min date in the test set is {}.'.format(test_df['year'].min()))
print('The max date in the test set is {}.'.format(test_df['year'].max()))
