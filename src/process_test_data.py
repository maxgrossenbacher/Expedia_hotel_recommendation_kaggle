import pandas as pd

from config import *

test_df = pd.read_csv(TEST)
test_df = feature_generation(test_df)
actual = test_df.pop('hotel_clusters')
test_df.to_json('../data/test.json', orient='values')
actual.to_json('../data/actual_values.json', orient='values')