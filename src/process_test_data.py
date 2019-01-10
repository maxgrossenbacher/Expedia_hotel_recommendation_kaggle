import pandas as pd

from config import *

test_df = pd.read_csv(TEST)
test_df = feature_generation(test_df)
test_df.to_json('../data/test.json', orient='values')
