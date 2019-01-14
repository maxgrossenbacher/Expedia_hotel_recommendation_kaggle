# Data Location
DESTINATION = '../data/destinations.csv'
TRAIN = '../data/train.csv'
TEST = '../data/train.csv'

# Model Training/Apply
TRAIN_FLAG = True
MODEL_DIR = '../models/'
MODEL_NAME = 'V3_model.pkl' # can be empty if TRAIN_FLAG == True
MODEL_PARAMS = {'n_estimators':10000,
                'n_jobs':-1}
