#!/usr/bin/env python
"""main.py: Training/Applying script for XGBoost Model for Exepdia Kaggle Competition """

# Package imports
import os

# Project imports
import helper
from model import ClassificationModel
from config import *

__author__ = "Max Grossenbacher"
__credits__ = ["Max Grossenbacher"]
__version__ = "1.0.1"
__maintainer__ = "Max Grossenbacher"
__email__ = "max.grossenbacher@delvedeeper.com"
__status__ = "Production"

def main(downsample=True):
    print('[main]: running main.py')
    # Train model
    print('[config]: train flag: ', TRAIN_FLAG)
    if TRAIN_FLAG == True:
        # Loading data
        print('[config]: destinations csv path: ', DESTINATION)
        destination_df = helper.read_data_local(DESTINATION)
        print('[config]: train csv path: ', TRAIN)
        train_df = helper.read_data_local(TRAIN)
        print('[config]: test csv path: ', TEST)
        test_df = helper.read_data_local(TEST)

        # Sample data
        if downsample:
            print('[main]: downsampling training data for quicker training')
            train_df = helper.downsample_training_data(train_df, sample_size=10000)

        # Feature generation
        print('[main]: feature generation')
        destination_df = helper.pca_features(destination_df, components=4)
        train_df = helper.feature_generation(
            train_df, destination_df)  # feature generation on train_df
        test_df = helper.feature_generation(
            test_df, destination_df)  # feature generation on test_df
        print('[main]: train-validation split')
        train_df, validation_df = helper.train_validation_split(train_df)
        print('[main]: training model')
        # Train new model using ClassificationModel class
        m = ClassificationModel(model_folder=MODEL_DIR,
                                model_params=MODEL_PARAMS,
                                model_name=MODEL_NAME)
        m.load_train_data(data=train_df, target='hotel_cluster')
        m.load_validation_data(data=validation_df, target='hotel_cluster')
        m.load_test_data(data=test_df, target='hotel_cluster')
        # Train model
        m.simple_train(eval_metric=['mlogloss'])
        # Validate model
        preds_probability = m.model.predict_proba(m.X_val[m.features])
        sorted_probabilities = helper.get_top_n_class_predictions(
            m.model, preds_probability, n=5)
        val_actual = [[l] for l in m.y_val['hotel_cluster']]
        score = helper.mapk(val_actual, sorted_probabilities, k=5)
        print('Model Mean Absolute Precision: ', score)

    # Apply model
    elif TRAIN_FLAG == False and os.path.isfile(MODEL_DIR+MODEL_NAME):
        # Loading data
        print('[config]: destinations csv path: ', DESTINATION)
        destination_df = helper.read_data_local(DESTINATION)
        print('[config]: test csv path: ', TEST)
        test_df = helper.read_data_local(TEST)
        # Feature generation
        print('[main]: feature generation')
        destination_df = helper.pca_features(destination_df, components=4)
        test_df = helper.feature_generation(
            test_df, destination_df)  # feature generation on test_df
        # Spliting off target
        y_test = test_df.pop('hotel_cluster')
        X_test = test_df
        print('[main]: applying model')
        # Apply trained model
        m = ClassificationModel(model_folder=MODEL_DIR,
                                model_params=None)
        # Loading trained model
        m.load_model(MODEL_DIR+MODEL_NAME)
        # Validate model
        preds_probability = m.apply(X_test)
        sorted_probabilities = helper.get_top_n_class_predictions(
            m.model, X_test, y_test, n=5)
        test_actual = [[l] for l in y_test['hotel_cluster']]
        score = helper.mapk(test_actual, sorted_probabilities, k=5)
        print('Model Mean Absolute Precision: ', score)
    else:
        print('[main]: MODEL does not exist')

if __name__ == "__main__":
    # print(__main__.__author__)
    main()
    
    
