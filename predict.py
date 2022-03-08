# python predict.py --model 1646588926 --data data/datasets_stage1/test_data.csv

import os
import argparse
import pandas as pd
from util import load_model, load_config
from train import MODEL_BASE_DIR

SUBMISSION_BASE_DIR = 'submission'
SUBMISSION_COLS = ['ID', 'pred']

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='The trained model hash to use for predictions')
    parser.add_argument('-d', '--data', type=str, help='The data to generate predictions for')
    args = vars(parser.parse_args())
    model_hash = args['model']
    test_data = args['data']

    model_dir = os.path.join(MODEL_BASE_DIR, model_hash)
    submission_dir = os.path.join(SUBMISSION_BASE_DIR, model_hash)
    predictions_path = os.path.join(submission_dir, 'predictions.csv')
    predictions_submission_path = os.path.join(submission_dir, 'predictions_submission.csv')
    if not os.path.isdir(submission_dir):
        os.mkdir(submission_dir)

    config = load_config(model_dir)
    model = load_model(model_dir)
    features = config['features']

    test = pd.read_csv(test_data)
    X = test[features]

    pred = model.predict_proba(X)[:, 1]
    test['pred'] = pred

    test.to_csv(predictions_path, index = False)
    test[SUBMISSION_COLS].to_csv(predictions_submission_path, index = False)
    print(f"\nPrediction files have been saved to {submission_dir}")