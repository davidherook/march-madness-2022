# python train.py --config config.yaml
import os
import yaml
import argparse
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from generate_dataset import TRAINING_SET, VALIDATION_SET
from util import save_model, generate_hash, load_config, save_config, save_evaluation

MODEL_HASH = generate_hash()
MODEL_BASE_DIR = 'model'
EVALUATION_BASE_DIR = 'evaluation'
MODEL_DIR = os.path.join(MODEL_BASE_DIR, MODEL_HASH)
EVALUATION_DIR = os.path.join(EVALUATION_BASE_DIR, MODEL_HASH)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='The config file')
    args = vars(parser.parse_args())
    config_path = args['config']

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    features = config['features']
    target = config['target']

    # Read datasets
    train = pd.read_csv(TRAINING_SET)
    val = pd.read_csv(VALIDATION_SET)
    X, y = train[features], train[target]
    X_val, y_val = val[features], val[target]

    # Train a model
    clf = RandomForestClassifier()
    clf.fit(X, y)

    # Save model
    os.mkdir(MODEL_DIR)
    save_model(clf, MODEL_DIR)
    save_config(config, MODEL_DIR)

    # Save Evalation
    os.mkdir(EVALUATION_DIR)
    save_evaluation()

    # Evaluate
    print(f"Training Accuracy = {clf.score(X, y)}")
    print(f"Validation Accuracy = {clf.score(X_val, y_val)}")