# python train.py --stage s1 --config config.yaml

import os
import yaml
import argparse
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import log_loss

from features import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from util import save_model, generate_hash, load_config, save_config, save_evaluation

MODEL_HASH = generate_hash()
MODEL_BASE_DIR = 'model'
EVALUATION_BASE_DIR = 'evaluation'
MODEL_DIR = os.path.join(MODEL_BASE_DIR, MODEL_HASH)
EVALUATION_DIR = os.path.join(EVALUATION_BASE_DIR, MODEL_HASH)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--stage', type=str, choices = ['s1', 's2'], help='Which stage of the competition?')
    parser.add_argument('-c', '--config', type=str, help='The config file')
    args = vars(parser.parse_args())
    stage = args['stage']
    config_path = args['config']

    GENERATED_DATA_DIR = 'data/datasets_stage2' if stage == 's2' else 'data/datasets_stage1'
    TRAINING_SET = os.path.join(GENERATED_DATA_DIR, 'training_data.csv')
    VALIDATION_SET = os.path.join(GENERATED_DATA_DIR, 'validation_data.csv')

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    features = config['features']
    target = config['target']

    # Read datasets
    train = pd.read_csv(TRAINING_SET)
    val = pd.read_csv(VALIDATION_SET)
    X, y = train[features], train[target]
    X_val, y_val = val[features], val[target]

    print(f"\nTraining on {train.shape[0]} records using {len(features)} features")

    # Train a model
    numeric_transformer = Pipeline(steps = [
        ('imputer', SimpleImputer(strategy = 'mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps = [
        ('imputer', SimpleImputer(strategy = 'constant')),
        ('one-hot', OneHotEncoder(handle_unknown = 'ignore'))
    ])

    transformer = ColumnTransformer(transformers = [
        ('numeric', numeric_transformer, NUMERIC_FEATURES),
        ('categorical', categorical_transformer, CATEGORICAL_FEATURES)
    ])

    clf = MLPClassifier(
        hidden_layer_sizes = (16, 16, 8, 8, ),
        random_state = 777
    )

    pipeline = Pipeline(steps = [
        ('transformer', transformer),
        ('classifier', clf)
    ])

    pipeline.fit(X, y)

    # Save model
    os.mkdir(MODEL_DIR)
    save_model(clf, MODEL_DIR)
    save_config(config, MODEL_DIR)

    # Save Evalation
    os.mkdir(EVALUATION_DIR)
    save_evaluation()

    # TODO
    # Evaluate
    print(f"Training Accuracy = {pipeline.score(X, y)}")
    print(f"Validation Accuracy = {pipeline.score(X_val, y_val)}")
    print(f"Validation LogLoss = {log_loss(y_val, pipeline.predict_proba(X_val)[:, 1])}")