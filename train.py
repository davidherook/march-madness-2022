# python train.py --stage s2 --config config.yaml
# python train.py --stage s2 --config config.yaml --gridsearch

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
from util import save_model, generate_hash, load_config, save_config, pretty_confusion_matrix, \
    get_classifier_from_config
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

MODEL_HASH = generate_hash()
MODEL_BASE_DIR = 'model'
MODEL_DIR = os.path.join(MODEL_BASE_DIR, MODEL_HASH)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--stage', type = str, choices = ['s1', 's2'], help = 'Which stage of the competition?')
    parser.add_argument('-c', '--config', type = str, help = 'The config file')
    parser.add_argument('-g', '--gridsearch', action = 'store_true', help='Run gridsearch?')
    args = vars(parser.parse_args())
    stage = args['stage']
    config_path = args['config']
    gridsearch = args['gridsearch']

    GENERATED_DATA_DIR = 'data/datasets_stage2' if stage == 's2' else 'data/datasets_stage1'
    TRAINING_SET = os.path.join(GENERATED_DATA_DIR, 'training_data.csv')
    VALIDATION_SET = os.path.join(GENERATED_DATA_DIR, 'validation_data.csv')

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    features = config['features']
    target = config['target']
    classifier = get_classifier_from_config(config['classifier'], config['parameters'], globals())

    NUMERIC_FEATURES = [f for f in NUMERIC_FEATURES if f in features]
    CATEGORICAL_FEATURES = [f for f in CATEGORICAL_FEATURES if f in features]

    # Read datasets
    train = pd.read_csv(TRAINING_SET)
    val = pd.read_csv(VALIDATION_SET)
    X, y = train[features], train[target]
    X_val, y_val = val[features], val[target]

    print(f"\nTraining on {train.shape[0]} records using {len(features)} features")
    print(f"Using the features {features}")

    # Prepare pipeline
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

    pipeline = Pipeline(steps = [
        ('transformer', transformer),
        ('classifier', classifier)
    ])

    if gridsearch:

        parameters = [
            {
                'classifier': [MLPClassifier()],
                'classifier__hidden_layer_sizes': [
                    (64, 16, 16, ),
                    (16, 16, ),
                    (8, 8, ),
                    (4, )
                ],
                'classifier__solver': ['lbfgs', 'adam']
            },
            {
                'classifier': [LogisticRegression()],
                'classifier__C': [0.1, 1.0, 10.0, 100.0]
            },
            {
                'classifier': [RandomForestClassifier()],
                'classifier__n_estimators': [10, 100]
            }
        ]

        search = GridSearchCV(
            pipeline,
            parameters,
            cv = 4,
            verbose = 1,
            scoring = ['precision', 'recall', 'f1', 'neg_log_loss'],
            refit = 'neg_log_loss'
        )
        search.fit(X, y)

        print(f"\n\nGrid search finished. Best parameters: {search.best_params_}")
        results = pd.DataFrame(search.cv_results_)
        cols = [c for c in results.columns if c.startswith('param_')] + ['mean_test_f1', 'rank_test_f1'] + ['mean_test_neg_log_loss', 'rank_test_neg_log_loss']
        results = results[cols].sort_values('mean_test_neg_log_loss', ascending = False)
        print(results)

    else:

        pipeline.fit(X, y)

        # Evaluate
        decision_threshold = 0.5
        y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba >= decision_threshold).astype(int)

        print(f"\nTraining Accuracy = {pipeline.score(X, y)}")
        print(f"Validation Accuracy = {pipeline.score(X_val, y_val)}")
        print(f"Validation LogLoss = {log_loss(y_val, y_pred_proba)}\n")
        print(pretty_confusion_matrix(y_val, y_pred))

        prec, rec, f1, sup = precision_recall_fscore_support(y_val, y_pred, average = 'binary')
        print(f"Precision: {prec}")
        print(f"Recall: {rec}")
        print(f"F1: {f1}\n")

        # Save model
        os.mkdir(MODEL_DIR)
        save_model(pipeline, MODEL_DIR)
        save_config(config, MODEL_DIR)
        print(f"Model artifacts saved to {MODEL_DIR}\n")