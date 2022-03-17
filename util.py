import os
import json
import time
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

def generate_hash():
    t = int(time.time())
    return f"{t}"

def get_config_path(model_dir):
    return os.path.join(model_dir, 'config.json')

def get_model_path(model_dir):
    return os.path.join(model_dir, 'model.pk')

def save_config(config, model_dir):
    filepath = get_config_path(model_dir)
    with open(filepath, 'w') as json_file:
        json.dump(config, json_file, indent=4)

def load_config(model_dir):
    filepath = get_config_path(model_dir)
    with open(filepath, 'r') as json_file:
        return json.load(json_file)

def save_model(model, model_dir):
    filepath = get_model_path(model_dir)
    joblib.dump(model, filepath, compress=0)

def load_model(model_dir):
    print(f"Loading model from {model_dir}")
    filepath = get_model_path(model_dir)
    return joblib.load(filepath)

def pretty_confusion_matrix(y_true, y_pred):
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred))
    cm.columns = ['Predicted {}'.format(c) for c in cm.columns]
    cm.index = ['Actual {}'.format(c) for c in cm.index]
    return cm

def get_classifier_from_config(clf_name_str, parameters_dict, _globals):
    return _globals[clf_name_str](**parameters_dict)