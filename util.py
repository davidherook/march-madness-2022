import os
import json
import time
import joblib

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
    filepath = get_model_path(model_dir)
    return joblib.load(filepath)

def save_evaluation():
    pass