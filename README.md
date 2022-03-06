# Kaggle March Madness 2022
Predict the outcome for each game of the 2022 March Madness NCAA Tournament for the [competition hosted on Kaggle](https://www.kaggle.com/c/mens-march-mania-2022)

## Getting Started

Clone the repository, create a virtual environment, and download requirements:
```
$ git clone https://github.com/davidherook/march-madness-2022.git
$ cd march-madness-2022
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
$ mkdir data model evaluation submission tableau
```

## Downloading Data
Download the data [provided by Kaggle](https://www.kaggle.com/c/mens-march-mania-2022/data)
Run `generate_dataset.py` to create training, validation, and test sets:
```
$ python generate_dataset.py

Converting data/MDataFiles_Stage1/MNCAATourneyCompactResults.csv to required Kaggle format...
Joining data/MDataFiles_Stage1/MNCAATourneySeeds.csv data...
Generating training, validation, and test sets...

Datasets have been saved to data/datasets_stage1.
All Data: (2317, 19)
Training Data: (1587, 19)
Validation Data: (396, 19)
Test Data: (11390, 11)
```

## Train a Model
Train a model with a config file which tells the model which features to use. The model and its config are saved to the `model` directory and several resources to evaluate the performance are saved to `evaluation`.
```
$ python train.py --config config.yaml

Training Accuracy = 0.7372400756143668
Validation Accuracy = 0.7095959595959596
```

## Make Predictions
To make predictions, pass the model hash you wish to use and the path to the test set. A new column named `pred` will be appended to the test set and it will be written to the `submission` folder in a subdirectory named with the model hash. Two files will be written: `predictions.csv` with the original test set plus predictions and `predictions_submission.csv` which contains the predictions in the required format for Kaggle submission.

```
$ python predict.py --model 1646588926 --data data/datasets_stage1/test_data.csv

Prediction files have been saved to submission/1646588926
```

# TODO
- evaluation resources
- more features
- different model
- adjust for stage2
