################################################
# python generate_dataset.py --stage s1
#
# Stage 1
# Train / Valdate = tournaments before 2016
# Test = tournaments from 2016 to 2021
#
# Stage 2
# Train / Valdate = tournaments before 2022
# Test = tournament 2022
################################################
import os
import re
import argparse
import pandas as pd

VALIDATION_SET_SIZE = .20

def get_seed(x):
    return int(re.search('\d+',str(x)).group(0))

def generate_test_set(submission_sample, seed_data):
    """Use the submission sample and join the features we need to make predictions"""
    submission_sample['Season'] = submission_sample['ID'].apply(lambda x: x.split('_')[0]).astype(int)
    submission_sample['team_a'] = submission_sample['ID'].apply(lambda x: x.split('_')[1]).astype(int)
    submission_sample['team_b'] = submission_sample['ID'].apply(lambda x: x.split('_')[2]).astype(int)
    test = pd.merge(submission_sample, seed_data, how='left', left_on=['Season', 'team_a'], right_on=['Season', 'TeamID'])
    test1 = pd.merge(test, seed_data, how='left', left_on=['Season', 'team_b'], right_on=['Season', 'TeamID'])
    test1.rename({'Seed_x': 'team_a_seed', 'Seed_y': 'team_b_seed'}, axis = 1, inplace = True)
    test1['team_a_seed_num'] = test1['team_a_seed'].apply(get_seed)
    test1['team_b_seed_num'] = test1['team_b_seed'].apply(get_seed)
    return test1

def convert_tourney_results_to_kaggle_format(tr):
    tr['winning_team_is_lower_id'] = tr['WTeamID'] < tr['LTeamID']
    tr['team_a'] = tr['LTeamID']
    tr['team_b'] = tr['WTeamID']
    tr['score_a'] = tr['LScore']
    tr['score_b'] = tr['WScore']
    tr.loc[tr['winning_team_is_lower_id'], 'team_a'] = tr['WTeamID']
    tr.loc[tr['winning_team_is_lower_id'], 'team_b'] = tr['LTeamID']
    tr.loc[tr['winning_team_is_lower_id'], 'score_a'] = tr['WScore']
    tr.loc[tr['winning_team_is_lower_id'], 'score_b'] = tr['LScore']
    assert ( tr['team_a'] + tr['team_b'] ).equals( tr['WTeamID'] + tr['LTeamID'] )
    assert ( tr['team_a'] < tr['team_b'] ).all()
    tr['ID'] = tr[['Season', 'team_a', 'team_b']].astype(str).agg('_'.join, axis = 1)
    tr['result'] = tr['winning_team_is_lower_id'].astype(int)
    return tr

def join_seed_features(tr, seed_data):
    seed_a = pd.merge(tr, seed_data, how = 'left', left_on = ['Season', 'team_a'], right_on = ['Season', 'TeamID'])
    seed_b = pd.merge(tr, seed_data, how = 'left', left_on = ['Season', 'team_b'], right_on = ['Season', 'TeamID'])
    tr['team_a_seed'] = seed_a['Seed']
    tr['team_b_seed'] = seed_b['Seed']
    tr['team_a_seed_num'] = tr['team_a_seed'].apply(get_seed)
    tr['team_b_seed_num'] = tr['team_b_seed'].apply(get_seed)
    return tr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--stage', type=str, choices = ['s1', 's2'], help='Which stage of the competition?')
    args = vars(parser.parse_args())
    stage = args['stage']

    DATA_DIR = 'data/MDataFiles_Stage2' if stage == 's2' else 'data/MDataFiles_Stage1'
    GENERATED_DATA_DIR = 'data/datasets_stage2' if stage == 's2' else 'data/datasets_stage1'
    TOURNEY_RESULTS_PATH = os.path.join(DATA_DIR, 'MNCAATourneyCompactResults.csv')
    TOURNEY_SEEDS_PATH = os.path.join(DATA_DIR, 'MNCAATourneySeeds.csv')
    TEAMS = os.path.join(DATA_DIR, 'MTeams.csv')
    SAMPLE_SUBMISSION_PATH = os.path.join(DATA_DIR, 'MSampleSubmissionStage2.csv' if stage == 's2' else 'MSampleSubmissionStage1.csv')

    tr = pd.read_csv(TOURNEY_RESULTS_PATH)
    ts = pd.read_csv(TOURNEY_SEEDS_PATH)
    ss = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    t = pd.read_csv(TEAMS)

    print(f"\nConverting {TOURNEY_RESULTS_PATH} to required Kaggle format...")
    tr1 = convert_tourney_results_to_kaggle_format(tr)

    print(f"Joining {TOURNEY_SEEDS_PATH} data...")
    tr2 = join_seed_features(tr1, ts)

    print(f"Generating training, validation, and test sets...")
    all_data = tr2.copy()

    test_ids = ss['ID']
    train = all_data[~all_data['ID'].isin(test_ids)]
    validate = train.sample(int(VALIDATION_SET_SIZE * train.shape[0]), random_state = 42)
    train = train.drop(validate.index)
    test = generate_test_set(ss, ts)
    assert set(train.index).isdisjoint(set(validate.index))
    assert test.shape[0] == ss.shape[0]

    if not os.path.isdir(GENERATED_DATA_DIR):
        os.mkdir(GENERATED_DATA_DIR)

    ALL_DATA = os.path.join(GENERATED_DATA_DIR, 'all_data.csv')
    TRAINING_SET = os.path.join(GENERATED_DATA_DIR, 'training_data.csv')
    VALIDATION_SET = os.path.join(GENERATED_DATA_DIR, 'validation_data.csv')
    TEST_SET = os.path.join(GENERATED_DATA_DIR, 'test_data.csv')

    all_data.to_csv(ALL_DATA)
    train.to_csv(TRAINING_SET)
    validate.to_csv(VALIDATION_SET)
    test.to_csv(TEST_SET)

    print(f"\nDatasets have been saved to {GENERATED_DATA_DIR}.")
    print(f"All Data: {all_data.shape}")
    print(f"Training Data: {train.shape}")
    print(f"Validation Data: {validate.shape}")
    print(f"Test Data: {test.shape}\n")