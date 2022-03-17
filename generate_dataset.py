################################################
# python generate_dataset.py --stage s2
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

def generate_test_set(submission_sample, seed_data, reg_season_data, teams_data):
    """Use the submission sample and join the features we need to make predictions"""
    submission_sample['Season'] = submission_sample['ID'].apply(lambda x: x.split('_')[0]).astype(int)
    submission_sample['team_a'] = submission_sample['ID'].apply(lambda x: x.split('_')[1]).astype(int)
    submission_sample['team_b'] = submission_sample['ID'].apply(lambda x: x.split('_')[2]).astype(int)
    test = pd.merge(submission_sample, seed_data, how = 'left', left_on = ['Season', 'team_a'], right_on = ['Season', 'TeamID'])
    test1 = pd.merge(test, seed_data, how = 'left', left_on = ['Season', 'team_b'], right_on = ['Season', 'TeamID'])
    test1.rename({'Seed_x': 'team_a_seed', 'Seed_y': 'team_b_seed'}, axis = 1, inplace = True)
    test1['team_a_seed_num'] = test1['team_a_seed'].apply(get_seed)
    test1['team_b_seed_num'] = test1['team_b_seed'].apply(get_seed)
    test2 = join_reg_season_features(test1, reg_season_data)
    test3 = pd.merge(test2, teams_data, how = 'left', left_on = 'TeamID_a', right_on = 'TeamID')
    test4 = pd.merge(test3, teams_data, how = 'left', left_on = 'TeamID_b', right_on = 'TeamID')
    return test4

def convert_team_order_to_kaggle_format(tr):
    tr['win_team_is_low_id'] = tr['WTeamID'] < tr['LTeamID']
    tr['team_a'] = tr['LTeamID']
    tr['team_b'] = tr['WTeamID']
    tr['score_a'] = tr['LScore']
    tr['score_b'] = tr['WScore']
    tr.loc[tr['win_team_is_low_id'], 'team_a'] = tr['WTeamID']
    tr.loc[tr['win_team_is_low_id'], 'team_b'] = tr['LTeamID']
    tr.loc[tr['win_team_is_low_id'], 'score_a'] = tr['WScore']
    tr.loc[tr['win_team_is_low_id'], 'score_b'] = tr['LScore']
    assert ( tr['team_a'] + tr['team_b'] ).equals( tr['WTeamID'] + tr['LTeamID'] )
    assert ( tr['team_a'] < tr['team_b'] ).all()
    return tr

def convert_tourney_results_to_kaggle_format(tr):
    tr1 = convert_team_order_to_kaggle_format(tr)
    tr1['ID'] = tr1[['Season', 'team_a', 'team_b']].astype(str).agg('_'.join, axis = 1)
    tr1['result'] = tr1['win_team_is_low_id'].astype(int)
    return tr1

def join_seed_features(tourney, seed_data):
    seed_a = pd.merge(tourney, seed_data, how = 'left', left_on = ['Season', 'team_a'], right_on = ['Season', 'TeamID'])
    seed_b = pd.merge(tourney, seed_data, how = 'left', left_on = ['Season', 'team_b'], right_on = ['Season', 'TeamID'])
    tourney['team_a_seed'] = seed_a['Seed']
    tourney['team_b_seed'] = seed_b['Seed']
    tourney['team_a_seed_num'] = tourney['team_a_seed'].apply(get_seed)
    tourney['team_b_seed_num'] = tourney['team_b_seed'].apply(get_seed)
    return tourney

def join_reg_season_features(tourney_t, reg_season_data):
    '''Assumes the tourney data is alreaddy transformed into kaggle format and contains
    the 'team_a' and 'team_b' columns'''

    def flatten_cols(df):
        df.columns = df.columns.to_flat_index()
        df.columns = ['_'.join(c) for c in df.columns]
        df.reset_index(inplace = True)
        return df

    reg_season_data_t = convert_team_order_to_kaggle_format(reg_season_data)

    reg_season_data_t['diff_a'] = reg_season_data_t['score_a'] -  reg_season_data_t['score_b']
    reg_season_data_t['diff_b'] = reg_season_data_t['diff_a'] * -1
    loc_reverse = {'H': 'A', 'A': 'H', 'N': 'N'}
    reg_season_data_t['loc_a'] = reg_season_data_t['WLoc']
    reg_season_data_t.loc[~reg_season_data_t['win_team_is_low_id'], 'loc_a'] = reg_season_data_t['WLoc'].map(loc_reverse)
    reg_season_data_t['loc_b'] = reg_season_data_t['loc_a'].map(loc_reverse)
    reg_season_data_t['is_home_a'] = (reg_season_data_t['loc_a'] == 'H').astype(int)
    reg_season_data_t['is_home_b'] = (reg_season_data_t['loc_b'] == 'H').astype(int)

    # Stack so each record represents a Season, team, DayNum
    cols = ['Season', 'team', 'score', 'diff', 'is_home']
    team_a = reg_season_data_t[['Season', 'team_a', 'score_a', 'diff_a', 'is_home_a']]
    team_b = reg_season_data_t[['Season', 'team_b', 'score_b', 'diff_b', 'is_home_b']]
    team_a.columns = cols
    team_b.columns = cols
    stacked = pd.concat([team_a, team_b])

    # Aggregate at the level of Season, team
    aggs = stacked.groupby(['Season', 'team']).agg(['mean', 'std', 'min', 'max', 'count'])[['score', 'diff', 'is_home']]
    aggs = flatten_cols(aggs)
    aggs['count_games'] = aggs['score_count']
    aggs.drop([c for c in aggs.columns if c.endswith('_count')], axis = 1, inplace = True)
    aggs.drop(['is_home_std', 'is_home_min', 'is_home_max'], axis = 1, inplace = True)

    # Join back to tournament dataset
    tourney_t_aggs_a = pd.merge(tourney_t, aggs, how = 'left', left_on = ['Season', 'team_a'], right_on = ['Season', 'team'])
    tourney_t_aggs_b = pd.merge(tourney_t_aggs_a, aggs, how = 'left', left_on = ['Season', 'team_b'], right_on = ['Season', 'team'])
    tourney_t_aggs_b.columns = [c.replace('_x', '_a').replace('_y', '_b') for c in tourney_t_aggs_b.columns]
    assert tourney_t.shape[0] == tourney_t_aggs_b.shape[0]
    return tourney_t_aggs_b

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--stage', type=str, choices = ['s1', 's2'], help='Which stage of the competition?')
    args = vars(parser.parse_args())
    stage = args['stage']

    DATA_DIR = 'data/MDataFiles_Stage2' if stage == 's2' else 'data/MDataFiles_Stage1'
    GENERATED_DATA_DIR = 'data/datasets_stage2' if stage == 's2' else 'data/datasets_stage1'
    REGULAR_SEASON_PATH = os.path.join(DATA_DIR, 'MRegularSeasonCompactResults.csv')
    TOURNEY_RESULTS_PATH = os.path.join(DATA_DIR, 'MNCAATourneyCompactResults.csv')
    TOURNEY_SEEDS_PATH = os.path.join(DATA_DIR, 'MNCAATourneySeeds.csv')
    TEAMS = os.path.join(DATA_DIR, 'MTeams.csv')
    SAMPLE_SUBMISSION_PATH = os.path.join(DATA_DIR, 'MSampleSubmissionStage2.csv' if stage == 's2' else 'MSampleSubmissionStage1.csv')

    rs = pd.read_csv(REGULAR_SEASON_PATH)
    tr = pd.read_csv(TOURNEY_RESULTS_PATH)
    ts = pd.read_csv(TOURNEY_SEEDS_PATH)
    ss = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    t = pd.read_csv(TEAMS)

    print(f"\nConverting {TOURNEY_RESULTS_PATH} to required Kaggle format...")
    tr1 = convert_tourney_results_to_kaggle_format(tr)

    print(f"Joining {TOURNEY_SEEDS_PATH} data...")
    tr2 = join_seed_features(tr1, ts)

    print(f"Joining {REGULAR_SEASON_PATH} data...")
    tr3 = join_reg_season_features(tr2, rs)

    print(f"Generating training, validation, and test sets...")
    all_data = tr3.copy()

    test_ids = ss['ID']
    train = all_data[~all_data['ID'].isin(test_ids)]
    validate = train.sample(int(VALIDATION_SET_SIZE * train.shape[0]), random_state = 42)
    train = train.drop(validate.index)
    test = generate_test_set(ss, ts, rs, t)
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