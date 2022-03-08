class FeatureType:
    NUMERIC = 'numeric'
    CATEGORICAL = 'categorical'

FEATURES = [
    {
    'name': 'team_a_seed_num',
    'type': FeatureType.CATEGORICAL
    },
    {
    'name': 'team_b_seed_num',
    'type': FeatureType.CATEGORICAL
    },
    {
    'name': 'score_mean_a',
    'type': FeatureType.NUMERIC
    },
    {
    'name': 'score_mean_b',
    'type': FeatureType.NUMERIC
    },
    {
    'name': 'score_std_a',
    'type': FeatureType.NUMERIC
    },
    {
    'name': 'score_std_b',
    'type': FeatureType.NUMERIC
    },
    {
    'name': 'score_min_a',
    'type': FeatureType.NUMERIC
    },
    {
    'name': 'score_min_b',
    'type': FeatureType.NUMERIC
    },
    {
    'name': 'score_max_a',
    'type': FeatureType.NUMERIC
    },
    {
    'name': 'score_max_b',
    'type': FeatureType.NUMERIC
    },
    {
    'name': 'diff_mean_a',
    'type': FeatureType.NUMERIC
    },
    {
    'name': 'diff_mean_b',
    'type': FeatureType.NUMERIC
    },
    {
    'name': 'diff_std_a',
    'type': FeatureType.NUMERIC
    },
    {
    'name': 'diff_std_b',
    'type': FeatureType.NUMERIC
    },
    {
    'name': 'diff_min_a',
    'type': FeatureType.NUMERIC
    },
    {
    'name': 'diff_min_b',
    'type': FeatureType.NUMERIC
    },
    {
    'name': 'diff_max_a',
    'type': FeatureType.NUMERIC
    },
    {
    'name': 'diff_max_b',
    'type': FeatureType.NUMERIC
    },
    {
    'name': 'is_home_mean_a',
    'type': FeatureType.CATEGORICAL
    },
    {
    'name': 'is_home_mean_b',
    'type': FeatureType.CATEGORICAL
    }
]

NUMERIC_FEATURES = [f['name'] for f in FEATURES if f['type'] == FeatureType.NUMERIC]
CATEGORICAL_FEATURES = [f['name'] for f in FEATURES if f['type'] == FeatureType.CATEGORICAL]