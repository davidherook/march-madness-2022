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
    }
]

NUMERIC_FEATURES = [f['name'] for f in FEATURES if f['type'] == FeatureType.NUMERIC]
CATEGORICAL_FEATURES = [f['name'] for f in FEATURES if f['type'] == FeatureType.CATEGORICAL]