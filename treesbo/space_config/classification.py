from hyperopt import hp

# Define the search space [LGBM]
lgb_space = {
    'seed': 666,
    'objective':'binary',
    'metric': 'auc',
    'n_jobs': -1,
    'learning_rate': 0.1, # 'learning_rate': hp.choice('learning_rate', [0.05, 0.01]),
    'boosting_type': 'gbdt',
    'bagging_fraction': hp.uniform('bagging_fraction', 0.1, 1.0),
    'feature_fraction': hp.uniform('feature_fraction', 0.1, 1.0),
    'bagging_freq': hp.quniform('bagging_freq', 1, 6, 1),
    'num_leaves': hp.quniform('num_leaves', 20, 300, 10),
    'max_depth': hp.quniform('max_depth', 5, 30, 1),
    'min_data_in_leaf': hp.quniform('min_data_in_leaf', 10, 50, 10),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 10.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 10.0)
}

# 'min_sum_hessian_in_leaf': hp.uniform('min_sum_hessian_in_leaf', 0.0, 10.0),
# 'min_split_gain': hp.uniform('min_split_gain', 0.0, 5.0),

SPACE_DICT = {
    'LGB': lgb_space
}
