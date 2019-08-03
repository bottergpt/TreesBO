from hyperopt import hp

## config example ##
# =================================================================================== #
## Define the search space RAW SPACE (for reference)

# space = {
#     'metric': hp.choice('metric', ['l2_root']),
#     'n_jobs': hp.choice('n_jobs',[-1]),
#     'learning_rate': hp.choice('learning_rate',[0.05]),
# #     'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
#     'boosting_type': hp.choice('boosting_type', [{'boosting_type': 'gbdt', 'subsample': hp.uniform('subsample', 0.5, 1)},
#                                                  {'boosting_type': 'dart', 'subsample': hp.uniform('subsample', 0.5, 1)},
#                                                  {'boosting_type': 'goss', 'subsample': 1.0}]),
#     'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
#     'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
#     'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
#     'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
#     'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
#     'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
# }
#     'reg_alpha': hp.loguniform('reg_alpha', np.log(0.01), np.log(5)),
#     'reg_lambda': hp.loguniform('reg_lambda', np.log(0.01), np.log(5))
#     'reg_alpha': hp.uniform('reg_alpha', 0.0, 5.0),
#     'reg_lambda': hp.uniform('reg_lambda', 0.0, 5.0)
# =================================================================================== #

# Define the search space [LGBM]
lgb_space = {
    'seed': 2019,
    'metric': 'l2_root',
    'n_jobs': -1,
    'learning_rate': 0.05, # 'learning_rate': hp.choice('learning_rate', [0.05, 0.01]),
    'boosting_type': 'gbdt',
    'bagging_fraction': hp.uniform('bagging_fraction', 0.1, 1.0),
    'feature_fraction': hp.uniform('feature_fraction', 0.1, 1.0),
    'bagging_freq': hp.quniform('bagging_freq', 2, 20, 2),
    'num_leaves': hp.quniform('num_leaves', 50, 450, 1),
    'max_depth': hp.quniform('max_depth', 6, 30, 1),
    'min_data_in_leaf': hp.quniform('min_data_in_leaf', 20, 1000, 5),
    'min_split_gain': hp.uniform('min_split_gain', 0.0, 5.0),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 10.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 10.0)
}
# 'min_sum_hessian_in_leaf': hp.uniform('min_sum_hessian_in_leaf', 0.0, 10.0),


# Define the search space [XGB]
xgb_space = {
    'booster': 'gbtree',
    'random_state': 2019,
    'eval_metric': 'rmse',
    'n_jobs': -1,
    'learning_rate': 0.05,
    'subsample': hp.uniform('subsample', 0.1, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1.0),
    'max_depth': hp.quniform('max_depth', 5, 30, 1),
    'gamma': hp.uniform('gamma', 0.0, 2.0),
    'min_child_weight': hp.uniform('min_child_weight', 0.0, 5.0),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 3.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 3.0)
}

# Define the search space [RF]
rf_space = {
    'criterion': 'mse',
    'random_state': 2019,
    'n_jobs': -1,
    'n_estimators': 200,
    'max_features': hp.uniform('max_features', 0.1, 1.0),
    'max_depth': hp.quniform('max_depth', 5, 30, 1),
    'min_samples_split': hp.quniform('min_samples_split', 20, 500, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 2, 100, 1),
    'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0.0, 5.0)
}

# Define the search space [ET]
et_space = {
    'criterion': 'mse',
    'random_state': 2019,
    'n_jobs': -1,
    'n_estimators': 200,
    'max_features': hp.uniform('max_features', 0.1, 1.0),
    'max_depth': hp.quniform('max_depth', 5, 30, 1),
    'min_samples_split': hp.quniform('min_samples_split', 20, 500, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 2, 100, 1),
    'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0.0, 5.0)
}

SPACE_DICT = {
    'RF': rf_space,
    'ET': et_space,
    'XGB': xgb_space,
    'LGB': lgb_space
}
