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