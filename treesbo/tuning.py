"""
Bayesian Optimization
"""
import os
import csv
from hyperopt import STATUS_OK, fmin, tpe, Trials, hp
from timeit import default_timer as timer
import time
import numpy as np
import pandas as pd
from functools import partial
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import make_scorer
import xgboost as xgb
import lightgbm as lgb
import logging

def sk_cv(model_nm, params, X_train, y_train, cv):

    model_nm = model_nm.lower()
    if model_nm == 'rf':
        sk_model = RandomForestRegressor(**params)
    elif model_nm == 'et':
        sk_model = ExtraTreesRegressor(**params)
    else:
        raise ValueError(f"model_nm: {model_nm} is not supported now! ")

    def calc_rmse(y, y_pred):
        return np.sqrt(np.mean((y - y_pred)**2))

    # Creating root mean square error for sklearns crossvalidation
    rmse_scorer = make_scorer(calc_rmse, greater_is_better=False)
    scores = cross_val_score(sk_model,
                             X_train,
                             y_train,
                             cv=cv,
                             n_jobs=-1,
                             pre_dispatch=10,
                             verbose=0,
                             scoring=rmse_scorer)
    return scores


def objective_base(params,
                   train_set,
                   model_nm='LGBM',
                   folds=None,
                   nfold=5,
                   writetoFile=True):
    """
    Objective function for Gradient Boosting Machine Hyperparameter Optimization
    Args:
        folds: This argument has highest priority over other data split arguments.
    
    Return:
    
    """
    model_nm = model_nm.lower()
    # Keep track of evals
    global _ITERATION
    _ITERATION += 1

    # Make sure parameters that need to be integers are integers
    for parameter_name in [
            'num_leaves', 'max_depth', 'bagging_freq', 'min_data_in_leaf',
            'min_samples_split', 'min_samples_leaf'
    ]:
        if parameter_name in params:
            params[parameter_name] = int(params[parameter_name])

    start = timer()

    logging.info(f"{_ITERATION} ITERATION")
    logging.info(f"params:\n{params}")
    logging.info("############################")
    logging.info(f"{model_nm} CV Running...")
    logging.info("############################")

    if model_nm in ['lightgbm', 'lgbm', 'lgb']:
        # Perform n_folds cross validation
        cv_dict = lgb.cv(params,
                         train_set,
                         num_boost_round=5000,
                         folds=folds,
                         nfold=nfold,
                         stratified=False,
                         shuffle=False,
                         metrics=None,
                         fobj=None,
                         feval=None,
                         init_model=None,
                         feature_name='auto',
                         categorical_feature='auto',
                         early_stopping_rounds=100,
                         fpreproc=None,
                         verbose_eval=10,
                         show_stdv=True,
                         seed=0,
                         callbacks=None)
        # Extract the min rmse, Loss must be minimized
        loss = np.min(cv_dict['rmse-mean'])
        # Boosting rounds that returned the lowest cv rmse
        n_estimators = int(np.argmin(cv_dict['rmse-mean']) + 1)
    elif model_nm in ['xgboost', 'xgb']:
        cv_dict = xgb.cv(params,
                         train_set,
                         num_boost_round=5000,
                         nfold=nfold,
                         stratified=False,
                         folds=folds,
                         early_stopping_rounds=100,
                         as_pandas=False,
                         verbose_eval=10,
                         seed=0,
                         shuffle=False)

        # Extract the min rmse, Loss must be minimized
        loss = np.min(cv_dict['test-rmse-mean'])
        # Boosting rounds that returned the lowest cv rmse
        n_estimators = int(np.argmin(cv_dict['test-rmse-mean']) + 1)

    elif model_nm in ['rf', 'et']:
        try:
            X_train, y_train = train_set
        except Exception as e:
            print("Error in assignment..", e)
            print(
                "train_set should be a tuple or list containing <X_train, y_train>"
            )
            raise
        if nfold is None:
            cv = folds
        else:
            cv = nfold
        scores = sk_cv(model_nm=model_nm,
                       params=params,
                       X_train=X_train,
                       y_train=y_train,
                       cv=cv)
        # Extract the min rmse, Loss must be minimized
        loss = -scores.mean()
        # Boosting rounds that returned the lowest cv rmse
        n_estimators = params['n_estimators']
    else:
        raise ValueError("No such model...")
    run_time = timer() - start
    # Write to the csv file ('a' means append)
    if writetoFile:
        random_datetime = str(int(time.time()))
        hyper_base_path = os.path.join(BASE_DIR, 'hyperopt_output')
        trial_file = os.path.join(hyper_base_path, 'trials.csv')
        trial_file_rename = os.path.join(hyper_base_path,
                                         'trials_%s.csv' % random_datetime)
        if not os.path.exists(hyper_base_path):
            os.makedirs(hyper_base_path)
            logging.info(" No trial file directory <hyperopt_output> exists, will be created...")
        if os.path.exists(trial_file) and _ITERATION == 1:
            logging.info(" Trial file exists, will be renamed...")
            os.rename(trial_file, trial_file_rename)
            assert os.path.exists(
                trial_file
            ) == False, "Trial file still exists, rename failed..."
            # File to save first results
            of_connection = open(trial_file, 'w')
            writer = csv.writer(of_connection)
            # Write the headers to the file
            writer.writerow(
                ['loss', 'params', 'iteration', 'estimators', 'train_time'])
            of_connection.close()
        of_connection = open(trial_file, 'a')
        writer = csv.writer(of_connection)
        writer.writerow([loss, params, _ITERATION, n_estimators, run_time])
    # Dictionary with information for evaluation
    return {
        'loss': loss,
        'params': params,
        'iteration': _ITERATION,
        'estimators': n_estimators,
        'train_time': run_time,
        'status': STATUS_OK
    }


def build_train_set(X_train, y_train, model_nm):

    isX_df = isinstance(X_train, pd.DataFrame)
    isY_sr = isinstance(y_train, pd.Series)
    isY_df = isinstance(y_train, pd.DataFrame)
    if isY_df:
        raise TypeError(
            f"y_train is df, with the shape {y_train.shape}, which is not supportable now."
        )
    model_nm = model_nm.lower()
    if isX_df ^ isY_sr:
        raise TypeError(f"X_train and y_train have different types!")
    if model_nm in ['xgb', 'xgboost']:
        if isX_df:
            train_set = xgb.DMatrix(X_train.values, y_train.values)
        else:
            train_set = xgb.DMatrix(X_train, y_train)
    elif model_nm in ['lgb', 'lgbm', 'lightgbm']:
        train_set = lgb.Dataset(X_train, y_train)
    elif model_nm in ['et', 'rf']:
        train_set = (X_train, y_train)
    else:
        raise ValueError(f"{model_nm} is a bug...")
    return train_set


def post_hyperopt(bayes_trials, train_set, model_nm, folds=None, nfold=5):

    model_nm = model_nm.lower()
    # get best params
    bayes_results = pd.DataFrame(bayes_trials.results)
    bayes_results = bayes_results.sort_values(by='loss')
    bayes_results.reset_index(drop=True, inplace=True)
    best_params = bayes_results.loc[0, 'params']
    _best_loss = bayes_results.loc[0, 'loss'] # best loss with big learning rate
    _best_estimators = bayes_results.loc[0, 'estimators'] # best n_estimators with big learning rate
    
#     print(f"best_loss_:{best_loss_}")
    if model_nm in ['xgboost', 'xgb']:
        # get best loss and trees
        best_params['learning_rate'] = 0.01
        # Perform n_folds cross validation
        cv_dict = xgb.cv(best_params,
                         train_set,
                         num_boost_round=5000,
                         folds=folds,
                         nfold=nfold,
                         stratified=False,
                         shuffle=False,
                         early_stopping_rounds=100,
                         as_pandas=False,
                         verbose_eval=10,
                         seed=2019)
        # Extract the min rmse, Loss must be minimized
        loss = np.min(cv_dict['test-rmse-mean'])
        # Boosting rounds that returned the lowest cv rmse
        n_estimators = int(np.argmin(cv_dict['test-rmse-mean']) + 1)
        best_params['n_estimators'] = n_estimators

    elif model_nm in ['lgb', 'lgbm', 'lightgbm']:
        # [get best loss and trees]
        best_params['learning_rate'] = 0.01
        # [Perform n_folds cross validation]
        cv_dict = lgb.cv(best_params,
                         train_set,
                         num_boost_round=5000,
                         folds=folds,
                         nfold=nfold,
                         stratified=False,
                         shuffle=False,
                         feature_name='auto',
                         categorical_feature='auto',
                         early_stopping_rounds=100,
                         verbose_eval=10,
                         seed=2019)
        # Extract the min rmse, Loss must be minimized
        loss = np.min(cv_dict['rmse-mean'])
        # Boosting rounds that returned the lowest cv rmse
        n_estimators = int(np.argmin(cv_dict['rmse-mean']) + 1)
        best_params['n_estimators'] = n_estimators

    elif model_nm in ['rf', 'et']:
        if nfold is None:
            cv = folds
        else:
            cv = nfold
        n_estimators = 1000
        best_params['n_estimators'] = n_estimators
        X_train, y_train = train_set
        scores = sk_cv(model_nm=model_nm,
                       params=best_params,
                       X_train=X_train,
                       y_train=y_train,
                       cv=cv)
        loss = -scores.mean()
    else:
        raise ValueError(f"model_nm: {model_nm} is a bug...")
        
    logging.info(f"best loss: {loss}, best n_estimators: {n_estimators}")
    logging.info(f"best params: {best_params}")
    
    if loss>=_best_loss: # score of LR_0.01 is worse than that of LR_0.05
        if model_nm not in ['rf', 'et']: # for GBDT
            logging.info("====================================================================================")
            logging.warning("model scores of LR_0.01 is worse than that of LR_0.05, LR_0.05 will be used instead!")
            logging.warning(f"before(LR_0.05): {_best_loss}, best n_estimators: {_best_estimators} | after(LR_0.01): {loss}, best n_estimators: {n_estimators}")
            n_estimators = _best_estimators
            loss = _best_loss
            best_params['n_estimators'] = n_estimators
            best_params['learning_rate'] = 0.05
            logging.info("====================================================================================")
            logging.info("After adjustment...")
            logging.info(f"best loss: {loss}, best n_estimators: {n_estimators}")
            logging.info(f"best params: {best_params}")
        else:
            logging.info("====================================================================================")
            logging.warning("model scores of 1000 Trees is worse than that of 200 Trees, 200 Trees will be used instead!")
            logging.warning(f"before(200): {_best_loss}, best n_estimators: {_best_estimators} | after(1000): {loss}, best n_estimators: {n_estimators}")
            n_estimators = _best_estimators
            loss = _best_loss
            best_params['n_estimators'] = n_estimators
            logging.info("====================================================================================")
            logging.info("After adjustment...")
            logging.info(f"best loss: {loss}, best n_estimators: {n_estimators}")
            logging.info(f"best params: {best_params}")
            
    return best_params, loss


def main_tuning_with_bo(X_train,
                        y_train,
                        model_nm=MODEL_NM,
                        max_evals=MAX_EVALS,
                        folds=FOLDS,
                        nfold=NFOLD):

    # Keep track of results
    bayes_trials = Trials()
    # Global variable
    global _ITERATION
    _ITERATION = 0

    if model_nm.lower() in ['lgbm', 'lightgbm', 'lgb']:
        model_nm = 'LGB'
    elif model_nm.lower() in ['xgb', 'xgboost']:
        model_nm = 'XGB'
    elif model_nm.lower() in ['rf', 'randomforest']:
        model_nm = 'RF'
    elif model_nm.lower() in ['et', 'extratrees']:
        model_nm = 'ET'
    else:
        raise ValueError(f"model_nm: {model_nm} is a bug...")

    TRAIN_SET = build_train_set(X_train, y_train, model_nm=model_nm)
    #     SPACE_DICT ={'RF':rf_space, 'ET':et_space, 'XGB':xgb_space, 'LGB':lgb_space}
    SPACE = SPACE_DICT[model_nm]

    # (params, train_set, model_nm='LGBM', folds=None, nfold=5, writetoFile=True)
    func_objective = partial(objective_base,
                             train_set=TRAIN_SET,
                             model_nm=model_nm,
                             folds=folds,
                             nfold=nfold,
                             writetoFile=True)
    # Run optimization
    best = fmin(fn=func_objective,
                space=SPACE,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=bayes_trials,
                rstate=np.random.RandomState(2019))

    best_params, loss = post_hyperopt(bayes_trials,
                                      train_set=TRAIN_SET,
                                      model_nm=model_nm,
                                      folds=folds,
                                      nfold=nfold)

    return best_params, loss
