"""
Bayesian Optimization For Tree Models
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
import xgboost as xgb
import lightgbm as lgb
import logging
from sklearn.metrics import make_scorer
from .utils import *

BASE_DIR = os.getcwd()  # current working directory


def _calc_rmse(y, y_pred):
    return np.sqrt(np.mean((y - y_pred)**2))


class MetricParser(object):
    @classmethod
    def parse(cls, metrics, task):
        if task.lower() in ['regression', 'r']:
            if metrics.upper() in ['L2', 'RMSE']:
                # L2, rmse and mse are combined here
                # Creating root mean square error for sklearns crossvalidation
                ET_RF_SCORER = make_scorer(_calc_rmse, greater_is_better=False)
                ET_RF_METRIC = 'mse'
                LGB_METRIC = 'rmse'
                XGB_METRIC = 'rmse'
            elif metrics.upper() in ['L1', 'MAE']:
                # ET_RF_SCORER = 'neg_mean_absolute_error'
                # ET_RF_METRIC = 'mae'
                # Creating root mean square error for sklearns crossvalidation
                ET_RF_SCORER = make_scorer(_calc_rmse, greater_is_better=False)
                ET_RF_METRIC = 'mse'
                logging.warn(
                    f" sklearn's RF or ET is quite slow to train when using L1 criterion! \nSo it will be used L2 istead here!"
                )
                LGB_METRIC = 'l1'
                XGB_METRIC = 'mae'
            else:
                raise ValueError(f"metrics: {metrics} is not supported now!")
        elif task.lower() in ['classification', 'c']:
            if metrics.upper() in ['AUC']:
                ET_RF_SCORER = 'auc'
                ET_RF_METRIC = 'auc'
                LGB_METRIC = 'auc'
                XGB_METRIC = 'auc'
            else:
                raise ValueError(f"metrics: {metrics} is not supported now!")
        else:
            raise ValueError(
                "TASK should be string, and its lowercase value should be in ['regression', 'r'] or 'classification', 'c']"
            )
        m = cls.__new__(cls)
        # m.metric={}
        # m.metric['RF'] = ET_RF_SCORER
        # m.metric['ET'] = ET_RF_SCORER
        # m.metric['LGB'] = LGB_METRIC
        # m.metric['XGB'] = XGB_METRIC
        m.ET_RF_SCORER = ET_RF_SCORER
        m.ET_RF_METRIC = ET_RF_METRIC
        m.LGB_METRIC = LGB_METRIC
        m.XGB_METRIC = XGB_METRIC
        return m


def update_metric(space, model_nm, metric_helper):
    if model_nm == 'LGB':
        space['metric'] = metric_helper.LGB_METRIC
    elif model_nm == 'XGB':
        space['eval_metric'] = metric_helper.XGB_METRIC
    elif model_nm in ['ET', 'RF']:
        space['criterion'] = metric_helper.ET_RF_METRIC
    else:
        raise ValueError(f"model_nm: {model_nm} is a bug...")
    return space


def sk_cv(model_nm, metric_helper, params, X_train, y_train, cv):

    model_nm = model_nm.lower()
    if model_nm == 'rf':
        sk_model = RandomForestRegressor(**params)
    elif model_nm == 'et':
        sk_model = ExtraTreesRegressor(**params)
    else:
        raise ValueError(f"model_nm: {model_nm} is not supported now! ")

    scores = cross_val_score(sk_model,
                             X_train,
                             y_train,
                             cv=cv,
                             n_jobs=-1,
                             pre_dispatch=10,
                             verbose=0,
                             scoring=metric_helper.ET_RF_SCORER)
    return scores


def objective_base(params,
                   metric_helper,
                   train_set,
                   val_set=None,
                   model_nm='LGBM',
                   folds=None,
                   nfold=5,
                   writetoFile=True):
    """
    Objective function for Gradient Boosting Machine Hyperparameter Optimization
    Args:
        val_set: default None, if not None, train_test_split will be used.
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

    if val_set is not None:
        if model_nm in ['lightgbm', 'lgbm', 'lgb']:
            model_lgb = lgb.train(params,
                                  train_set,
                                  num_boost_round=5000,
                                  valid_sets=[train_set, val_set],
                                  verbose_eval=50,
                                  early_stopping_rounds=200)

            if metric_helper.LGB_METRIC in ['auc']: # the higer, the better 
                loss = -model_lgb.best_score['valid_1'][metric_helper.LGB_METRIC]
                # Boosting rounds that returned the lowest cv rmse/mae
                n_estimators = model_lgb.best_iteration
            else:
                loss = model_lgb.best_score['valid_1'][metric_helper.LGB_METRIC]
                # use the first eval metric for default
                n_estimators = model_lgb.best_iteration
        else:
            ### Todo ###
            raise ValueError(f"{model_nm} is not supported now!")

    else:
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
                             early_stopping_rounds=200,
                             fpreproc=None,
                             verbose_eval=30,
                             show_stdv=True,
                             seed=0,
                             callbacks=None)

            if metric_helper.LGB_METRIC in ['auc']: # the higer, the better 
                loss = -np.max(cv_dict['%s-mean' % metric_helper.LGB_METRIC])
                # Boosting rounds that returned the lowest cv rmse/mae
                n_estimators = int(
                    np.argmax(cv_dict['%s-mean' % metric_helper.LGB_METRIC]) + 1)
            else:
                loss = np.min(cv_dict['%s-mean' % metric_helper.LGB_METRIC])
                # Boosting rounds that returned the lowest cv rmse/mae
                n_estimators = int(
                    np.argmin(cv_dict['%s-mean' % metric_helper.LGB_METRIC]) + 1)

        elif model_nm in ['xgboost', 'xgb']:
            cv_dict = xgb.cv(params,
                             train_set,
                             num_boost_round=5000,
                             nfold=nfold,
                             stratified=False,
                             folds=folds,
                             early_stopping_rounds=200,
                             as_pandas=False,
                             verbose_eval=30,
                             seed=0,
                             shuffle=False)

            # Extract the min rmse/mae, Loss must be minimized
            loss = np.min(cv_dict['test-%s-mean' % metric_helper.XGB_METRIC])
            # Boosting rounds that returned the lowest cv rmse/mae
            n_estimators = int(
                np.argmin(cv_dict['test-%s-mean' % metric_helper.XGB_METRIC]) +
                1)

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
                           metric_helper=metric_helper,
                           params=params,
                           X_train=X_train,
                           y_train=y_train,
                           cv=cv)
            # Extract the min rmse/mae, Loss must be minimized
            loss = -scores.mean()  # scores returns negative!

            # All scorer objects follow the convention that higher return values are better than lower return values.
            # Thus metrics which measure the distance between the model and the data, like metrics.mean_squared_error,
            # are available as neg_mean_squared_error which return the negated value of the metric.

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
            print(
                "No trial file directory <hyperopt_output> exists, will be created..."
            )
        if os.path.exists(trial_file) and _ITERATION == 1:
            print("Trial file exists, will be renamed...")
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


def post_hyperopt(bayes_trials,
                  metric_helper,
                  train_set,
                  val_set,
                  model_nm=None,
                  folds=None,
                  nfold=5):

    model_nm = model_nm.lower()
    # get best params
    bayes_results = pd.DataFrame(bayes_trials.results)
    bayes_results = bayes_results.sort_values(by='loss')
    bayes_results.reset_index(drop=True, inplace=True)
    best_params = bayes_results.loc[0, 'params']
    # best loss with big learning rate
    _best_loss = bayes_results.loc[0, 'loss']
    # best n_estimators with big learning rate
    _best_estimators = bayes_results.loc[0, 'estimators']
    if val_set is not None:
        if model_nm in ['lgb', 'lgbm', 'lightgbm']:
            # [get best loss and trees]
            best_params['learning_rate'] = 0.01

            model_lgb = lgb.train(best_params,
                                  train_set,
                                  num_boost_round=5000,
                                  valid_sets=[train_set, val_set],
                                  verbose_eval=50,
                                  early_stopping_rounds=200)
            loss = model_lgb.best_score['valid_1'][metric_helper.LGB_METRIC]
            # use the first eval metric for default
            n_estimators = model_lgb.best_iteration
            best_params['n_estimators'] = n_estimators
        else:
            ### Todo ###
            raise ValueError(f"{model_nm} is not supported now!")
    else:
        # print(f"best_loss_:{best_loss_}")
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
                             verbose_eval=30,
                             seed=2019)
            # Extract the min rmse/mae, Loss must be minimized
            loss = np.min(cv_dict['test-%s-mean' % metric_helper.XGB_METRIC])
            # Boosting rounds that returned the lowest cv rmse/mae
            n_estimators = int(
                np.argmin(cv_dict['test-%s-mean' % metric_helper.XGB_METRIC]) +
                1)
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
                             verbose_eval=30,
                             seed=2019)

            if metric_helper.LGB_METRIC in ['auc']: # the higer, the better 
                loss = -np.max(cv_dict['%s-mean' % metric_helper.LGB_METRIC])
                # Boosting rounds that returned the lowest cv rmse/mae
                n_estimators = int(
                    np.argmax(cv_dict['%s-mean' % metric_helper.LGB_METRIC]) + 1)
            else:
                loss = np.min(cv_dict['%s-mean' % metric_helper.LGB_METRIC])
                # Boosting rounds that returned the lowest cv rmse/mae
                n_estimators = int(
                    np.argmin(cv_dict['%s-mean' % metric_helper.LGB_METRIC]) + 1)
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
                           metric_helper=metric_helper,
                           params=best_params,
                           X_train=X_train,
                           y_train=y_train,
                           cv=cv)
            loss = -scores.mean()
        else:
            raise ValueError(f"model_nm: {model_nm} is a bug...")

    logging.info(f"best loss: {loss}, best n_estimators: {n_estimators}")
    logging.info(f"best params: {best_params}")

    if loss >= _best_loss:  # score of LR_0.01 is worse than that of LR_0.05
        if model_nm not in ['rf', 'et']:  # for GBDT
            logging.info(
                "===================================================================================="
            )
            logging.warn(
                "model scores of LR_0.01 is worse than that of LR_0.05, LR_0.05 will be used instead!"
            )
            logging.warn(
                f"before(LR_0.05): {_best_loss}, best n_estimators: {_best_estimators} | after(LR_0.01): {loss}, best n_estimators: {n_estimators}"
            )
            n_estimators = _best_estimators
            loss = _best_loss
            best_params['n_estimators'] = n_estimators
            best_params['learning_rate'] = 0.05
            logging.info(
                "===================================================================================="
            )
            logging.info("After adjustment...")
            logging.info(
                f"best loss: {loss}, best n_estimators: {n_estimators}")
            logging.info(f"best params: {best_params}")
        else:
            logging.info(
                "===================================================================================="
            )
            logging.warn(
                "model scores of 1000 Trees is worse than that of 200 Trees, 200 Trees will be used instead!"
            )
            logging.warn(
                f"before(200): {_best_loss}, best n_estimators: {_best_estimators} | after(1000): {loss}, best n_estimators: {n_estimators}"
            )
            n_estimators = _best_estimators
            loss = _best_loss
            best_params['n_estimators'] = n_estimators
            logging.info(
                "===================================================================================="
            )
            logging.info("After adjustment...")
            logging.info(
                f"best loss: {loss}, best n_estimators: {n_estimators}")
            logging.info(f"best params: {best_params}")

    return best_params, loss


def main_tuning_with_bo(X_train,
                        y_train,
                        X_val=None,
                        y_val=None,
                        model_nm='LGBM',
                        max_evals=100,
                        folds=5,
                        nfold=None,
                        eval_metric='l2',
                        task='regression'):
    """
    Args:
        if X_val and y_val are assigned, cv will not be used for validation.
    """

    SPACE_DICT = load_space(task)
    metric_helper = MetricParser().parse(eval_metric,task)
    # Keep track of results
    bayes_trials = Trials()
    # Global variable
    global _ITERATION
    _ITERATION = 0
    model_nm = parse_model_nm(model_nm)
    TRAIN_SET = build_train_set(X_train, y_train, model_nm=model_nm)
    if (X_val is not None) and (y_val is not None):
        VAL_SET = build_train_set(X_val, y_val, model_nm=model_nm)
    else:
        VAL_SET = None
    # SPACE_DICT ={'RF':rf_space, 'ET':et_space, 'XGB':xgb_space, 'LGB':lgb_space}
    SPACE = SPACE_DICT[model_nm]
    # update metric key in SPACE
    SPACE = update_metric(space=SPACE,
                          model_nm=model_nm,
                          metric_helper=metric_helper)
    # print(SPACE)
    # (params, train_set, model_nm='LGBM', folds=None, nfold=5, writetoFile=True)
    func_objective = partial(objective_base,
                             metric_helper=metric_helper,
                             train_set=TRAIN_SET,
                             val_set=VAL_SET,
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
                rstate=np.random.RandomState(666))

    best_params, loss = post_hyperopt(bayes_trials,
                                      metric_helper=metric_helper,
                                      train_set=TRAIN_SET,
                                      val_set=VAL_SET,
                                      model_nm=model_nm,
                                      folds=folds,
                                      nfold=nfold)

    return best_params, loss
