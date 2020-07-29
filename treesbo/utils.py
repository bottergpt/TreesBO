import xgboost as xgb
import lightgbm as lgb
import pandas as pd


def load_space(task):
    if task.lower() in ['regression', 'r']:
        from .space_config.regression import SPACE_DICT
    elif task.lower() in ['classification', 'c']:
        from .space_config.classification import SPACE_DICT
    else:
        raise ValueError(
            "TASK should be string, and its lowercase value should be in ['regression', 'r'] or 'classification', 'c']"
        )
    return SPACE_DICT


def parse_model_nm(model_nm):
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
    return model_nm


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
        raise NameError(f"{model_nm} is a bug...")
    return train_set