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
