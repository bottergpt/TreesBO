MODEL_NM = 'LGBM'
FOLDS = None
NFOLD = 5
MAX_EVALS = 100
try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except:
    BASE_DIR ='..'