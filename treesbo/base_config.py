import os 

TASK = 'regression' # 'regression','r', 'classification', 'c'
MODEL_NM = 'LGBM'
FOLDS = None
NFOLD = 5
MAX_EVALS = 100
BASE_DIR = os.getcwd()

# try:
#     BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# except:
#     BASE_DIR = os.getcwd()


