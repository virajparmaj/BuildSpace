import os

# ===========================================
# Directory Configurations
# ===========================================

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw_data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed_data')
NORMALIZED_DATA_DIR = os.path.join(DATA_DIR, 'normalized_datasets')

# Path to the main dataset
BINANCE_BTC_PERP_CSV = os.path.join(NORMALIZED_DATA_DIR, 'binance_btc_perp.csv')

# Models directory
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SAVED_MODELS_DIR = os.path.join(MODELS_DIR, 'saved_models')

# Model file paths
MODEL1_RF_PATH = os.path.join(SAVED_MODELS_DIR, 'model1_RF.pkl')
SCALER1_RF_PATH = os.path.join(SAVED_MODELS_DIR, 'scaler1_RF.pkl')
MODEL1_LR_PATH = os.path.join(SAVED_MODELS_DIR, 'model1_LR.pkl')
SCALER1_LR_PATH = os.path.join(SAVED_MODELS_DIR, 'scaler1_LR.pkl')
MODEL2_GARCH_PATH = os.path.join(SAVED_MODELS_DIR, 'model2_GARCH.pkl')
MODEL3_RFR_PATH = os.path.join(SAVED_MODELS_DIR, 'model3_RFR.pkl')
MODEL3_SARIMAX_PATH = os.path.join(SAVED_MODELS_DIR, 'model3_SARIMAX.pkl')

# Instrument specifications
TARDIS_TYPE = 'derivative_ticker'
INSTRUMENT = 'btcusdt'
START_DATE = '2020-01-01'
END_DATE = '2024-10-18'
EXCHANGE = 'binance-futures'

# Results directory
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
PREDICTIONS_RFR_CSV = os.path.join(RESULTS_DIR, 'predictions_RFR.csv')
PREDICTIONS_SARIMAX_CSV = os.path.join(RESULTS_DIR, 'predictions_SARIMAX.csv')

# ===========================================
# General Configurations
# ===========================================

# Random seed for reproducibility
RANDOM_STATE = 42