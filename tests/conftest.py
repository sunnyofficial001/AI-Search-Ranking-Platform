# Workaround for OpenMP library conflict on Windows
# Importing lightgbm before xgboost/torch prevents OSError access violations during predictions.
import lightgbm
