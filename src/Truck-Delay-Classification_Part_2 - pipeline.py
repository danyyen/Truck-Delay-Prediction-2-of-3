# ======================================================
# Truck Delay Classification – Production Training Script
# ======================================================

import os
import sys
import logging
import warnings
warnings.filterwarnings("ignore")

# -----------------------
# Logging
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# -----------------------
# Environment Variables
# -----------------------
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

if not HOPSWORKS_API_KEY:
    logger.error("Missing HOPSWORKS_API_KEY environment variable")
    sys.exit(1)

if not WANDB_API_KEY:
    logger.error("Missing WANDB_API_KEY environment variable")
    sys.exit(1)

os.environ["WANDB_API_KEY"] = WANDB_API_KEY

# -----------------------
# Imports (unchanged)
# -----------------------
import pandas as pd
import hopsworks
import joblib
import wandb
import xgboost as xgb

from pickle import dump
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, recall_score, confusion_matrix, roc_auc_score
)

# -----------------------
# Login to Hopsworks
# -----------------------
logger.info("Logging into Hopsworks...")
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

# -----------------------
# Read Feature Groups
# -----------------------
final_data = fs.get_feature_group("final_data", version=1)
final_merge = final_data.select_all().read(read_options={"use_hive": True})

routes_data = fs.get_feature_group("routes_details_fg", version=1)
routes_df = routes_data.select_all().read(read_options={"use_hive": True})

weather_data = fs.get_feature_group("city_weather_details_fg", version=1)
weather_df = weather_data.select_all().read(read_options={"use_hive": True})

# -----------------------
# Drop Null Weather Rows
# -----------------------
final_merge = final_merge.dropna(
    subset=[
        "origin_temp",
        "origin_wind_speed",
        "origin_precip",
        "origin_humidity",
        "origin_visibility",
        "origin_pressure",
    ]
).reset_index(drop=True)

# -----------------------
# Feature Columns
# -----------------------
cts_cols = [
    "route_avg_temp", "route_avg_wind_speed", "route_avg_precip",
    "route_avg_humidity", "route_avg_visibility", "route_avg_pressure",
    "distance", "average_hours",
    "origin_temp", "origin_wind_speed", "origin_precip",
    "origin_humidity", "origin_visibility", "origin_pressure",
    "destination_temp", "destination_wind_speed", "destination_precip",
    "destination_humidity", "destination_visibility", "destination_pressure",
    "avg_no_of_vehicles", "truck_age", "load_capacity_pounds",
    "mileage_mpg", "age", "experience", "average_speed_mph"
]

cat_cols = [
    "route_description",
    "origin_description",
    "destination_description",
    "accident",
    "fuel_type",
    "gender",
    "driving_style",
    "ratings",
    "is_midnight",
]

# -----------------------
# Time-Based Split
# -----------------------
train_df = final_merge[final_merge["estimated_arrival"] <= "2019-01-30"]
validation_df = final_merge[
    (final_merge["estimated_arrival"] > "2019-01-30")
    & (final_merge["estimated_arrival"] <= "2019-02-07")
]
test_df = final_merge[final_merge["estimated_arrival"] > "2019-02-07"]

X_train = train_df[cts_cols + cat_cols]
y_train = train_df["delay"]

X_valid = validation_df[cts_cols + cat_cols]
y_valid = validation_df["delay"]

X_test = test_df[cts_cols + cat_cols]
y_test = test_df["delay"]

# -----------------------
# Fill Missing Load Capacity
# -----------------------
load_capacity_mode = X_train["load_capacity_pounds"].mode().iloc[0]

X_train["load_capacity_pounds"] = X_train["load_capacity_pounds"].fillna(load_capacity_mode)
X_valid["load_capacity_pounds"] = X_valid["load_capacity_pounds"].fillna(load_capacity_mode)
X_test["load_capacity_pounds"] = X_test["load_capacity_pounds"].fillna(load_capacity_mode)

# -----------------------
# One-Hot Encoding
# -----------------------
encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

encode_columns = [
    "route_description",
    "origin_description",
    "destination_description",
    "fuel_type",
    "gender",
    "driving_style",
]

encoder.fit(X_train[encode_columns])

encoded_features = list(encoder.get_feature_names_out(encode_columns))

X_train[encoded_features] = encoder.transform(X_train[encode_columns])
X_valid[encoded_features] = encoder.transform(X_valid[encode_columns])
X_test[encoded_features] = encoder.transform(X_test[encode_columns])

dump(encoder, open("truck_data_encoder.pkl", "wb"))

X_train.drop(encode_columns, axis=1, inplace=True)
X_valid.drop(encode_columns, axis=1, inplace=True)
X_test.drop(encode_columns, axis=1, inplace=True)

# -----------------------
# Scaling
# -----------------------
scaler = StandardScaler()

X_train[cts_cols] = scaler.fit_transform(X_train[cts_cols])
X_valid[cts_cols] = scaler.transform(X_valid[cts_cols])
X_test[cts_cols] = scaler.transform(X_test[cts_cols])

dump(scaler, open("truck_data_scaler.pkl", "wb"))

# -----------------------
# Weights
# -----------------------
weights = (
    len(X_train) / (2 * y_train.value_counts()[0]),
    len(X_train) / (2 * y_train.value_counts()[1]),
)
w = {0: weights[0], 1: weights[1]}

# -----------------------
# W&B Setup
# -----------------------
USER_NAME = "enter_your_username"
PROJECT_NAME = "enter_your_project_name"

wandb.login()

# -----------------------
# Evaluation Function (UNCHANGED)
# -----------------------
comparison_columns = [
    "Model_Name",
    "Train_F1score",
    "Train_Recall",
    "Valid_F1score",
    "Valid_Recall",
    "Test_F1score",
    "Test_Recall",
]

comparison_df = pd.DataFrame()
final_list = []

def evaluate_models(model_name, model_defined_var, X_train, y_train, X_valid, y_valid, X_test, y_test):
    y_train_pred = model_defined_var.predict(X_train)
    y_valid_pred = model_defined_var.predict(X_valid)
    y_test_pred = model_defined_var.predict(X_test)

    train_f1 = f1_score(y_train, y_train_pred)
    valid_f1 = f1_score(y_valid, y_valid_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    train_recall = recall_score(y_train, y_train_pred)
    valid_recall = recall_score(y_valid, y_valid_pred)
    test_recall = recall_score(y_test, y_test_pred)

    print(f"\n{model_name} Results")
    print("Train:", train_f1, train_recall)
    print("Valid:", valid_f1, valid_recall)
    print("Test:", test_f1, test_recall)

    return dict(
        zip(
            comparison_columns,
            [model_name, train_f1, train_recall, valid_f1, valid_recall, test_f1, test_recall],
        )
    )

def add_dic_to_final_df(d):
    final_list.append(d)
    global comparison_df
    comparison_df = pd.DataFrame(final_list, columns=comparison_columns)

# -----------------------
# Logistic Regression
# -----------------------
log_reg = LogisticRegression(random_state=13, class_weight=w)
log_reg.fit(X_train, y_train)

add_dic_to_final_df(
    evaluate_models("Logistic Regression", log_reg, X_train, y_train, X_valid, y_valid, X_test, y_test)
)

joblib.dump(log_reg, "log-truck-model.pkl")

# -----------------------
# Random Forest
# -----------------------
rf = RandomForestClassifier(n_estimators=20, class_weight=w, random_state=7)
rf.fit(X_train, y_train)

add_dic_to_final_df(
    evaluate_models("Random Forest", rf, X_train, y_train, X_valid, y_valid, X_test, y_test)
)

joblib.dump(rf, "randomf-truck-model.pkl")

# -----------------------
# XGBoost
# -----------------------
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {"objective": "multi:softmax", "num_class": 2, "seed": 7}
xgbmodel = xgb.train(params, dtrain, 30, evals=[(dvalid, "validation")], early_stopping_rounds=10)

joblib.dump(xgbmodel, "xgb-truck-model.pkl")

# -----------------------
# Final Output
# -----------------------
print("\nModel Comparison:")
print(comparison_df)

logger.info("Pipeline completed successfully")
sys.exit(0)
