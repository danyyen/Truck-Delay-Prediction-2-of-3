# ======================================================
# Random Forest Sweep Pipeline (W&B)
# ======================================================
# PURPOSE:
# - Run hyperparameter sweeps for RandomForest
# - NOT Airflow-safe
# - NOT cron-safe
# - Intended for manual / CI execution
# ======================================================

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import wandb
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# ------------------------------------------------------
# REQUIRED ENV VARS
# ------------------------------------------------------
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
PROJECT_NAME = os.getenv("WANDB_PROJECT_NAME")

if not WANDB_API_KEY:
    raise RuntimeError("Missing WANDB_API_KEY")

if not PROJECT_NAME:
    raise RuntimeError("Missing WANDB_PROJECT_NAME")

os.environ["WANDB_API_KEY"] = WANDB_API_KEY

wandb.login()

# ------------------------------------------------------
# EXPECTED GLOBAL OBJECTS
# (Injected via import or runner script)
# ------------------------------------------------------
# These MUST already exist in memory when this script runs:
#
# X_train, y_train
# X_valid, y_valid
# X_test,  y_test
# w  -> class weights
#
# This script assumes data prep has already happened.
# ------------------------------------------------------

# ======================================================
# TRAIN FUNCTION (UNCHANGED BEHAVIOR)
# ======================================================
def train_rf_model():
    with wandb.init(project=PROJECT_NAME) as run:
        config = wandb.config

        model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            random_state=7,
            class_weight=w
        )

        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_valid_pred = model.predict(X_valid)
        y_test_pred = model.predict(X_test)

        # Metrics
        train_f1 = f1_score(y_train, y_train_pred)
        valid_f1 = f1_score(y_valid, y_valid_pred)
        test_f1 = f1_score(y_test, y_test_pred)

        # Log metrics
        wandb.log({
            "f1_score_train": train_f1,
            "f1_score_valid": valid_f1,
            "f1_score_test": test_f1
        })

        # Save model
        model_path = "random_f_sweep_model.pkl"
        joblib.dump(model, model_path)

        artifact = wandb.Artifact(
            name="RandomForest_Sweep_Model",
            type="model"
        )
        artifact.add_file(model_path)
        run.log_artifact(artifact)

# ======================================================
# SWEEP CONFIG
# ======================================================
sweep_config = {
    "method": "grid",
    "metric": {
        "name": "f1_score_valid",
        "goal": "maximize"
    },
    "parameters": {
        "n_estimators": {
            "values": [8, 12, 16, 20]
        },
        "max_depth": {
            "values": [None, 5, 10, 15, 20]
        },
        "min_samples_split": {
            "values": [2, 4, 8, 12]
        }
    }
}

# ------------------------------------------------------
# INITIALIZE SWEEP
# ------------------------------------------------------
sweep_id = wandb.sweep(
    sweep=sweep_config,
    project=PROJECT_NAME
)

# ------------------------------------------------------
# RUN AGENT
# ------------------------------------------------------
wandb.agent(
    sweep_id=sweep_id,
    function=train_rf_model
)
