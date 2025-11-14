"""
TabPFN Client Example Usage
---------------------------
Toy script to check that the TabPFN client is working.
Use the breast cancer dataset for classification and the diabetes dataset for regression,
and try various prediction types.
"""

import argparse
import logging
from unittest.mock import patch

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split

# from tabpfn_client import UserDataClient
from tabpfn_client import TabPFNClassifier, TabPFNRegressor

logging.basicConfig(level=logging.DEBUG)


FULL_BREAST_CANCER_DESCRIPTION = """**Breast Cancer Wisconsin (Original) Data Set.** Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. The target feature records the prognosis (malignant or benign)."""

DIABETES_DESCRIPTION = """**Diabetes Dataset.** Ten baseline variables (age, sex, body mass index, average blood pressure, and six blood serum measurements) were obtained for diabetes patients. The target is a quantitative measure of disease progression one year after baseline."""

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick test for TabPFN reasoning classifier and regressor"
    )
    parser.add_argument(
        "--mode",
        choices=("both", "classifier", "regressor"),
        default="both",
        help="Select which reasoning model(s) to run",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    # Patch webbrowser.open to prevent browser login
    with patch("webbrowser.open", return_value=False):
        if args.mode in ("both", "classifier"):
            # Classification with reasoning
            print("\n" + "="*60)
            print("Testing Classification with Reasoning Mode")
            print("="*60 + "\n")

            X, y = load_breast_cancer(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.33, random_state=42
            )
            dataset_description = FULL_BREAST_CANCER_DESCRIPTION

            print(f"length of dataset description: {len(dataset_description)}")

            tabpfn_r = TabPFNClassifier(
                thinking=True,
                thinking_params={
                    "max_iterations": 2,
                    "prompt_name": "baseline",
                    "langfuse_tags": ["USING_BOTH_WITH_METADATA"]
                }
            )
            # print("checking estimator", check_estimator(tabpfn))
            print("fitting")
            tabpfn_r.fit(X_train[:99], y_train[:99], description=dataset_description)
            print("predicting")
            print(f"Probabilities: {tabpfn_r.predict_proba(X_test)}")
            # print(f"Predictions: {tabpfn_r.predict(X_test)}")

            print(f"last meta: {tabpfn_r.last_meta}")

        if args.mode in ("both", "regressor"):
            # Regression with reasoning
            print("\n" + "="*60)
            print("Testing Regression with Reasoning Mode")
            print("="*60 + "\n")
            
            X, y = load_diabetes(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.33, random_state=42
            )
            
            print(f"length of dataset description: {len(DIABETES_DESCRIPTION)}")
            
            tabpfn_reg_r = TabPFNRegressor(
                thinking=True,
                thinking_params={
                    "max_iterations": 2,
                    "prompt_name": "baseline",
                    "langfuse_tags": ["REGRESSION_REASONING_TEST"]
                }
            )
            
            print("fitting regression model")
            tabpfn_reg_r.fit(X_train[:99], y_train[:99], description=DIABETES_DESCRIPTION)
            print("predicting with regression model (mean)")
            print(tabpfn_reg_r.predict(X_test, output_type="mean"))
            
            # Test predict_full with quantiles
            print("predicting with regression model (full)")
            print(
                tabpfn_reg_r.predict(X_test[:30], output_type="full", quantiles=[0.1, 0.5, 0.9])
            )
            print(f"last meta: {tabpfn_reg_r.last_meta}")
