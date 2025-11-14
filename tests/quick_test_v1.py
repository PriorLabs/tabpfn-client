import os
import json
import requests
from typing import Optional

from tabpfn_common_utils.utils import get_example_dataset

TARGET_NAME = "target"


def call_fit(
    train_path: str, target_name: str = TARGET_NAME, api_key: Optional[str] = None
) -> str:
    """
    Call the /v1/fit endpoint to train a model.

    Args:
        train_path: Path to the training CSV file
        target_name: Name of the target column (default: "target")
        api_key: API key for authentication (if None, reads from PRIORLABS_API_KEY env var)

    Returns:
        model_id: The model ID returned from the fit endpoint
    """
    headers = {"Authorization": f"Bearer {api_key}"}

    payload = {
        "task": "classification",
        "schema": {
            "target": target_name,
            "description": "Iris dataset (quick test)",
        },
    }
    files = {
        "data": (None, json.dumps(payload), "application/json"),
        "dataset_file": (train_path, open(train_path, "rb")),
    }
    response = requests.post(
        "http://localhost:8000/v1/fit",
        headers=headers,
        files=files,
    )
    if response.status_code != 200:
        raise RuntimeError(f"[FIT] HTTP {response.status_code}: {response.text}")
    res_j = response.json()
    model_id = res_j.get("model_id")
    if not model_id:
        raise RuntimeError(f"[FIT] No model_id in response: {res_j}")
    print(f"✅ Model trained: {model_id}")
    return model_id


def call_predict(test_path: str, model_id: str, api_key: Optional[str] = None) -> None:
    """
    Call the /v1/predict endpoint to get predictions.

    Args:
        test_path: Path to the test CSV file
        model_id: The model ID from the fit call
        api_key: API key for authentication (if None, reads from PRIORLABS_API_KEY env var)
    """
    headers = {"Authorization": f"Bearer {api_key}"}

    payload = {
        "task": "classification",
        "model_id": model_id,
    }
    files = {
        "data": (None, json.dumps(payload), "application/json"),
        "file": (test_path, open(test_path, "rb")),
    }
    response = requests.post(
        "http://localhost:8000/v1/predict",
        headers=headers,
        files=files,
    )
    if response.status_code != 200:
        raise RuntimeError(f"[PREDICT] HTTP {response.status_code}: {response.text}")
    print("✅ Predictions:")
    print(json.dumps(response.json(), indent=2))


def main() -> None:
    """Main function to generate dataset and test both endpoints."""
    # === Generate/train/test data as in quick_test.py ===
    x_train, x_test, y_train, y_test = get_example_dataset("iris")

    train_df = x_train.copy()
    train_df[TARGET_NAME] = y_train.values

    test_df = x_test.copy()
    test_df[TARGET_NAME] = y_test.values

    train_path = "train.csv"
    test_path = "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Get the API key
    api_key = os.getenv("PRIORLABS_API_KEY")

    # Test /v1/fit
    print("--- Testing /v1/fit ---")
    model_id = call_fit(train_path, api_key=api_key)

    # Test /v1/predict
    print("\n--- Testing /v1/predict ---")
    call_predict(test_path, model_id, api_key=api_key)


if __name__ == "__main__":
    main()
