"""
Sample client script to test the MRD Detection API.
"""

import requests
import pandas as pd
import numpy as np
import sys


def test_health():
    """Test the /health endpoint."""
    print("\n" + "=" * 60)
    print("Testing /health endpoint...")
    print("=" * 60)
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")


def test_info():
    """Test the /info endpoint."""
    print("\n" + "=" * 60)
    print("Testing /info endpoint...")
    print("=" * 60)
    try:
        response = requests.get("http://localhost:8000/info")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")


def test_predict_single_file(csv_file_path):
    """Test /predict with a single CSV file."""
    print("\n" + "=" * 60)
    print(f"Testing /predict with single file: {csv_file_path}")
    print("=" * 60)

    try:
        with open(csv_file_path, "rb") as f:
            files = {"files": (csv_file_path, f, "text/csv")}
            response = requests.post("http://localhost:8000/predict", files=files)

        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response:")
        for key, value in result.items():
            print(f"  {key}: {value}")

        return result
    except Exception as e:
        print(f"Error: {e}")
        return None


def test_predict_multiple_files(csv_file_paths):
    """Test /predict with multiple CSV files (multi-file upload)."""
    print("\n" + "=" * 60)
    print(f"Testing /predict with {len(csv_file_paths)} files")
    print("=" * 60)

    try:
        files = []
        for path in csv_file_paths:
            files.append(("files", (path, open(path, "rb"), "text/csv")))

        response = requests.post("http://localhost:8000/predict", files=files)

        # Close files
        for _, (_, f, _) in files:
            f.close()

        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response:")
        for key, value in result.items():
            print(f"  {key}: {value}")

        return result
    except Exception as e:
        print(f"Error: {e}")
        return None


def test_predict_with_details(csv_file_path):
    """Test /predict with return_details=True."""
    print("\n" + "=" * 60)
    print(f"Testing /predict with details flag: {csv_file_path}")
    print("=" * 60)

    try:
        with open(csv_file_path, "rb") as f:
            files = {"files": (csv_file_path, f, "text/csv")}
            response = requests.post(
                "http://localhost:8000/predict",
                params={"return_details": True},
                files=files,
            )

        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response:")
        for key, value in result.items():
            print(f"  {key}: {value}")

        return result
    except Exception as e:
        print(f"Error: {e}")
        return None


def create_sample_csv(output_path, n_samples=1000, n_features=14):
    """Create a sample CSV file for testing."""
    print(
        f"\nCreating sample CSV file with {n_samples} rows and {n_features} features..."
    )
    data = np.random.rand(n_samples, n_features)
    df = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(n_features)])
    df.to_csv(output_path, index=False)
    print(f"Sample CSV saved to: {output_path}")


if __name__ == "__main__":
    print("MRD Detection API - Sample Client")
    print("=" * 60)

    # Test basic endpoints
    test_health()
    test_info()

    # Create sample CSV files for testing
    sample_file_1 = "sample_data_1.csv"
    sample_file_2 = "sample_data_2.csv"
    sample_file_3 = "sample_data_3.csv"

    create_sample_csv(sample_file_1, n_samples=5000, n_features=14)
    create_sample_csv(sample_file_2, n_samples=5000, n_features=14)
    create_sample_csv(sample_file_3, n_samples=3000, n_features=14)

    # Test single file prediction
    test_predict_single_file(sample_file_1)

    # Test with details
    test_predict_with_details(sample_file_1)

    # Test multiple files (concatenation)
    test_predict_multiple_files([sample_file_1, sample_file_2, sample_file_3])

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
