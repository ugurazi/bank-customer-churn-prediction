#!/usr/bin/env python
"""
Download Bank Customer Churn dataset from Kaggle.
Requires: pip install kaggle

Setup:
1. Go to https://www.kaggle.com/settings/account
2. Download API key JSON
3. Place at ~/.kaggle/kaggle.json
4. chmod 600 ~/.kaggle/kaggle.json
"""

import os
import sys

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    print("kaggle package not found. Install with: pip install kaggle")
    sys.exit(1)

def download_dataset():
    api = KaggleApi()
    api.authenticate()

    dataset_name = "gauravtopre/bank-customer-churn-dataset"
    output_dir = "data"

    print(f"Downloading {dataset_name}...")
    api.dataset_download_files(dataset_name, path=output_dir, unzip=True)
    print(f"Dataset downloaded to {output_dir}/")

    # Find and rename the CSV file
    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
    if csv_files:
        source = os.path.join(output_dir, csv_files[0])
        target = os.path.join(output_dir, 'bank_churn.csv')
        if source != target:
            os.rename(source, target)
        print(f"Data saved as: {target}")
        return target

    return None

if __name__ == "__main__":
    download_dataset()
