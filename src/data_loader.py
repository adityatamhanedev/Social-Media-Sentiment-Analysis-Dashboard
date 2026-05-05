import pandas as pd
import os


def load_data(file_path):
    """
    Load dataset from CSV file.

    Parameters:
    file_path (str): Path to the CSV file

    Returns:
    df (DataFrame): Loaded pandas DataFrame
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at path: {file_path}")

    try:
        df = pd.read_csv(file_path)
        print("✅ Data loaded successfully!")
        print(f"Shape of dataset: {df.shape}")
        return df

    except Exception as e:
        print("❌ Error while loading data:", e)
        return None


def preview_data(df, n=5):
    """
    Display first few rows of dataset

    Parameters:
    df (DataFrame): Input dataset
    n (int): Number of rows to show
    """
    print("\n🔍 Data Preview:")
    print(df.head(n))


def check_nulls(df):
    """
    Check missing values in dataset

    Parameters:
    df (DataFrame): Input dataset
    """
    print("\n🧪 Missing Values:")
    print(df.isnull().sum())


def validate_columns(df, required_columns):
    """
    Validate if required columns exist

    Parameters:
    df (DataFrame): Input dataset
    required_columns (list): List of required column names
    """

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"❌ Missing columns: {missing}")
    else:
        print("✅ All required columns are present!")


def basic_info(df):
    """
    Print basic dataset info
    """
    print("\n📊 Dataset Info:")
    print(df.info())