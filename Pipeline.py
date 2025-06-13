import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
import os

class ETLProcessor:
    """
    A class to perform ETL (Extract, Transform, Load) operations for machine learning
    data preprocessing. It handles data loading, feature selection,
    numerical and categorical transformations, and saving of processed data and pipelines.
    """

    def __init__(self, data_url: str):
        """
        Initializes the ETLProcessor with the data source URL.

        Args:
            data_url (str): The URL or path to the dataset.
        """
        self.data_url = data_url
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None

    def extract_data(self) -> pd.DataFrame:
        """
        Extracts data from the specified URL.

        Returns:
            pd.DataFrame: The loaded dataset.
        Raises:
            Exception: If data extraction fails.
        """
        try:
            print(f"Extracting data from: {self.data_url}")
            self.data = pd.read_csv(self.data_url)
            print("Initial Data Preview:\n", self.data.head())
            return self.data
        except Exception as e:
            raise Exception(f"Failed to extract data: {e}")

    def define_features_and_target(self, features: list, target: str):
        """
        Defines the features (X) and target (y) for the model.

        Args:
            features (list): A list of feature column names.
            target (str): The name of the target column.
        """
        if self.data is None:
            raise ValueError("Data not extracted. Call extract_data() first.")

        print(f"\nSelected Features: {features}")
        print(f"Target Variable: {target}")

        self.X = self.data[features]
        self.y = self.data[target]

        self.numeric_features = [f for f in features if self.data[f].dtype in ['int64', 'float64']]
        self.categorical_features = [f for f in features if self.data[f].dtype == 'object']

        print(f"Identified Numeric Features: {self.numeric_features}")
        print(f"Identified Categorical Features: {self.categorical_features}")

    def create_preprocessing_pipeline(self):
        """
        Creates and configures the preprocessing pipeline for numerical and categorical features.
        """
        if not hasattr(self, 'numeric_features') or not hasattr(self, 'categorical_features'):
            raise ValueError("Features and target not defined. Call define_features_and_target() first.")

        print("\nCreating preprocessing pipelines...")

        # Numerical pipeline: Impute missing values with mean and scale
        numeric_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        # Categorical pipeline: Impute missing values with most frequent and one-hot encode
        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine both pipelines using ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_pipeline, self.numeric_features),
                ('cat', categorical_pipeline, self.categorical_features)
            ])
        print("Preprocessing pipeline created successfully.")

    def split_data(self, test_size: float = 0.2, random_state: int = 42):
        """
        Splits the data into training and testing sets.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            random_state (int): Controls the shuffling applied to the data before applying the split.
        """
        if self.X is None or self.y is None:
            raise ValueError("Features or target not defined. Call define_features_and_target() first.")

        print(f"\nSplitting data into training and testing sets (test_size={test_size}, random_state={random_state})...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        print(f"Training data shape: {self.X_train.shape}")
        print(f"Testing data shape: {self.X_test.shape}")

    def transform_data(self):
        """
        Applies the created preprocessing transformations to the training and testing data.
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessing pipeline not created. Call create_preprocessing_pipeline() first.")
        if self.X_train is None or self.X_test is None:
            raise ValueError("Data not split. Call split_data() first.")

        print("\nApplying transformations to data...")
        self.X_train_transformed = self.preprocessor.fit_transform(self.X_train)
        self.X_test_transformed = self.preprocessor.transform(self.X_test)
        print("Data transformation completed.")
        print(f"Transformed X_train shape: {self.X_train_transformed.shape}")
        print(f"Transformed X_test shape: {self.X_test_transformed.shape}")


    def save_processed_data_and_pipeline(self, output_dir: str = "processed_data"):
        """
        Saves the transformed data and the preprocessing pipeline to specified files.

        Args:
            output_dir (str): The directory where processed files will be saved.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        print(f"\nSaving processed data and preprocessor pipeline to '{output_dir}/'...")

        np.save(os.path.join(output_dir, "X_train.npy"), self.X_train_transformed)
        np.save(os.path.join(output_dir, "X_test.npy"), self.X_test_transformed)
        np.save(os.path.join(output_dir, "y_train.npy"), self.y_train)
        np.save(os.path.join(output_dir, "y_test.npy"), self.y_test)
        joblib.dump(self.preprocessor, os.path.join(output_dir, "preprocessor_pipeline.pkl"))

        print("✅ ETL Pipeline completed and files saved.")

# --- Main execution block ---
if __name__ == "__main__":
    DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    FEATURES = ['age', 'fare', 'sex', 'embarked']
    TARGET = 'survived'
    OUTPUT_DIRECTORY = "processed_titanic_data"

    try:
        etl_processor = ETLProcessor(data_url=DATA_URL)
        etl_
