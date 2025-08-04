import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Configurable data preprocessing for customer churn prediction
    """

    def __init__(self):
        self.scaler = None
        self.label_encoders = {}
        self.onehot_encoder = None
        self.imputer = None
        self.categorical_columns = []
        self.numerical_columns = []
        self.cleaning_config = {}
        self.is_fitted = False

    def analyze_data(self, df: pd.DataFrame) -> Dict:
        """
        Analyze the dataset and return cleaning recommendations

        Args:
            df: Input DataFrame

        Returns:
            Analysis results with recommendations
        """
        analysis = {
            "dataset_info": {
                "rows": len(df),
                "columns": len(df.columns),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
            },
            "column_analysis": {},
            "missing_values": {},
            "categorical_columns": [],
            "numerical_columns": [],
            "recommendations": {}
        }

        for column in df.columns:
            col_data = df[column]

            analysis["column_analysis"][column] = {
                "data_type": str(col_data.dtype),
                "unique_values": col_data.nunique(),
                "null_count": col_data.isnull().sum(),
                "null_percentage": (col_data.isnull().sum() / len(df)) * 100
            }

            analysis["missing_values"][column] = {
                "count": col_data.isnull().sum(),
                "percentage": (col_data.isnull().sum() / len(df)) * 100
            }

            if pd.api.types.is_numeric_dtype(col_data):
                analysis["numerical_columns"].append(column)
            else:
                analysis["categorical_columns"].append(column)

        analysis["recommendations"] = self._generate_recommendations(df, analysis)

        return analysis

    def _generate_recommendations(self, df: pd.DataFrame, analysis: Dict) -> Dict:
        recommendations = {
            "missing_value_strategy": {},
            "categorical_encoding": {},
            "scaling_strategy": "standard",
            "suggested_actions": []
        }

        for column, missing_info in analysis["missing_values"].items():
            if missing_info["percentage"] > 50:
                recommendations["missing_value_strategy"][column] = "drop"
                recommendations["suggested_actions"].append(f"Drop column '{column}' (high missing values)")
            elif missing_info["percentage"] > 0:
                if column in analysis["numerical_columns"]:
                    recommendations["missing_value_strategy"][column] = "impute_median"
                else:
                    recommendations["missing_value_strategy"][column] = "impute_mode"
                recommendations["suggested_actions"].append(f"Impute missing values in '{column}'")
            else:
                recommendations["missing_value_strategy"][column] = "none"

        for column in analysis["categorical_columns"]:
            unique_count = analysis["column_analysis"][column]["unique_values"]
            if unique_count <= 5:
                recommendations["categorical_encoding"][column] = "label"
            else:
                recommendations["categorical_encoding"][column] = "onehot"

        if len(analysis["numerical_columns"]) > 0:
            recommendations["suggested_actions"].append("Apply scaling to numerical features")

        return recommendations

    def clean_data(self,
                   df: pd.DataFrame,
                   target_column: str,
                   missing_strategy: Dict = None,
                   categorical_encoding: str = "label",
                   scaling: str = "standard") -> Tuple[pd.DataFrame, Dict]:
        self.cleaning_config = {
            "missing_strategy": missing_strategy or {},
            "categorical_encoding": categorical_encoding,
            "scaling": scaling,
            "target_column": target_column
        }

        cleaned_df = df.copy()

        cleaned_df = self._handle_missing_values(cleaned_df, missing_strategy)
        cleaned_df = self._encode_categorical_variables(cleaned_df, target_column, categorical_encoding)
        cleaned_df = self._scale_numerical_variables(cleaned_df, target_column, scaling)

        self.categorical_columns = [col for col in cleaned_df.columns
                                     if not pd.api.types.is_numeric_dtype(cleaned_df[col]) and col != target_column]
        self.numerical_columns = [col for col in cleaned_df.columns
                                   if pd.api.types.is_numeric_dtype(cleaned_df[col]) and col != target_column]

        self.is_fitted = True

        preprocessing_info = {
            "original_shape": df.shape,
            "cleaned_shape": cleaned_df.shape,
            "missing_values_handled": len([s for s in missing_strategy.values() if s != "none"]) if missing_strategy else 0,
            "categorical_columns_encoded": len(self.categorical_columns),
            "numerical_columns_scaled": len(self.numerical_columns),
            "encoding_method": categorical_encoding,
            "scaling_method": scaling,
            "feature_columns": self.categorical_columns + self.numerical_columns
        }

        return cleaned_df, preprocessing_info

    def _handle_missing_values(self, df: pd.DataFrame, missing_strategy: Dict) -> pd.DataFrame:
        for column, strategy in missing_strategy.items():
            if column not in df.columns:
                continue

            if strategy == "drop":
                df = df.dropna(subset=[column])
            elif strategy == "impute_median":
                imputer = SimpleImputer(strategy='median')
                df[column] = imputer.fit_transform(df[[column]]).ravel()
            elif strategy == "impute_mode":
                imputer = SimpleImputer(strategy='most_frequent')
                df[column] = imputer.fit_transform(df[[column]]).ravel()
            elif strategy == "impute_mean":
                imputer = SimpleImputer(strategy='mean')
                df[column] = imputer.fit_transform(df[[column]]).ravel()

        return df


    def _encode_categorical_variables(self, df: pd.DataFrame, target_column: str, encoding_method: str) -> pd.DataFrame:
        categorical_columns = [col for col in df.columns
                               if not pd.api.types.is_numeric_dtype(df[col]) and col != target_column]

        if encoding_method == "label":
            for column in categorical_columns:
                if column not in self.label_encoders:
                    self.label_encoders[column] = LabelEncoder()
                    df[column] = self.label_encoders[column].fit_transform(df[column])
                else:
                    df[column] = self.label_encoders[column].transform(df[column])

        elif encoding_method == "onehot":
            if categorical_columns:
                self.onehot_encoder = OneHotEncoder(drop='first', sparse=False)
                encoded_features = self.onehot_encoder.fit_transform(df[categorical_columns])
                encoded_df = pd.DataFrame(
                    encoded_features,
                    columns=self.onehot_encoder.get_feature_names_out(categorical_columns),
                    index=df.index
                )
                df = df.drop(columns=categorical_columns)
                df = pd.concat([df, encoded_df], axis=1)

        return df

    def _scale_numerical_variables(self, df: pd.DataFrame, target_column: str, scaling_method: str) -> pd.DataFrame:
        numerical_columns = [col for col in df.columns
                             if pd.api.types.is_numeric_dtype(df[col]) and col != target_column]

        if scaling_method == "standard" and numerical_columns:
            self.scaler = StandardScaler()
            df[numerical_columns] = self.scaler.fit_transform(df[numerical_columns])
        elif scaling_method == "minmax" and numerical_columns:
            self.scaler = MinMaxScaler()
            df[numerical_columns] = self.scaler.fit_transform(df[numerical_columns])

        return df

    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming new data")

        transformed_df = df.copy()

        if self.cleaning_config.get("missing_strategy"):
            transformed_df = self._handle_missing_values(transformed_df, self.cleaning_config["missing_strategy"])

        transformed_df = self._encode_categorical_variables(
            transformed_df,
            self.cleaning_config.get("target_column", ""),
            self.cleaning_config.get("categorical_encoding", "label")
        )

        transformed_df = self._scale_numerical_variables(
            transformed_df,
            self.cleaning_config.get("target_column", ""),
            self.cleaning_config.get("scaling", "standard")
        )

        return transformed_df

    def get_preprocessing_summary(self) -> Dict:
        if not self.is_fitted:
            return {"status": "Preprocessor not fitted"}

        return {
            "cleaning_config": self.cleaning_config,
            "categorical_columns": self.categorical_columns,
            "numerical_columns": self.numerical_columns,
            "label_encoders": list(self.label_encoders.keys()),
            "scaler_type": type(self.scaler).__name__ if self.scaler else None,
            "onehot_encoder": "fitted" if self.onehot_encoder else None
        }


 
