import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
from typing import Dict, List, Tuple, Optional
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnPredictionModel:
    """
    Machine Learning model for customer churn prediction
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = None
        self.model_type = None
        self.is_trained = False


    def preprocess_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess the dataset for training

        Args:
            df: Input DataFrame
            target_column: Name of the target column

        Returns:
            Tuple of (X, y) for training
        """
        # Create a copy to avoid modifying original data
        data = df.copy()

        # Handle missing values
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        categorical_columns = data.select_dtypes(include=['object']).columns

        # Initialize dictionary to store imputation values
        self.imputation_values = {}

        # Fill missing values and store medians for numeric columns
        for col in numeric_columns:
            if col != target_column:
                median_val = data[col].median()
                data[col] = data[col].fillna(median_val)
                self.imputation_values[col] = median_val  # store median for later imputation

        # Fill missing values and store modes for categorical columns
        for col in categorical_columns:
            if col != target_column:
                mode_val = data[col].mode()[0] if len(data[col].mode()) > 0 else 'Unknown'
                data[col] = data[col].fillna(mode_val)
                self.imputation_values[col] = mode_val  # store mode for later imputation

        # Encode categorical variables
        for col in categorical_columns:
            if col != target_column:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    data[col] = self.label_encoders[col].fit_transform(data[col])
                else:
                    # Handle new categories in test data
                    unique_values = data[col].unique()
                    known_values = self.label_encoders[col].classes_
                    unknown_values = set(unique_values) - set(known_values)

                    if unknown_values:
                        most_common = data[col].mode()[0] if len(data[col].mode()) > 0 else known_values[0]
                        data[col] = data[col].replace(list(unknown_values), most_common)

                    data[col] = self.label_encoders[col].transform(data[col])

        # Prepare features and target
        feature_columns = [col for col in data.columns if col != target_column]
        self.feature_columns = feature_columns
        self.target_column = target_column

        X = data[feature_columns]
        y = data[target_column]

        # Convert target to binary if needed
        if y.dtype == 'object':
            if y.nunique() == 2:
                # Binary classification
                y = (y == y.value_counts().index[0]).astype(int)
            else:
                # Multi-class - encode
                if target_column not in self.label_encoders:
                    self.label_encoders[target_column] = LabelEncoder()
                    y = self.label_encoders[target_column].fit_transform(y)
                else:
                    y = self.label_encoders[target_column].transform(y)

        return X, y

    
    def train_models(self, df: pd.DataFrame, target_column: str, model_type: str = 'random_forest') -> Dict:
        print(f"DEBUG: Entered train_models with model_type: {model_type}")
        print(f"DEBUG: self.__dict__ = {self.__dict__}")  # This will show all attributes
        
        logger.info(f"Starting training for model: {model_type}")
        try:
            # Preprocess data
            X, y = self.preprocess_data(df, target_column)
           
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Initialize and train model
            if model_type == 'random_forest':
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
            elif model_type == 'logistic_regression':
                self.model = LogisticRegression(random_state=42, max_iter=1000)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            self.model_type = model_type
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Feature importance (for Random Forest)
            feature_importance = {}
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
            
            self.is_trained = True
            
            return {
                "model_type": model_type,
                "accuracy": accuracy,
                "confusion_matrix": conf_matrix.tolist(),
                "classification_report": class_report,
                "feature_importance": feature_importance,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "feature_count": len(self.feature_columns)
            }
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise e
    
    def predict(self, data: pd.DataFrame, model_name: Optional[str] = None) -> Dict:
        if model_name:
            model = self.get_model_by_name(model_name)
            print(f"DEBUG: model Received by the USER:", model)
        else:
            model = self.model  # default/best model

        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        try:
            processed_data = data.copy()

            # Step 1: Identify missing columns
            missing_cols = [col for col in self.feature_columns if col not in processed_data.columns]
            print(f"DEBUG: missing_cols:", missing_cols)

            # Step 2: Add missing columns with NaN
            for col in missing_cols:
                processed_data[col] = np.nan

            print("DEBUG: Columns after NaN assignment:", processed_data.columns.tolist())


            # Step 3: Fill missing values for all feature columns (numeric and categorical)
            for col in missing_cols:
                if col in self.imputation_values:
                    processed_data[col] = self.imputation_values[col]
                else:
                    processed_data[col] = 'Unknown'  # fallback for categorical or unknown

            # Step 4: Encode categorical variables
            for col in self.feature_columns:
                if col in self.label_encoders:
                    unique_values = processed_data[col].unique()
                    known_values = self.label_encoders[col].classes_
                    unknown_values = set(unique_values) - set(known_values)

                    if unknown_values:
                        most_common = processed_data[col].mode()[0] if len(processed_data[col].mode()) > 0 else known_values[0]
                        processed_data[col] = processed_data[col].replace(list(unknown_values), most_common)

                    processed_data[col] = self.label_encoders[col].transform(processed_data[col])

            # Step 5: Scale and predict
            X = processed_data[self.feature_columns]
            X_scaled = self.scaler.transform(X)

            predictions = model.predict(X_scaled)
            prediction_proba = model.predict_proba(X_scaled)

            # Step 6: Decode predictions if label-encoded
            if self.target_column in self.label_encoders:
                predictions = self.label_encoders[self.target_column].inverse_transform(predictions)

            return {
                "predictions": predictions.tolist(),
                "probabilities": prediction_proba.tolist(),
                "feature_importance": dict(zip(self.feature_columns, model.feature_importances_)) if hasattr(model, 'feature_importances_') else {}
            }

        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise e


    
    def save_model(self, filepath: str):
        """Save the trained model to disk"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from disk"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.target_column = model_data['target_column']
        self.model_type = model_data['model_type']
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_info(self) -> Dict:
        """Get information about the trained model"""
        if not self.is_trained:
            return {"status": "Model not trained"}
        
        return {
            "model_type": self.model_type,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "feature_count": len(self.feature_columns),
            "is_trained": self.is_trained
        }