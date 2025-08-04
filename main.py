from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import io
import json
from typing import Optional, Dict, List
import os
from datetime import datetime
import numpy as np
from ml_model import ChurnPredictionModel
from genai_explainer import ChurnExplanationGenerator
from data_preprocessor import DataPreprocessor
from utils import safe_json_serialize, convert_numpy_types,safe_cast
import logging
from pydantic import BaseModel,Field
from typing import List


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Churn Prediction Platform",
    description="A FastAPI application for predicting customer churn using ML and GenAI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables to store the uploaded dataset and ML components
uploaded_dataset = None
target_column = None
ml_model = ChurnPredictionModel()
explainer = ChurnExplanationGenerator()
preprocessor = DataPreprocessor()
cleaned_dataset = None


from pydantic import BaseModel, Field

class TrainRequest(BaseModel):
    models: List[str] = Field(..., alias="model_types")

    class Config:
        protected_namespaces = ()  # Optional: disables Pydantic's namespace protection
        populate_by_name = True    # For backwards compatibility with 'allow_population_by_field_name'



@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML interface"""
    with open("templates/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "message": "Customer Churn Prediction Platform API",
        "version": "1.0.0",
        "endpoints": {
            "/upload": "Upload CSV dataset and select target column",
            "/data_cleaning": "Clean and preprocess dataset",
            "/train": "Train multiple ML models",
            "/predict": "Make churn predictions",
            "/explain": "Generate explanations for predictions",
            "/predict_batch": "Batch predictions from CSV"
        }
    }

@app.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    target_column_name: Optional[str] = Form(None)
):
    """
    Upload CSV dataset and optionally specify target column
    
    Args:
        file: CSV file to upload
        target_column_name: Name of the target column (e.g., 'Churn')
    
    Returns:
        Dataset metadata including column types, missing values, and sample preview
    """
    global uploaded_dataset, target_column
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400, 
            detail="Only CSV files are supported"
        )
    
    try:
        # Read the CSV file
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        # Normalize column names to remove leading/trailing spaces
        df.columns = df.columns.str.strip()
        
        # Store the dataset globally
        uploaded_dataset = df
        
        # Validate and assign the target column
        if target_column_name:
            cleaned_target = target_column_name.strip()
            if cleaned_target not in df.columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Target column '{cleaned_target}' not found in dataset after cleaning. Available columns: {df.columns.tolist()}"
                )
            target_column = cleaned_target
        else:
            target_column = None
        
        # Generate dataset metadata
        metadata = generate_dataset_metadata(df, target_column_name)
        
        response_dict = {
            "message": "Dataset uploaded successfully",
            "filename": file.filename,
            "file_size": len(content),
            "upload_timestamp": datetime.now().isoformat(),
            "metadata": metadata
        }

        json_content = json.loads(json.dumps(response_dict, default=convert_numpy_types))

        return JSONResponse(content=json_content, status_code=200)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )

def generate_dataset_metadata(df: pd.DataFrame, target_col: Optional[str] = None) -> dict:
    """
    Generate comprehensive metadata for the uploaded dataset
    
    Args:
        df: Pandas DataFrame
        target_col: Target column name
    
    Returns:
        Dictionary containing dataset metadata
    """
    # Basic dataset info
    metadata = {
        "dataset_info": {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "memory_usage": int(df.memory_usage(deep=True).sum()),
            "shape": [int(x) for x in df.shape]
        },
        "columns": {},
        "missing_values": {},
        "data_types": {},
        "sample_data": {},
        "target_column": target_col if target_col else None
    }
    
    # Analyze each column
    for column in df.columns:
        col_data = df[column]
        # Column info
        metadata["columns"][column] = {
            "unique_values": int(col_data.nunique()),
            "null_count": int(col_data.isnull().sum()),
            "null_percentage": float((col_data.isnull().sum() / len(df)) * 100),
            "data_type": str(col_data.dtype)
        }
        
        # Missing values
        metadata["missing_values"][column] = {
            "count": int(col_data.isnull().sum()),
            "percentage": float((col_data.isnull().sum() / len(df)) * 100)
        }
        
        # Data types
        metadata["data_types"][column] = str(col_data.dtype)
        
        # Sample data (first 5 non-null values)
        sample_values = col_data.dropna().head(5).tolist()
        metadata["sample_data"][column] = [safe_cast(v) for v in sample_values]
    
    # Overall missing values summary
    total_missing = df.isnull().sum().sum()
    metadata["overall_missing"] = {
        "total_missing_values": int(total_missing),
        "total_missing_percentage": float((total_missing / (len(df) * len(df.columns))) * 100)
    }

    
    # Target column validation
    if target_col:
        if target_col not in df.columns:
            metadata["target_column_error"] = f"Target column '{target_col}' not found in dataset"
        else:
            # Clean first
            df[target_col] = df[target_col].astype(str).str.strip().str.lower()
            target_data = df[target_col]  # Now assign cleaned version
        

            metadata["target_column_info"] = {
                "data_type": str(target_data.dtype),
                "unique_values": int(target_data.nunique()),
                "value_counts": {
                    str(k): int(v) for k, v in target_data.value_counts().to_dict().items()
                },
                "null_count": int(target_data.isnull().sum())
            }

    
    return metadata


@app.get("/preview")
async def preview_dataset(rows: int = 10):
    """
    Preview the uploaded dataset
    
    Args:
        rows: Number of rows to preview (default: 10)
    
    Returns:
        Sample of the dataset
    """
    global uploaded_dataset
    
    if uploaded_dataset is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset uploaded. Please upload a CSV file first."
        )
    
    preview_df = uploaded_dataset.head(rows)
    
    return {
        "preview_rows": rows,
        "total_rows": len(uploaded_dataset),
        "data": safe_json_serialize(preview_df.to_dict(orient='records'))
    }

@app.get("/analyze")
async def analyze_dataset():
    """
    Get detailed analysis of the uploaded dataset
    
    Returns:
        Comprehensive dataset analysis
    """
    global uploaded_dataset, target_column
    
    if uploaded_dataset is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset uploaded. Please upload a CSV file first."
        )
    
    df = uploaded_dataset
    
    # Statistical analysis
    analysis = {
        "basic_stats": {
            "total_rows": int(len(df)),
            "total_columns": int(len(df.columns)),
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024)
        },
        "column_analysis": {},
        "correlation_analysis": {},
        "target_analysis": {}
    }
    
    # Analyze each column
    for column in df.columns:
        col_data = df[column]
        
        # Skip non-numeric columns for statistical analysis
        if pd.api.types.is_numeric_dtype(col_data):
            analysis["column_analysis"][column] = {
                "mean": float(col_data.mean()) if not col_data.isnull().all() else None,
                "median": float(col_data.median()) if not col_data.isnull().all() else None,
                "std": float(col_data.std()) if not col_data.isnull().all() else None,
                "min": float(col_data.min()) if not col_data.isnull().all() else None,
                "max": float(col_data.max()) if not col_data.isnull().all() else None,
                "null_count": int(col_data.isnull().sum()),
                "unique_count": int(col_data.nunique())
            }
        else:
            analysis["column_analysis"][column] = {
                "data_type": "categorical",
                "unique_count": int(col_data.nunique()),
                "null_count": int(col_data.isnull().sum()),
                "most_common": {str(k): int(v) for k, v in col_data.value_counts().head(3).to_dict().items()}
            }
    
    # Correlation analysis for numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        correlation_matrix = numeric_df.corr()
        # Convert correlation matrix to JSON-serializable format
        corr_dict = {}
        for col1 in correlation_matrix.columns:
            corr_dict[col1] = {}
            for col2 in correlation_matrix.columns:
                corr_dict[col1][col2] = float(correlation_matrix.loc[col1, col2])
        
        analysis["correlation_analysis"] = {
            "correlation_matrix": corr_dict,
            "high_correlations": []
        }
        
        # Find high correlations
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # High correlation threshold
                    analysis["correlation_analysis"]["high_correlations"].append({
                        "column1": correlation_matrix.columns[i],
                        "column2": correlation_matrix.columns[j],
                        "correlation": float(corr_value)
                    })
    
    # Target column analysis
    if target_column and target_column in df.columns:
        target_data = df[target_column]
        analysis["target_analysis"] = {
            "data_type": str(target_data.dtype),
            "unique_values": int(target_data.nunique()),
            "value_distribution": {str(k): int(v) for k, v in target_data.value_counts().to_dict().items()},
            "null_count": int(target_data.isnull().sum()),
            "balance_ratio": float(target_data.value_counts(normalize=True).iloc[0]) if len(target_data.value_counts()) > 0 else None
        }
    
    return analysis

@app.post("/data_cleaning")
async def clean_dataset(
    missing_strategy: Dict = None,
    categorical_encoding: str = "label",
    scaling: str = "standard"
):
    """
    Clean and preprocess the uploaded dataset
    
    Args:
        missing_strategy: Dictionary of column -> strategy for missing values
        categorical_encoding: 'label' or 'onehot'
        scaling: 'standard', 'minmax', or 'none'
    
    Returns:
        Preprocessing results and cleaned dataset preview
    """
    global uploaded_dataset, target_column, preprocessor, cleaned_dataset
    
    if uploaded_dataset is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset uploaded. Please upload a CSV file first."
        )
    
    if not target_column:
        raise HTTPException(
            status_code=400,
            detail="Target column not specified. Please specify the target column during upload."
        )
    
    try:
        # Clean the dataset
        cleaned_df, preprocessing_info = preprocessor.clean_data(
            uploaded_dataset, 
            target_column, 
            missing_strategy, 
            categorical_encoding, 
            scaling
        )
        
        # Store cleaned dataset
        cleaned_dataset = cleaned_df
        
        # Get preview of cleaned data
        preview_data = cleaned_df.head(10).to_dict(orient='records')
        
        return {
            "message": "Dataset cleaned successfully",
            "preprocessing_info": preprocessing_info,
            "preview_data": preview_data,
            "cleaned_shape": cleaned_df.shape,
            "original_shape": uploaded_dataset.shape
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error cleaning dataset: {str(e)}"
        )

@app.post("/train")
async def train_model(request: TrainRequest):
    """
    Train multiple churn prediction models
    """
    model_types = request.models
    print("model_types:",model_types)
    global uploaded_dataset, target_column, ml_model, cleaned_dataset
    
    print(f"DEBUG: Received request: {request}")
    print(f"DEBUG: ml_model object type: {type(ml_model)}")
    print(f"DEBUG: ml_model attributes: {dir(ml_model)}")
    
    if uploaded_dataset is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset uploaded. Please upload a CSV file first."
        )
    
    if not target_column:
        raise HTTPException(
            status_code=400,
            detail="Target column not specified. Please specify the target column during upload."
        )
    
    # Extract model_types from the request
    
    print(f"DEBUG: Extracted model_types: {model_types}")
    print(f"DEBUG: model_types type: {type(model_types)}")
    
    logger.info(f"Training models: {model_types}")
    
    try:
        # Use cleaned dataset if available, otherwise use original
        dataset_to_use = cleaned_dataset if cleaned_dataset is not None else uploaded_dataset
        print(f"DEBUG: Dataset shape: {dataset_to_use.shape}")
        print(f"DEBUG: Target column: {target_column}")
        
        # Train each model type individually
        all_results = {}
        for i, model_type in enumerate(model_types):
            print(f"DEBUG: Processing model {i+1}/{len(model_types)}: '{model_type}'")
            print(f"DEBUG: model_type type: {type(model_type)}")
            
            try:
                print(f"DEBUG: About to call ml_model.train_models with:")
                print(f"  - dataset_to_use shape: {dataset_to_use.shape}")
                print(f"  - target_column: '{target_column}'")
                print(f"  - model_type: '{model_type}'")
                
                # This is where the error likely occurs
                training_result = ml_model.train_models(dataset_to_use, target_column, model_type)
                
                print(f"DEBUG: Successfully trained {model_type}")
                all_results[model_type] = training_result
                
            except AttributeError as attr_e:
                print(f"DEBUG: AttributeError for {model_type}: {str(attr_e)}")
                print(f"DEBUG: ml_model.__dict__: {ml_model.__dict__}")
                import traceback
                traceback.print_exc()
                raise attr_e
            except Exception as other_e:
                print(f"DEBUG: Other error for {model_type}: {str(other_e)}")
                import traceback
                traceback.print_exc()
                raise other_e
        
        # Find the best model based on accuracy
        best_model_name = max(all_results.keys(), key=lambda k: all_results[k]['accuracy'])
        best_model_results = all_results[best_model_name]
        
        print(f"DEBUG: Best model: {best_model_name} with accuracy: {best_model_results['accuracy']}")
        
        # Generate model summary for best model
        if best_model_results.get('feature_importance'):
            model_summary = explainer.generate_model_summary(
                best_model_results, 
                best_model_results['feature_importance']
            )
        else:
            model_summary = {"model_summary": f"{len(model_types)} model(s) trained successfully"}
        
        return {
            "message": f"{len(model_types)} model(s) trained successfully",
            "results": all_results,
            "best_model": {
                "name": best_model_name,
                "accuracy": best_model_results['accuracy']
            },
            "model_summary": model_summary,
            "model_info": ml_model.get_model_info()
        }
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error training model: {str(e)}"
        )
@app.post("/predict")
async def predict_churn(data: List[Dict], model_name: str = None):
    """
    Make churn predictions for customer data
    
    Args:
        data: List of customer data dictionaries
        model_name: Name of the model to use (if None, uses best model)
    
    Returns:
        Predictions and probabilities
    """
    global ml_model
    
    if not ml_model.is_trained:
        raise HTTPException(
            status_code=400,
            detail="Model not trained. Please train the model first using /train endpoint."
        )
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(data)
        print(f"DEBUG: top5 data:",df.head())
        
        # Make predictions
        prediction_results = ml_model.predict(df, model_name)
        
        return {
            "message": "Predictions completed successfully",
            "predictions": prediction_results["predictions"],
            "probabilities": prediction_results["probabilities"],
            "feature_importance": prediction_results["feature_importance"],
            "total_customers": len(data)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making predictions: {str(e)}"
        )

@app.post("/explain")
async def explain_predictions(data: List[Dict], model_name: str = None):
    """
    Generate explanations for churn predictions
    
    Args:
        data: List of customer data dictionaries
        model_name: Name of the model to use (if None, uses best model)
    
    Returns:
        Predictions with natural language explanations
    """
    global ml_model, explainer
    
    if not ml_model.is_trained:
        raise HTTPException(
            status_code=400,
            detail="Model not trained. Please train the model first using /train endpoint."
        )
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Make predictions
        prediction_results = ml_model.predict(df, model_name)
        
        # Generate explanations
        explanations = explainer.generate_batch_explanations(
            predictions=prediction_results["predictions"],
            probabilities=[max(proba) for proba in prediction_results["probabilities"]],
            customer_data_list=data,
            feature_importance=prediction_results["feature_importance"],
            model_type=prediction_results.get("model_used", "ML Model")
        )
        
        return {
            "message": "Predictions and explanations completed successfully",
            "results": explanations,
            "total_customers": len(data)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating explanations: {str(e)}"
        )

@app.get("/model-info")
async def get_model_info():
    """
    Get information about the trained model
    
    Returns:
        Model information and status
    """
    global ml_model
    
    return {
        "model_info": ml_model.get_model_info(),
        "is_trained": ml_model.is_trained
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 