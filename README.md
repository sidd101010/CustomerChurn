# Customer Churn Prediction Platform

A FastAPI-based application for predicting customer churn using traditional Machine Learning and generating natural language explanations using Generative AI.

## Features

- **Dataset Upload & Analysis**: Upload CSV files and analyze dataset metadata
- **Data Cleaning & Preprocessing**: Configurable data cleaning with missing value handling, categorical encoding, and scaling
- **Multiple ML Models**: Train Random Forest, Logistic Regression, XGBoost, SVM, KNN, and Decision Tree models
- **Churn Prediction**: Make predictions on new customer data with model selection
- **Batch Prediction**: Process CSV files for batch predictions with explanations
- **AI Explanations**: Generate natural language explanations for predictions using OpenAI GPT
- **Interactive Web Interface**: User-friendly HTML frontend for testing

## System Requirements

- Python 3.8+
- OpenAI API key (optional, for enhanced explanations)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd CustomerChurn
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up OpenAI API key** (optional):
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   Or create a `.env` file:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## Usage

### 1. Start the Server

#### Option 1: Quick Start (with UI)
```bash
python demo.py
```
This will start the server and automatically open the web interface in your browser.

#### Option 2: Manual Start
```bash
python main.py
# or
python start_server.py
```

The server will start at `http://localhost:8000`

### 2. Access the Web Interface

Open `http://localhost:8000` in your browser to use the interactive interface.

### 3. API Endpoints

#### Upload Dataset
```bash
POST /upload
```
- **Parameters**: 
  - `file`: CSV file
  - `target_column`: Target column name (e.g., "Churn")
- **Returns**: Dataset metadata including column types, missing values, and sample preview

#### Analyze Dataset
```bash
GET /analyze
```
- **Returns**: Comprehensive dataset analysis with statistics and correlations

#### Preview Dataset
```bash
GET /preview?rows=10
```
- **Parameters**: `rows`: Number of rows to preview
- **Returns**: Sample of the uploaded dataset

#### Data Cleaning
```bash
POST /data_cleaning
```
- **Parameters**: 
  - `missing_strategy`: Dictionary of column -> strategy for missing values
  - `categorical_encoding`: "label" or "onehot"
  - `scaling`: "standard", "minmax", or "none"
- **Returns**: Preprocessing results and cleaned dataset preview

#### Train Models
```bash
POST /train
```
- **Parameters**: `model_types`: List of model types to train (default: ["random_forest", "logistic_regression", "xgboost"])
- **Returns**: Training results and model information for all models

#### Make Predictions
```bash
POST /predict
```
- **Body**: JSON array of customer data
- **Parameters**: `model_name`: Name of the model to use (optional)
- **Returns**: Predictions and probabilities

#### Generate Explanations
```bash
POST /explain
```
- **Body**: JSON array of customer data
- **Parameters**: `model_name`: Name of the model to use (optional)
- **Returns**: Predictions with natural language explanations

#### Batch Predictions
```bash
POST /predict_batch
```
- **Parameters**: 
  - `file`: CSV file with customer data
  - `model_name`: Name of the model to use (optional)
- **Returns**: Predictions with explanations for each row

#### Get Model Info
```bash
GET /model-info
```
- **Returns**: Current model information and status

## Example Usage

### 1. Upload a Dataset

```python
import requests

# Upload CSV file
with open('customer_data.csv', 'rb') as f:
    files = {'file': f}
    data = {'target_column': 'Churn'}
    response = requests.post('http://localhost:8000/upload', files=files, data=data)
    print(response.json())
```

### 2. Train a Model

```python
# Train Random Forest model
response = requests.post('http://localhost:8000/train?model_type=random_forest')
print(response.json())
```

### 3. Make Predictions

```python
# Customer data for prediction
customer_data = [
    {
        "customer_id": "C001",
        "tenure": 12,
        "monthly_charges": 29.85,
        "total_charges": 358.2,
        "contract": "Month-to-month",
        "payment_method": "Electronic check"
    }
]

# Make prediction
response = requests.post('http://localhost:8000/predict', json=customer_data)
print(response.json())

# Get prediction with explanation
response = requests.post('http://localhost:8000/explain', json=customer_data)
print(response.json())
```

## Dataset Format

The platform expects CSV files with the following characteristics:

- **Target Column**: Binary or categorical column indicating churn status
- **Features**: Mix of numerical and categorical features
- **Missing Values**: Handled automatically during preprocessing

### Sample Dataset Structure

```csv
customer_id,tenure,monthly_charges,total_charges,contract,payment_method,churn
C001,12,29.85,358.2,Month-to-month,Electronic check,Yes
C002,24,56.95,1366.8,One year,Credit card,No
C003,6,19.85,119.1,Month-to-month,Electronic check,Yes
```

## Model Information

### Supported Models

1. **Random Forest Classifier**
   - Handles both numerical and categorical features
   - Provides feature importance scores
   - Good for complex, non-linear relationships

2. **Logistic Regression**
   - Linear model with regularization
   - Interpretable coefficients
   - Good for linear relationships

3. **XGBoost**
   - Gradient boosting algorithm
   - Excellent performance on structured data
   - Built-in feature importance

4. **Support Vector Machine (SVM)**
   - Good for high-dimensional data
   - Effective with kernel methods

5. **K-Nearest Neighbors (KNN)**
   - Simple and interpretable
   - Good for small datasets

6. **Decision Tree**
   - Highly interpretable
   - Good for feature selection

### Preprocessing

- **Missing Values**: Configurable strategies (drop, impute_median, impute_mode, impute_mean)
- **Categorical Encoding**: Label encoding or One-hot encoding
- **Feature Scaling**: StandardScaler, MinMaxScaler, or no scaling
- **Data Splitting**: 80% training, 20% testing with stratification
- **Model Selection**: Automatic selection of best performing model

## AI Explanations

The platform uses OpenAI's GPT model to generate natural language explanations for predictions:

- **Business-friendly language**: Clear, non-technical explanations
- **Feature importance**: Highlights key factors contributing to predictions
- **Actionable insights**: Suggests retention strategies for at-risk customers
- **Fallback mode**: Works without OpenAI API using rule-based explanations

## API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI).

## Project Structure

```
CustomerChurn/
├── main.py                 # FastAPI application
├── ml_model.py            # Machine learning model
├── genai_explainer.py     # AI explanation generator
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html         # Web interface
└── README.md              # This file
```

## Error Handling

The application includes comprehensive error handling:

- **File validation**: Ensures CSV files are uploaded
- **Data validation**: Checks for required columns and data types
- **Model validation**: Ensures model is trained before predictions
- **API error handling**: Graceful handling of OpenAI API errors

## Performance Considerations

- **Memory usage**: Large datasets are processed efficiently
- **Model persistence**: Trained models can be saved and loaded
- **Batch processing**: Supports multiple predictions at once
- **Caching**: Consider implementing Redis for production use

## Security Notes

- **File upload**: Only CSV files are accepted
- **API key**: Store OpenAI API key securely
- **CORS**: Configured for development; adjust for production
- **Input validation**: All inputs are validated and sanitized

## Troubleshooting

### Common Issues

1. **"No dataset uploaded"**: Upload a CSV file first
2. **"Model not trained"**: Train the model before making predictions
3. **"Target column not found"**: Check the target column name in your dataset
4. **OpenAI API errors**: Check your API key and internet connection
5. **"Object of type int64 is not JSON serializable"**: Fixed - NumPy types are now automatically converted

### JSON Serialization Fix

The platform now automatically handles NumPy data types (int64, float64, etc.) by converting them to native Python types for JSON serialization. This prevents the common error when uploading CSV files with numeric data.

### Debug Mode

Enable debug logging by modifying the logging level in the modules.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions, please create an issue in the repository. 