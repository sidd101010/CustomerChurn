import numpy as np
import pandas as pd
from typing import Any, Dict, List, Union

def convert_numpy_types(obj: Any) -> Any:
    """
    Convert NumPy types to native Python types for JSON serialization
    
    Args:
        obj: Object that may contain NumPy types
        
    Returns:
        Object with NumPy types converted to native Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, pd.Series):
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, pd.DataFrame):
        return convert_numpy_types(obj.to_dict())
    else:
        return obj

def safe_json_serialize(obj: Any) -> Any:
    """
    Safely serialize an object for JSON response
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON-serializable object
    """
    return convert_numpy_types(obj)

def convert_dataframe_to_dict(df: pd.DataFrame) -> Dict:
    """
    Convert DataFrame to JSON-serializable dictionary
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        Dictionary representation of DataFrame
    """
    return convert_numpy_types(df.to_dict(orient='records'))

def convert_series_to_dict(series: pd.Series) -> Dict:
    """
    Convert Series to JSON-serializable dictionary
    
    Args:
        series: Pandas Series
        
    Returns:
        Dictionary representation of Series
    """
    return convert_numpy_types(series.to_dict()) 


def safe_cast(value):
    """
    Safely cast NumPy types and other non-serializable objects to Python native types.
    """
    import numpy as np

    if isinstance(value, (np.integer,)):
        return int(value)
    elif isinstance(value, (np.floating,)):
        return float(value)
    elif isinstance(value, (np.bool_)):
        return bool(value)
    return value  # str, list, etc. are already serializable
