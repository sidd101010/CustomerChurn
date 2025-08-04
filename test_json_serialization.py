#!/usr/bin/env python3
"""
Test script to verify JSON serialization works with NumPy types
"""

import json
import numpy as np
import pandas as pd
from utils import safe_json_serialize

def test_json_serialization():
    """Test JSON serialization with various NumPy types"""
    print("🧪 Testing JSON Serialization...")
    
    # Test data with NumPy types
    test_data = {
        "int64_value": np.int64(42),
        "float64_value": np.float64(3.14),
        "array": np.array([1, 2, 3]),
        "nested_dict": {
            "numpy_int": np.int32(100),
            "numpy_float": np.float32(2.718)
        },
        "list_with_numpy": [np.int64(1), np.float64(2.5), "string"]
    }
    
    try:
        # Try to serialize without conversion (should fail)
        try:
            json.dumps(test_data)
            print("❌ Original data should not be JSON serializable")
        except TypeError:
            print("✅ Original data correctly not JSON serializable")
        
        # Convert and serialize
        converted_data = safe_json_serialize(test_data)
        json_string = json.dumps(converted_data)
        print("✅ Converted data is JSON serializable")
        
        # Verify the conversion
        assert isinstance(converted_data["int64_value"], int)
        assert isinstance(converted_data["float64_value"], float)
        assert isinstance(converted_data["array"], list)
        print("✅ Type conversion successful")
        
        print("🎉 All JSON serialization tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_dataframe_serialization():
    """Test DataFrame serialization"""
    print("\n📊 Testing DataFrame Serialization...")
    
    # Create test DataFrame
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [1.1, 2.2, 3.3],
        'C': ['a', 'b', 'c']
    })
    
    try:
        # Convert DataFrame to dict
        df_dict = safe_json_serialize(df.to_dict(orient='records'))
        json_string = json.dumps(df_dict)
        print("✅ DataFrame serialization successful")
        return True
    except Exception as e:
        print(f"❌ DataFrame serialization failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 JSON Serialization Test Suite")
    print("=" * 40)
    
    test1 = test_json_serialization()
    test2 = test_dataframe_serialization()
    
    if test1 and test2:
        print("\n🎉 All tests passed! JSON serialization is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")