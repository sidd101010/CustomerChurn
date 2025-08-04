#!/usr/bin/env python3
"""
Test script for the Customer Churn Prediction Platform API
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_upload():
    """Test dataset upload"""
    print("Testing dataset upload...")
    
    with open('sample_data.csv', 'rb') as f:
        files = {'file': f}
        data = {'target_column_name': 'churn'}
        
        response = requests.post(f"{BASE_URL}/upload", files=files, data=data)
        
        if response.status_code == 200:
            print("✅ Dataset uploaded successfully!")
            result = response.json()
            print(f"   - Rows: {result['metadata']['dataset_info']['rows']}")
            print(f"   - Columns: {result['metadata']['dataset_info']['columns']}")
            return True
        else:
            print(f"❌ Upload failed: {response.text}")
            return False

def test_analyze():
    """Test dataset analysis"""
    print("\nTesting dataset analysis...")
    
    response = requests.get(f"{BASE_URL}/analyze")
    
    if response.status_code == 200:
        print("✅ Analysis completed successfully!")
        result = response.json()
        print(f"   - Total rows: {result['basic_stats']['total_rows']}")
        print(f"   - Total columns: {result['basic_stats']['total_columns']}")
        return True
    else:
        print(f"❌ Analysis failed: {response.text}")
        return False

def test_preview():
    """Test dataset preview"""
    print("\nTesting dataset preview...")
    
    response = requests.get(f"{BASE_URL}/preview?rows=5")
    
    if response.status_code == 200:
        print("✅ Preview completed successfully!")
        result = response.json()
        print(f"   - Preview rows: {result['preview_rows']}")
        print(f"   - Total rows: {result['total_rows']}")
        return True
    else:
        print(f"❌ Preview failed: {response.text}")
        return False

def test_train():
    """Test model training"""
    print("\nTesting model training...")
    
    response = requests.post(f"{BASE_URL}/train?model_type=random_forest")
    
    if response.status_code == 200:
        print("✅ Model training completed successfully!")
        result = response.json()
        print(f"   - Model type: {result['training_results']['model_type']}")
        print(f"   - Accuracy: {result['training_results']['accuracy']:.2%}")
        print(f"   - Training samples: {result['training_results']['training_samples']}")
        return True
    else:
        print(f"❌ Training failed: {response.text}")
        return False

def test_predict():
    """Test predictions"""
    print("\nTesting predictions...")
    
    # Sample customer data
    customer_data = [
        {
            "customer_id": "TEST001",
            "gender": "Male",
            "age": 35,
            "tenure": 15,
            "monthly_charges": 45.50,
            "total_charges": 682.5,
            "contract": "Month-to-month",
            "payment_method": "Electronic check",
            "internet_service": "DSL",
            "online_security": "No",
            "online_backup": "No",
            "device_protection": "No",
            "tech_support": "No",
            "streaming_tv": "No",
            "streaming_movies": "No",
            "paperless_billing": "Yes"
        },
        {
            "customer_id": "TEST002",
            "gender": "Female",
            "age": 42,
            "tenure": 36,
            "monthly_charges": 85.30,
            "total_charges": 3070.8,
            "contract": "Two year",
            "payment_method": "Credit card",
            "internet_service": "Fiber optic",
            "online_security": "Yes",
            "online_backup": "Yes",
            "device_protection": "Yes",
            "tech_support": "Yes",
            "streaming_tv": "Yes",
            "streaming_movies": "Yes",
            "paperless_billing": "Yes"
        }
    ]
    
    response = requests.post(f"{BASE_URL}/predict", json=customer_data)
    
    if response.status_code == 200:
        print("✅ Predictions completed successfully!")
        result = response.json()
        print(f"   - Total customers: {result['total_customers']}")
        print(f"   - Predictions: {result['predictions']}")
        return True
    else:
        print(f"❌ Predictions failed: {response.text}")
        return False

def test_explain():
    """Test explanations"""
    print("\nTesting explanations...")
    
    # Sample customer data
    customer_data = [
        {
            "customer_id": "TEST001",
            "gender": "Male",
            "age": 35,
            "tenure": 15,
            "monthly_charges": 45.50,
            "total_charges": 682.5,
            "contract": "Month-to-month",
            "payment_method": "Electronic check",
            "internet_service": "DSL",
            "online_security": "No",
            "online_backup": "No",
            "device_protection": "No",
            "tech_support": "No",
            "streaming_tv": "No",
            "streaming_movies": "No",
            "paperless_billing": "Yes"
        }
    ]
    
    response = requests.post(f"{BASE_URL}/explain", json=customer_data)
    
    if response.status_code == 200:
        print("✅ Explanations completed successfully!")
        result = response.json()
        print(f"   - Total customers: {result['total_customers']}")
        if result['results']:
            explanation = result['results'][0]['explanation']
            print(f"   - Sample explanation: {explanation[:100]}...")
        return True
    else:
        print(f"❌ Explanations failed: {response.text}")
        return False

def test_model_info():
    """Test model info"""
    print("\nTesting model info...")
    
    response = requests.get(f"{BASE_URL}/model-info")
    
    if response.status_code == 200:
        print("✅ Model info retrieved successfully!")
        result = response.json()
        print(f"   - Model trained: {result['is_trained']}")
        if result['is_trained']:
            print(f"   - Model type: {result['model_info']['model_type']}")
            print(f"   - Feature count: {result['model_info']['feature_count']}")
        return True
    else:
        print(f"❌ Model info failed: {response.text}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Customer Churn Prediction Platform Tests")
    print("=" * 50)
    
    tests = [
        test_upload,
        test_analyze,
        test_preview,
        test_train,
        test_predict,
        test_explain,
        test_model_info
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The API is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the server and try again.")
    
    print("\n💡 Tips:")
    print("   - Make sure the server is running: python main.py")
    print("   - Check that sample_data.csv exists")
    print("   - Verify all dependencies are installed")

if __name__ == "__main__":
    main() 