#!/usr/bin/env python3
"""
Startup script for the Customer Churn Prediction Platform
"""

import uvicorn
import os
import sys

def main():
    """Start the FastAPI server"""
    print("🚀 Starting Customer Churn Prediction Platform...")
    print("=" * 50)
    
    # Check if required files exist
    required_files = ['main.py', 'ml_model.py', 'genai_explainer.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Please make sure all files are in the current directory.")
        sys.exit(1)
    
    # Check if sample data exists
    if not os.path.exists('sample_data.csv'):
        print("⚠️  Warning: sample_data.csv not found")
        print("   You can still use the API with your own CSV files")
    
    print("✅ All required files found")
    print("📊 API will be available at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🌐 Web Interface: http://localhost:8000")
    print("=" * 50)
    
    # Start the server
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()