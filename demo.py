#!/usr/bin/env python3
"""
Demo script for the Customer Churn Prediction Platform
Shows how to start the server and access the UI
"""

import webbrowser
import time
import subprocess
import sys
import os

def main():
    """Start the server and open the UI"""
    print("🚀 Customer Churn Prediction Platform Demo")
    print("=" * 50)
    
    # Check if required files exist
    required_files = ['main.py', 'ml_model.py', 'genai_explainer.py', 'data_preprocessor.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Please make sure all files are in the current directory.")
        sys.exit(1)
    
    print("✅ All required files found")
    print("\n📋 Starting the server...")
    print("🌐 Web Interface will open automatically at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔧 API Endpoints: http://localhost:8000/api")
    print("\n💡 Tips:")
    print("   - Upload sample_data.csv to test the platform")
    print("   - Use the web interface for interactive testing")
    print("   - Check the API documentation for programmatic access")
    print("\n⏳ Starting server in 3 seconds...")
    
    # Wait a moment
    time.sleep(3)
    
    # Open browser
    try:
        webbrowser.open('http://localhost:8000')
        print("✅ Browser opened automatically")
    except Exception as e:
        print(f"⚠️  Could not open browser automatically: {e}")
        print("   Please open http://localhost:8000 manually")
    
    print("\n🎯 Server is starting...")
    print("   Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start the server
    try:
        subprocess.run([sys.executable, "main.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    main()