#!/usr/bin/env python3
"""
Simple validation script - tests core functionality
"""

def test_basic_imports():
    """Test basic imports."""
    print("🧪 Testing basic imports...")
    
    try:
        import pandas as pd
        import numpy as np
        from loguru import logger
        import requests
        from dotenv import load_dotenv
        print("✅ Basic dependencies imported successfully")
        
        from config.settings import Settings
        print("✅ Settings imported successfully")
        
        validation = Settings.validate_config()
        print(f"✅ Settings validation: {validation}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_xgboost():
    """Test XGBoost specifically."""
    print("\n🤖 Testing XGBoost...")
    
    try:
        import xgboost as xgb
        print("✅ XGBoost imported successfully")
        
        # Test basic functionality
        import numpy as np
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        model = xgb.XGBClassifier(n_estimators=10)
        model.fit(X, y)
        print("✅ XGBoost basic functionality works")
        
        return True
        
    except Exception as e:
        print(f"❌ XGBoost error: {e}")
        return False

def test_database():
    """Test database creation."""
    print("\n🗄️ Testing database...")
    
    try:
        from data.database.nba import NBADatabase
        db = NBADatabase()
        summary = db.get_data_summary()
        print("✅ NBA Database works")
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def test_manager():
    """Test data manager."""
    print("\n📊 Testing data manager...")
    
    try:
        from data.manager import UniversalSportsDataManager
        manager = UniversalSportsDataManager(use_redis=False)
        print("✅ Data manager created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Manager error: {e}")
        return False

def test_api_client():
    """Test API client."""
    print("\n🌐 Testing API client...")
    
    try:
        from api_clients.sports_api import SportsAPIClient
        client = SportsAPIClient('nba')
        print("✅ API client created successfully")
        return True
        
    except Exception as e:
        print(f"❌ API client error: {e}")
        return False

def test_nba_model():
    """Test NBA model import."""
    print("\n🏀 Testing NBA model...")
    
    try:
        from models.nba.nba_model import NBAPredictionModel
        model = NBAPredictionModel('game_winner')
        print("✅ NBA model created successfully")
        return True
        
    except Exception as e:
        print(f"❌ NBA model error: {e}")
        return False

def main():
    """Run validation."""
    print("🚀 Simple Sports Platform Validation")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("XGBoost", test_xgboost),
        ("Database", test_database),
        ("Data Manager", test_manager),
        ("API Client", test_api_client),
        ("NBA Model", test_nba_model),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Platform is ready!")
        print("\n🚀 NEXT STEPS:")
        print("1. Try: python examples/NFL/nfl_prediction_pipeline.py")
        print("2. Test data ingestion with real APIs")
        print("3. Create some sample data and train models")
    elif passed >= total - 1:
        print("✅ Almost there! Just one issue to fix.")
    else:
        print("⚠️ A few issues to address.")

if __name__ == "__main__":
    main()
    
