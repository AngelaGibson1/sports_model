#!/usr/bin/env python3
"""
Quick setup validation script to test all imports and basic functionality.
Run this after installing requirements to make sure everything works.
"""

import sys
from pathlib import Path

def test_imports():
    """Test all critical imports."""
    print("🧪 Testing imports...")
    
    try:
        # Core dependencies
        import pandas as pd
        import numpy as np
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from loguru import logger
        import requests
        from dotenv import load_dotenv
        print("✅ Core dependencies imported successfully")
        
        # Project imports
        from config.settings import Settings
        print("✅ Settings imported successfully")
        
        # Test Settings validation
        validation = Settings.validate_config()
        print(f"✅ Settings validation: {validation}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_database_creation():
    """Test database creation."""
    print("\n🗄️ Testing database creation...")
    
    try:
        from data.database.nba import NBADatabase
        
        # Create test database
        test_db = NBADatabase()
        print("✅ NBA Database created successfully")
        
        # Test getting summary (should work even with empty DB)
        summary = test_db.get_data_summary()
        print(f"✅ Database summary: {summary}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def test_api_client():
    """Test API client creation."""
    print("\n🌐 Testing API client...")
    
    try:
        from api_clients.sports_api import SportsAPIClient
        
        # Create test client
        client = SportsAPIClient('nba')
        print("✅ NBA API client created successfully")
        
        # Test connection (this might fail without API key, but class should load)
        print("✅ API client instantiation works")
        
        return True
        
    except Exception as e:
        print(f"❌ API client error: {e}")
        return False

def test_model_creation():
    """Test model creation."""
    print("\n🤖 Testing model creation...")
    
    try:
        from models.nba.nba_model import NBAPredictionModel
        
        # Create test model
        model = NBAPredictionModel('game_winner')
        print("✅ NBA model created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering."""
    print("\n🔧 Testing feature engineering...")
    
    try:
        from data.features.nba_features import NBAFeatureEngineer
        
        # Create test feature engineer
        fe = NBAFeatureEngineer()
        print("✅ NBA Feature Engineer created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature engineering error: {e}")
        return False

def test_data_manager():
    """Test data manager creation."""
    print("\n📊 Testing data manager...")
    
    try:
        from data.manager import UniversalSportsDataManager
        
        # Create test manager (without Redis)
        manager = UniversalSportsDataManager(use_redis=False)
        print("✅ Data manager created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Data manager error: {e}")
        return False

def check_api_keys():
    """Check if API keys are configured."""
    print("\n🔑 Checking API key configuration...")
    
    try:
        from config.settings import Settings
        
        api_sports_key = Settings.API_SPORTS_KEY
        odds_api_key = Settings.ODDS_API_KEY
        
        if api_sports_key and api_sports_key != "insert key":
            print("✅ API Sports key is configured")
        else:
            print("⚠️ API Sports key not configured (update .env file)")
        
        if odds_api_key and odds_api_key != "insert key":
            print("✅ Odds API key is configured")
        else:
            print("⚠️ Odds API key not configured (update .env file)")
        
        return True
        
    except Exception as e:
        print(f"❌ API key check error: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🚀 Sports Prediction Platform - Setup Validation")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Database Test", test_database_creation),
        ("API Client Test", test_api_client),
        ("Model Test", test_model_creation),
        ("Feature Engineering Test", test_feature_engineering),
        ("Data Manager Test", test_data_manager),
        ("API Keys Check", check_api_keys),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready to go!")
    elif passed >= total - 2:  # Allow for API key warnings
        print("✅ Setup is mostly ready. Check API key configuration.")
    else:
        print("⚠️ Some issues found. Check the errors above.")
    
    # Next steps
    print("\n🚀 NEXT STEPS:")
    if passed >= total - 2:
        print("1. Configure API keys in .env file")
        print("2. Try running: python examples/NFL/nfl_prediction_pipeline.py")
        print("3. Test data ingestion with: python -c \"from data.manager import UniversalSportsDataManager; print('Manager works!')\"")
    else:
        print("1. Fix the import/setup errors shown above")
        print("2. Re-run this validation script")
        print("3. Check that all files are in the correct locations")

if __name__ == "__main__":
    main()
