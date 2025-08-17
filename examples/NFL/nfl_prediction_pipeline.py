#!/usr/bin/env python3
"""
Quick test script to verify NFL model works
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Test if the model can be imported and initialized
def test_nfl_model():
    print("🏈 Testing NFL Model...")
    
    try:
        # Import the model
        from models.nfl.nfl_model import NFLPredictionModel, NFLModelEnsemble
        print("✅ Import successful")
        
        # Initialize models
        game_model = NFLPredictionModel('game_winner')
        total_points_model = NFLPredictionModel('total_points')
        qb_model = NFLPredictionModel('qb_passing_yards')
        
        print("✅ Model initialization successful")
        
        # Test ensemble
        ensemble = NFLModelEnsemble(['game_winner', 'total_points'])
        print("✅ Ensemble initialization successful")
        
        # Create dummy data to test feature preparation
        dummy_data = pd.DataFrame({
            'game_id': [1, 2, 3],
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'week': [1, 2, 3],
            'season': [2024, 2024, 2024],
            'home_team_id': [1, 2, 3],
            'away_team_id': [4, 5, 6],
            'home_team_name': ['Patriots', 'Cowboys', 'Packers'],
            'away_team_name': ['Bills', 'Giants', 'Bears'],
            'home_score': [21, 28, 14],
            'away_score': [17, 21, 10],
            'home_win': [1, 1, 1],
            'total_points': [38, 49, 24],
            'passing_yards': [250, 300, 180],
            # Add some feature columns
            'home_form_4': [0.75, 0.5, 0.25],
            'away_form_4': [0.25, 0.75, 0.5],
            'rest_advantage': [0, 1, -1],
            'divisional_game': [0, 1, 0],
            'conference_game': [1, 1, 0]
        })
        
        # Test feature preparation
        X, y = game_model.prepare_features(dummy_data)
        print(f"✅ Feature preparation successful: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Test if we can create a small model (without actual training)
        if len(X) > 0:
            print("✅ Feature preparation produced valid data")
        else:
            print("⚠️ Feature preparation produced no valid data")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure the file is at models/nfl/nfl_model.py")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_settings_integration():
    print("\n⚙️ Testing Settings integration...")
    
    try:
        from config.settings import Settings
        
        # Test NFL model parameters
        nfl_params = Settings.get_model_params('nfl')
        print(f"✅ NFL model parameters loaded: {len(nfl_params)} params")
        
        # Test NFL model paths
        nfl_paths = Settings.MODEL_PATHS.get('nfl', {})
        print(f"✅ NFL model paths configured: {len(nfl_paths)} model types")
        
        # Test NFL database path
        nfl_db_path = Settings.get_db_path('nfl')
        print(f"✅ NFL database path: {nfl_db_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Settings error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 NFL Model Compatibility Test")
    print("=" * 40)
    
    # Test model functionality
    model_test = test_nfl_model()
    
    # Test settings integration
    settings_test = test_settings_integration()
    
    # Overall result
    print("\n📊 Test Results:")
    print(f"   Model functionality: {'✅ PASS' if model_test else '❌ FAIL'}")
    print(f"   Settings integration: {'✅ PASS' if settings_test else '❌ FAIL'}")
    
    if model_test and settings_test:
        print("\n🎉 All tests passed! NFL model is ready to use.")
    else:
        print("\n⚠️ Some tests failed. Check the issues above.")
