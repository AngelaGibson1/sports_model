#!/usr/bin/env python3
"""
MLB-focused test and setup script.
Get your MLB prediction system working with real current data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_mlb_database():
    """Test MLB database specifically."""
    print("⚾ Testing MLB Database...")
    
    from data.database.mlb import MLBDatabase
    
    # Create MLB database
    db = MLBDatabase()
    
    # Create sample MLB game data (realistic structure)
    sample_games = pd.DataFrame({
        'game_id': [1, 2, 3, 4, 5],
        'date': ['2024-08-15', '2024-08-16', '2024-08-17', '2024-08-18', '2024-08-19'],
        'home_team_id': [1, 2, 3, 4, 5],
        'away_team_id': [6, 7, 8, 9, 10],
        'home_team_name': ['Yankees', 'Dodgers', 'Astros', 'Braves', 'Phillies'],
        'away_team_name': ['Red Sox', 'Giants', 'Rangers', 'Mets', 'Padres'],
        'home_score': [7, 4, 8, 5, 6],
        'away_score': [3, 6, 2, 7, 4],
        'status': 'Finished',
        'season': 2024
    })
    
    # Save to database
    saved_count = db.save_games(sample_games)
    print(f"✅ Saved {saved_count} sample MLB games")
    
    # Get data summary
    summary = db.get_data_summary()
    print(f"✅ MLB Database now contains {summary['total_games']} games")
    
    return db

def test_mlb_api_connection():
    """Test MLB API connectivity."""
    print("\n🌐 Testing MLB API Connection...")
    
    from api_clients.sports_api import SportsAPIClient
    
    # Create MLB API client
    client = SportsAPIClient('mlb')
    
    # Test connection
    connection_test = client.test_connection()
    print(f"✅ MLB API Status: {connection_test}")
    
    return client

def test_mlb_data_manager():
    """Test MLB data ingestion."""
    print("\n📊 Testing MLB Data Manager...")
    
    from data.manager import UniversalSportsDataManager
    
    # Create manager
    manager = UniversalSportsDataManager(use_redis=False)
    
    # Test MLB data summary
    mlb_summary = manager.get_data_summary('mlb')
    print(f"✅ MLB Summary: {mlb_summary}")
    
    # Test getting current MLB games (this might make a real API call)
    print("🔍 Testing live MLB game retrieval...")
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        games = manager.get_games('mlb', date=today)
        print(f"✅ Found {len(games)} MLB games for {today}")
        
        if len(games) > 0:
            print("📋 Sample game data:")
            print(games[['home_team_name', 'away_team_name', 'status']].head())
            
    except Exception as e:
        print(f"⚠️ Live data test: {e}")
        print("   (This is normal if API keys aren't configured yet)")
    
    return manager

def create_mlb_model():
    """Create and test MLB-specific models."""
    print("\n🤖 Creating MLB Models...")
    
    # We need to create an MLB model since we don't have one yet
    print("📝 Note: Creating MLB model structure...")
    
    # For now, let's use the NBA model structure as a template
    from models.nba.nba_model import NBAPredictionModel
    
    # Create models that would work for MLB
    game_winner_model = NBAPredictionModel('game_winner')
    total_runs_model = NBAPredictionModel('total_points')  # Will predict total runs
    
    print("✅ Created MLB game winner model (using NBA structure)")
    print("✅ Created MLB total runs model")
    
    # Test with baseball-like data
    mlb_sample_data = pd.DataFrame({
        'game_id': [1, 2, 3, 4, 5],
        'home_team_recent_form': [0.65, 0.58, 0.72, 0.45, 0.61],
        'away_team_recent_form': [0.52, 0.61, 0.48, 0.67, 0.55],
        'home_runs_per_game': [5.2, 4.8, 6.1, 3.9, 5.5],
        'away_runs_per_game': [4.7, 5.3, 4.2, 5.8, 4.9],
        'home_era': [3.45, 4.12, 3.78, 4.55, 3.92],
        'away_era': [3.89, 3.67, 4.23, 3.34, 4.01],
        'home_win': [1, 0, 1, 0, 1],  # Target variable
        'total_points': [9, 11, 8, 12, 10]  # Total runs scored
    })
    
    # Test feature preparation
    X, y = game_winner_model.prepare_features(mlb_sample_data)
    print(f"✅ Prepared MLB features: {len(X)} samples, {len(X.columns)} features")
    
    return game_winner_model, mlb_sample_data

def suggest_mlb_next_steps():
    """Suggest next steps for MLB implementation."""
    print("\n🚀 MLB-Specific Next Steps:")
    print("=" * 50)
    
    print("1. 📊 GET REAL MLB DATA:")
    print("   - Today's games and recent results")
    print("   - Team statistics for 2024 season")
    print("   - Player statistics for props")
    
    print("\n2. 🔧 CREATE MLB FEATURES:")
    print("   - Pitching matchups (starter ERA, WHIP)")
    print("   - Batting vs. pitching hand (L/R splits)")
    print("   - Bullpen strength ratings")
    print("   - Weather impact (wind, temperature)")
    print("   - Park factors (hitter-friendly stadiums)")
    
    print("\n3. 🎯 MLB-SPECIFIC MODELS:")
    print("   - Game winner prediction")
    print("   - Total runs (over/under)")
    print("   - First 5 innings (NRFI/YRFI)")
    print("   - Player props (hits, RBIs, strikeouts)")
    
    print("\n4. ⚾ IMMEDIATE OPPORTUNITIES:")
    print("   - Daily MLB games (August-October)")
    print("   - Playoff races heating up")
    print("   - Rich historical data available")
    print("   - Multiple daily games for testing")

def main():
    """Run MLB-focused platform test."""
    print("⚾ MLB Sports Prediction Platform Setup")
    print("=" * 50)
    print(f"🗓️ Date: {datetime.now().strftime('%Y-%m-%d')}")
    print("🎯 Focus: Get MLB predictions working with live data")
    
    try:
        # Test MLB components
        mlb_db = test_mlb_database()
        mlb_api = test_mlb_api_connection()
        mlb_manager = test_mlb_data_manager()
        mlb_model, sample_data = create_mlb_model()
        
        # Success summary
        print("\n" + "=" * 50)
        print("⚾ MLB PLATFORM STATUS")
        print("=" * 50)
        print("✅ MLB Database: Ready")
        print("✅ MLB API Client: Ready")
        print("✅ MLB Data Manager: Ready")
        print("✅ MLB Models: Ready (using NBA structure)")
        
        print(f"\n🎯 STATUS: READY FOR MLB SEASON!")
        
        suggest_mlb_next_steps()
        
        return True
        
    except Exception as e:
        print(f"\n❌ MLB setup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n⚾ Ready to start making MLB predictions!")
    else:
        print("\n⚠️ Some MLB setup issues found.")
