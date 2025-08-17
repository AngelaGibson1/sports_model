# Fixed config/settings.py - Key corrections

import os
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

class Settings:
    """Comprehensive configuration settings for sports prediction platform."""
    
    # ============= API CONFIGURATION =============
    # Direct API-Sports Key
    API_SPORTS_KEY = os.getenv("API_SPORTS_KEY")
    
    # The Odds API Key (direct)
    ODDS_API_KEY = os.getenv("ODDS_API_KEY")
    
    # API-Sports Base URLs (direct access) - FIXED
    NBA_API_URL = "https://v2.nba.api-sports.io"
    MLB_API_URL = "https://v1.baseball.api-sports.io"
    NFL_API_URL = "https://v1.american-football.api-sports.io"
    
    # FIXED: Add these missing properties referenced in other files
    NBA_HOST = NBA_API_URL.replace("https://", "")
    MLB_HOST = MLB_API_URL.replace("https://", "")
    NFL_HOST = NFL_API_URL.replace("https://", "")
    
    # The Odds API URL
    ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
    
    # ============= DATABASE CONFIGURATION =============
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_USER = os.getenv("DB_USER", "sports_user")
    DB_PASS = os.getenv("DB_PASS", "password")
    DB_NAME = os.getenv("DB_NAME", "sports_prediction")
    DB_PORT = int(os.getenv("DB_PORT", "5432"))
    
    # SQLite paths for local development
    SQLITE_DB_DIR = Path("data/database/sqlite")
    NBA_DB_PATH = SQLITE_DB_DIR / "nba_data.db"
    MLB_DB_PATH = SQLITE_DB_DIR / "mlb_data.db"
    NFL_DB_PATH = SQLITE_DB_DIR / "nfl_data.db"
    
    # ============= REDIS CONFIGURATION =============
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
    
    # ... rest of your existing configuration ...
    
    # ============= HELPER METHODS - FIXED =============
    @classmethod
    def get_model_params(cls, sport: str) -> Dict[str, Any]:
        """Get model parameters for a specific sport."""
        param_map = {
            'nba': cls.NBA_MODEL_PARAMS,
            'mlb': cls.MLB_MODEL_PARAMS,
            'nfl': cls.NFL_MODEL_PARAMS
        }
        return param_map.get(sport.lower(), cls.NBA_MODEL_PARAMS)
    
    @classmethod
    def get_api_url(cls, sport: str) -> str:  # FIXED: was get_api_host
        """Get API URL for a specific sport."""
        url_map = {
            'nba': cls.NBA_API_URL,
            'mlb': cls.MLB_API_URL,
            'nfl': cls.NFL_API_URL
        }
        return url_map.get(sport.lower(), cls.NBA_API_URL)
    
    @classmethod
    def get_db_path(cls, sport: str) -> Path:
        """Get database path for a specific sport."""
        path_map = {
            'nba': cls.NBA_DB_PATH,
            'mlb': cls.MLB_DB_PATH,
            'nfl': cls.NFL_DB_PATH
        }
        return path_map.get(sport.lower(), cls.NBA_DB_PATH)
    
    @classmethod
    def validate_config(cls) -> Dict[str, bool]:
        """Validate that required configuration is present."""
        validation_results = {
            'api_sports_key': bool(cls.API_SPORTS_KEY),  # FIXED
            'odds_api_key': bool(cls.ODDS_API_KEY),
            'directories_exist': True
        }
        
        try:
            cls.ensure_directories()
        except Exception:
            validation_results['directories_exist'] = False
        
        return validation_results

# Initialize directories on import
Settings.ensure_directories()
