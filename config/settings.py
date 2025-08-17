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
    # Direct API-Sports Key (better than RapidAPI!)
    API_SPORTS_KEY = os.getenv("API_SPORTS_KEY")
    
    # The Odds API Key (direct)
    ODDS_API_KEY = os.getenv("ODDS_API_KEY")
    
    # API-Sports Base URLs (direct access)
    API_SPORTS_BASE_URL = "https://v3.football.api-sports.io"  # Main endpoint
    NBA_API_URL = "https://v2.nba.api-sports.io"
    MLB_API_URL = "https://v1.baseball.api-sports.io"
    NFL_API_URL = "https://v1.american-football.api-sports.io"
    
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
    
    # ============= MODEL PARAMETERS =============
    NBA_MODEL_PARAMS = {
        'n_estimators': 300,
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'early_stopping_rounds': 20,
        'tree_method': 'hist'  # Faster training
    }
    
    MLB_MODEL_PARAMS = {
        'n_estimators': 250,
        'max_depth': 6,
        'learning_rate': 0.12,
        'subsample': 0.85,
        'colsample_bytree': 0.9,
        'random_state': 42,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'early_stopping_rounds': 15,
        'tree_method': 'hist'
    }
    
    NFL_MODEL_PARAMS = {
        'n_estimators': 200,
        'max_depth': 7,
        'learning_rate': 0.15,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'early_stopping_rounds': 10,
        'tree_method': 'hist'
    }
    
    # ============= FEATURE ENGINEERING PARAMETERS =============
    # Rolling window sizes for different sports
    ROLLING_WINDOWS = {
        'nba': {'short': 5, 'medium': 10, 'long': 20},
        'mlb': {'short': 3, 'medium': 7, 'long': 15}, 
        'nfl': {'short': 2, 'medium': 4, 'long': 8}
    }
    
    # Minimum games required for rolling calculations
    MIN_GAMES_FOR_ROLLING = {
        'nba': 10,
        'mlb': 15,
        'nfl': 4
    }
    
    # ============= API RATE LIMITING =============
    API_RATE_LIMITS = {
        'rapidapi': {
            'requests_per_minute': 100,
            'requests_per_day': 1000,
            'retry_delays': [1, 2, 4, 8, 16]  # Exponential backoff
        },
        'odds_api': {
            'requests_per_minute': 60,
            'requests_per_day': 500,
            'retry_delays': [2, 4, 8, 16, 32]
        },
        'pybaseball': {
            'requests_per_minute': 30,  # Be conservative with MLB data
            'delay_between_requests': 2.0
        }
    }
    
    # ============= CACHING CONFIGURATION =============
    CACHE_DURATIONS = {
        'team_stats': 3600 * 24,      # 24 hours
        'player_stats': 3600 * 12,    # 12 hours  
        'game_schedule': 3600 * 6,    # 6 hours
        'odds': 300,                  # 5 minutes
        'live_games': 60,             # 1 minute
        'historical_data': 3600 * 24 * 7  # 1 week
    }
    
    # ============= MODEL STORAGE PATHS =============
    MODEL_BASE_DIR = Path("models")
    
    MODEL_PATHS = {
        'nba': {
            'game_winner': MODEL_BASE_DIR / "nba" / "game_winner_model.joblib",
            'player_points': MODEL_BASE_DIR / "nba" / "player_points_model.joblib",
            'player_assists': MODEL_BASE_DIR / "nba" / "player_assists_model.joblib",
            'player_rebounds': MODEL_BASE_DIR / "nba" / "player_rebounds_model.joblib"
        },
        'mlb': {
            'game_winner': MODEL_BASE_DIR / "mlb" / "game_winner_model.joblib",
            'nrfi': MODEL_BASE_DIR / "mlb" / "nrfi_model.joblib",
            'total_runs': MODEL_BASE_DIR / "mlb" / "total_runs_model.joblib"
        },
        'nfl': {
            'game_winner': MODEL_BASE_DIR / "nfl" / "game_winner_model.joblib",
            'total_points': MODEL_BASE_DIR / "nfl" / "total_points_model.joblib",
            'qb_touchdowns': MODEL_BASE_DIR / "nfl" / "qb_touchdowns_model.joblib"
        }
    }
    
    # ============= LOGGING CONFIGURATION =============
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    LOG_DIR = Path("logs")
    
    # ============= VALIDATION THRESHOLDS =============
    VALIDATION_THRESHOLDS = {
        'min_accuracy': 0.52,         # Minimum model accuracy
        'max_feature_correlation': 0.95,  # Max correlation between features
        'min_data_points': {
            'nba': 1000,
            'mlb': 2000, 
            'nfl': 500
        }
    }
    
    # ============= PREDICTION CONFIDENCE LEVELS =============
    CONFIDENCE_LEVELS = {
        'high': 0.75,
        'medium': 0.60,
        'low': 0.55
    }
    
    # ============= SPORT-SPECIFIC CONFIGURATIONS =============
    SPORT_CONFIGS = {
        'nba': {
            'season_start_month': 10,
            'season_end_month': 4,
            'games_per_season': 82,
            'playoff_teams': 16,
            'key_stats': ['PTS', 'REB', 'AST', 'FG%', 'FT%', '3P%'],
            'team_count': 30
        },
        'mlb': {
            'season_start_month': 3,
            'season_end_month': 10,
            'games_per_season': 162,
            'playoff_teams': 12,
            'key_stats': ['BA', 'OBP', 'SLG', 'ERA', 'WHIP', 'K/9'],
            'team_count': 30
        },
        'nfl': {
            'season_start_month': 9,
            'season_end_month': 2,
            'games_per_season': 17,
            'playoff_teams': 14,
            'key_stats': ['Pass_Yds', 'Rush_Yds', 'TD', 'INT', 'Sacks'],
            'team_count': 32
        }
    }
    
    # ============= EXTERNAL API CONFIGURATIONS =============
    WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
    WEATHER_API_URL = "http://api.openweathermap.org/data/2.5"
    
    # ============= HELPER METHODS =============
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
    def get_api_host(cls, sport: str) -> str:
        """Get API host for a specific sport."""
        host_map = {
            'nba': cls.NBA_HOST,
            'mlb': cls.MLB_HOST,
            'nfl': cls.NFL_HOST
        }
        return host_map.get(sport.lower(), cls.NBA_HOST)
    
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
    def ensure_directories(cls):
        """Create necessary directories if they don't exist."""
        directories = [
            cls.MODEL_BASE_DIR,
            cls.SQLITE_DB_DIR,
            cls.LOG_DIR,
            *[path.parent for paths in cls.MODEL_PATHS.values() for path in paths.values()]
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_config(cls) -> Dict[str, bool]:
        """Validate that required configuration is present."""
        validation_results = {
            'rapidapi_key': bool(cls.RAPIDAPI_KEY),
            'odds_api_key': bool(cls.ODDS_API_KEY),
            'directories_exist': True
        }
        
        try:
            cls.ensure_directories()
        except Exception:
            validation_results['directories_exist'] = False
        
        return validation_results
    
    @classmethod
    def get_connection_string(cls, use_sqlite: bool = True) -> str:
        """Get database connection string."""
        if use_sqlite:
            return f"sqlite:///{cls.NBA_DB_PATH}"  # Default to NBA for now
        else:
            return f"postgresql://{cls.DB_USER}:{cls.DB_PASS}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"

# Initialize directories on import
Settings.ensure_directories()
