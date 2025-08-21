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
    
    # The Odds API URL
    ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
    
    # Weather API
    WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
    WEATHER_API_URL = "http://api.openweathermap.org/data/2.5"
    
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
    
    # ============= CENTRALIZED SPORT CONFIGURATIONS =============
    SPORT_CONFIGS = {
        'nba': {
            'api_sports': {
                'base_url': 'https://v1.basketball.api-sports.io',
                'api_type': 'basketball',
                'default_league_id': 12,
                'default_league_name': 'NBA',
                'season_format': 'hyphenated'  # "2023-2024" format
            },
            'season_start_month': 10,  # October
            'season_end_month': 4,     # April
            'games_per_season': 82,
            'playoff_teams': 16,
            'key_stats': ['PTS', 'REB', 'AST', 'FG%', 'FT%', '3P%', 'STL', 'BLK', 'TO'],
            'team_count': 30
        },
        'mlb': {
            'api_sports': {
                'base_url': 'https://v1.baseball.api-sports.io',
                'api_type': 'baseball',
                'default_league_id': 1,
                'default_league_name': 'MLB',
                'season_format': 'year'  # 2024 format
            },
            'season_start_month': 3,   # March
            'season_end_month': 10,    # October
            'games_per_season': 162,
            'playoff_teams': 12,
            'key_stats': ['BA', 'OBP', 'SLG', 'ERA', 'WHIP', 'K/9', 'HR', 'RBI', 'SB'],
            'team_count': 30
        },
        'nfl': {
            'api_sports': {
                'base_url': 'https://v1.american-football.api-sports.io',
                'api_type': 'american-football',
                'default_league_id': 1,
                'default_league_name': 'NFL',
                'season_format': 'year'  # 2024 format
            },
            'season_start_month': 9,   # September
            'season_end_month': 2,     # February
            'games_per_season': 17,
            'playoff_teams': 14,
            'key_stats': ['Pass_Yds', 'Rush_Yds', 'TD', 'INT', 'Sacks', 'Fumbles', 'Comp%'],
            'team_count': 32
        }
    }
    
    # ============= COMMON STATS MAPPING =============
    # Cross-sport stat name standardization for PlayerMapper
    COMMON_STATS = {
        'games_played': ['games', 'gp', 'g', 'games_played'],
        'minutes_played': ['minutes', 'min', 'mp', 'time_on_ice'],
        'points': ['points', 'pts', 'p', 'runs', 'goals'],
        'assists': ['assists', 'ast', 'a', 'assists'],
        'turnovers': ['turnovers', 'to', 'tov', 'turnovers'],
        'field_goals_made': ['fgm', 'fg_made', 'field_goals'],
        'field_goals_attempted': ['fga', 'fg_attempted', 'field_goal_attempts'],
        'free_throws_made': ['ftm', 'ft_made', 'free_throws'],
        'free_throws_attempted': ['fta', 'ft_attempted', 'free_throw_attempts'],
        'rebounds': ['rebounds', 'reb', 'r', 'total_rebounds'],
        'steals': ['steals', 'stl', 's'],
        'blocks': ['blocks', 'blk', 'b'],
        'fouls': ['fouls', 'pf', 'personal_fouls']
    }
    
    # ============= LEGACY API CONFIGURATION (for backward compatibility) =============
    # Keep these for any existing code that might reference them
    NBA_API_URL = SPORT_CONFIGS['nba']['api_sports']['base_url']
    MLB_API_URL = SPORT_CONFIGS['mlb']['api_sports']['base_url']
    NFL_API_URL = SPORT_CONFIGS['nfl']['api_sports']['base_url']
    
    NBA_HOST = NBA_API_URL.replace("https://", "")
    MLB_HOST = MLB_API_URL.replace("https://", "")
    NFL_HOST = NFL_API_URL.replace("https://", "")
    
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
            'requests_per_hour': 1000,
            'retry_delays': [1, 2, 4, 8, 16]  # Exponential backoff
        },
        'odds_api': {
            'requests_per_minute': 60,
            'requests_per_day': 500,
            'requests_per_hour': 500,
            'retry_delays': [2, 4, 8, 16, 32]
        },
        'api_sports': {
            'requests_per_minute': 100,
            'requests_per_day': 1000,
            'requests_per_hour': 1000,
            'retry_delays': [1, 2, 4, 8, 16]
        }
    }
    
    # ============= CACHING CONFIGURATION =============
    CACHE_DURATIONS = {
        'team_stats': 3600 * 24,      # 24 hours
        'player_stats': 3600 * 12,    # 12 hours  
        'game_schedule': 3600 * 6,    # 6 hours
        'odds': 300,                  # 5 minutes
        'live_games': 60,             # 1 minute
        'historical_data': 3600 * 24 * 7,  # 1 week
        'player_props': 600,          # 10 minutes
        'standings': 3600 * 24,       # 24 hours
        'injuries': 3600 * 6          # 6 hours
    }
    
    # ============= MODEL STORAGE PATHS =============
    MODEL_BASE_DIR = Path("models")
    LOG_DIR = Path("logs")
    
    MODEL_PATHS = {
        'nba': {
            'game_winner': MODEL_BASE_DIR / "nba" / "game_winner_model.joblib",
            'player_points': MODEL_BASE_DIR / "nba" / "player_points_model.joblib",
            'player_assists': MODEL_BASE_DIR / "nba" / "player_assists_model.joblib",
            'player_rebounds': MODEL_BASE_DIR / "nba" / "player_rebounds_model.joblib",
            'player_props': MODEL_BASE_DIR / "nba" / "player_props_model.joblib"
        },
        'mlb': {
            'game_winner': MODEL_BASE_DIR / "mlb" / "game_winner_model.joblib",
            'nrfi': MODEL_BASE_DIR / "mlb" / "nrfi_model.joblib",
            'total_runs': MODEL_BASE_DIR / "mlb" / "total_runs_model.joblib",
            'player_props': MODEL_BASE_DIR / "mlb" / "player_props_model.joblib"
        },
        'nfl': {
            'game_winner': MODEL_BASE_DIR / "nfl" / "game_winner_model.joblib",
            'total_points': MODEL_BASE_DIR / "nfl" / "total_points_model.joblib",
            'qb_touchdowns': MODEL_BASE_DIR / "nfl" / "qb_touchdowns_model.joblib",
            'player_props': MODEL_BASE_DIR / "nfl" / "player_props_model.joblib"
        }
    }
    
    # ============= LOGGING CONFIGURATION =============
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    
    # ============= VALIDATION THRESHOLDS =============
    VALIDATION_THRESHOLDS = {
        'min_accuracy': 0.52,         # Minimum model accuracy
        'max_feature_correlation': 0.95,  # Max correlation between features
        'min_data_points': {
            'nba': 1000,
            'mlb': 2000, 
            'nfl': 500
        },
        'min_games_for_prediction': {
            'nba': 10,
            'mlb': 15,
            'nfl': 4
        }
    }
    
    # ============= PREDICTION CONFIDENCE LEVELS =============
    CONFIDENCE_LEVELS = {
        'high': 0.75,
        'medium': 0.60,
        'low': 0.55
    }
    
    # ============= BETTING ANALYSIS CONFIGURATION =============
    BETTING_CONFIG = {
        'min_edge': 0.02,              # Minimum edge for bet recommendation
        'max_bet_percentage': 0.05,    # Max % of bankroll per bet
        'kelly_fraction': 0.25,        # Kelly criterion fraction
        'min_odds': -300,              # Minimum odds to consider
        'max_odds': 500,               # Maximum odds to consider
        'confidence_multiplier': {
            'high': 1.0,
            'medium': 0.7,
            'low': 0.4
        }
    }
    
    # ============= ODDS API SPORT MAPPINGS =============
    ODDS_API_SPORT_KEYS = {
        'nba': 'basketball_nba',
        'mlb': 'baseball_mlb',
        'nfl': 'americanfootball_nfl'
    }
    
    # ============= DATA REFRESH SCHEDULES =============
    REFRESH_SCHEDULES = {
        'game_data': '*/15 * * * *',      # Every 15 minutes
        'player_stats': '0 */6 * * *',    # Every 6 hours
        'team_stats': '0 8 * * *',        # Daily at 8 AM
        'standings': '0 9 * * *',         # Daily at 9 AM
        'odds': '*/5 * * * *',            # Every 5 minutes
        'player_props': '*/10 * * * *',   # Every 10 minutes
        'models': '0 4 * * 0'             # Weekly on Sunday at 4 AM
    }
    
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
    def get_api_url(cls, sport: str) -> str:
        """Get API URL for a specific sport."""
        sport_lower = sport.lower()
        if sport_lower in cls.SPORT_CONFIGS:
            return cls.SPORT_CONFIGS[sport_lower]['api_sports']['base_url']
        return cls.NBA_API_URL  # fallback
    
    @classmethod
    def get_api_config(cls, sport: str) -> Dict[str, Any]:
        """Get complete API configuration for a specific sport."""
        sport_lower = sport.lower()
        if sport_lower in cls.SPORT_CONFIGS:
            return cls.SPORT_CONFIGS[sport_lower]['api_sports']
        return cls.SPORT_CONFIGS['nba']['api_sports']  # fallback
    
    @classmethod
    def get_sport_config(cls, sport: str) -> Dict[str, Any]:
        """Get complete sport configuration."""
        sport_lower = sport.lower()
        return cls.SPORT_CONFIGS.get(sport_lower, cls.SPORT_CONFIGS['nba'])
    
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
    def get_odds_sport_key(cls, sport: str) -> str:
        """Get odds API sport key for a specific sport."""
        return cls.ODDS_API_SPORT_KEYS.get(sport.lower(), 'basketball_nba')
    
    @classmethod
    def get_current_season(cls, sport: str) -> int:
        """Get current season year for a sport based on calendar."""
        from datetime import datetime
        now = datetime.now()
        sport_config = cls.get_sport_config(sport)
        
        if sport_config['season_start_month'] > sport_config['season_end_month']:
            # Season crosses calendar year (NBA, NFL)
            if now.month >= sport_config['season_start_month']:
                return now.year
            else:
                return now.year - 1
        else:
            # Season within calendar year (MLB)
            return now.year
    
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
            'api_sports_key': bool(cls.API_SPORTS_KEY),
            'odds_api_key': bool(cls.ODDS_API_KEY),
            'directories_exist': True,
            'sport_configs_valid': True
        }
        
        try:
            cls.ensure_directories()
        except Exception:
            validation_results['directories_exist'] = False
        
        # Validate sport configurations
        required_sport_keys = ['api_sports', 'season_start_month', 'season_end_month', 'key_stats', 'team_count']
        for sport, config in cls.SPORT_CONFIGS.items():
            for key in required_sport_keys:
                if key not in config:
                    validation_results['sport_configs_valid'] = False
                    break
        
        return validation_results
    
    @classmethod
    def get_connection_string(cls, use_sqlite: bool = True) -> str:
        """Get database connection string."""
        if use_sqlite:
            return f"sqlite:///{cls.NBA_DB_PATH}"  # Default to NBA for now
        else:
            return f"postgresql://{cls.DB_USER}:{cls.DB_PASS}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"
    
    @classmethod
    def get_api_headers(cls, api_type: str = 'api_sports') -> Dict[str, str]:
        """Get API headers for different API types."""
        if api_type == 'api_sports':
            return {
                'X-API-Key': cls.API_SPORTS_KEY,
                'Content-Type': 'application/json'
            }
        elif api_type == 'odds_api':
            return {
                'Content-Type': 'application/json'
            }
        else:
            return {'Content-Type': 'application/json'}
    
    @classmethod
    def is_season_active(cls, sport: str) -> bool:
        """Check if the current date is within the sport's active season."""
        from datetime import datetime
        now = datetime.now()
        sport_config = cls.get_sport_config(sport)
        
        start_month = sport_config['season_start_month']
        end_month = sport_config['season_end_month']
        
        if start_month > end_month:
            # Season crosses calendar year
            return now.month >= start_month or now.month <= end_month
        else:
            # Season within calendar year
            return start_month <= now.month <= end_month

# Initialize directories on import
Settings.ensure_directories()
