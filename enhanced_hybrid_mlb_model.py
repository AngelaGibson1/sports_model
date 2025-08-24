#!/usr/bin/env python3
"""
Enhanced Hybrid MLB Model - Complete Player Stats Integration
- Training: API Sports historical data (your paid investment)
- Current Stats: PyBaseball real-time team & player performance (free)
- Pitcher Rotation: Track starting pitchers and their stats
- Hitting Matchups: Opponent hitting vs pitcher types
- Runs Prediction: Pitcher vs lineup matchup analysis
- Name Dictionary: CSV files for team/player validation 
- Live Games & Odds: Your paid Odds API (keep using!)
- Game Time: Handle doubleheaders properly
- Learning: Track predictions vs actual outcomes
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, Tuple, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')

# Timezone handling
from zoneinfo import ZoneInfo  # stdlib in Python 3.9+

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, mean_absolute_error

# Advanced models
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Import logger first
from loguru import logger

# Data sources
try:
    import pybaseball as pyb
    pyb.cache.enable()
    PYBASEBALL_AVAILABLE = True
    logger.info("✅ PyBaseball available for current team & player stats")
except ImportError:
    PYBASEBALL_AVAILABLE = False
    logger.warning("⚠️ PyBaseball not available - install with: pip install pybaseball")

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Your existing components
try:
    from data.database.mlb import MLBDatabase
    from api_clients.odds_api import OddsAPIClient
    from data.player_mapping import EnhancedPlayerMapper
    COMPONENTS_AVAILABLE = True
    logger.info("✅ Your paid APIs available")
except ImportError:
    COMPONENTS_AVAILABLE = False
    logger.error("❌ Your paid APIs not available")
    MLBDatabase = None
    OddsAPIClient = None
    EnhancedPlayerMapper = None

# Team mapping for pitcher rotation tracking
CANON_TEAM_MAP = {
    # AL East
    "baltimore orioles": 4, "orioles": 4, "bal": 4,
    "boston red sox": 3, "red sox": 3, "bos": 3,
    "new york yankees": 1, "yankees": 1, "nyy": 1,
    "tampa bay rays": 6, "rays": 6, "tbr": 6, "tb": 6,
    "toronto blue jays": 5, "blue jays": 5, "tor": 5,
    # AL Central
    "chicago white sox": 7, "white sox": 7, "cws": 7, "chw": 7,
    "cleveland guardians": 8, "guardians": 8, "cleveland indians": 8, "indians": 8, "cle": 8,
    "detroit tigers": 9, "tigers": 9, "det": 9,
    "kansas city royals": 10, "royals": 10, "kcr": 10, "kc": 10,
    "minnesota twins": 11, "twins": 11, "min": 11,
    # AL West
    "houston astros": 12, "astros": 12, "hou": 12,
    "los angeles angels": 13, "la angels": 13, "angels": 13, "ana": 13, "laa": 13,
    "oakland athletics": 14, "athletics": 14, "oak": 14, "ath": 14, "as": 14,
    "seattle mariners": 15, "mariners": 15, "sea": 15,
    "texas rangers": 16, "rangers": 16, "tex": 16,
    # NL East
    "atlanta braves": 17, "braves": 17, "atl": 17,
    "miami marlins": 18, "marlins": 18, "mia": 18, "fla": 18,
    "new york mets": 2, "mets": 2, "nym": 2,
    "philadelphia phillies": 19, "phillies": 19, "phi": 19,
    "washington nationals": 20, "nationals": 20, "wsh": 20, "wsn": 20,
    # NL Central
    "chicago cubs": 21, "cubs": 21, "chc": 21,
    "cincinnati reds": 22, "reds": 22, "cin": 22,
    "milwaukee brewers": 23, "brewers": 23, "mil": 23,
    "pittsburgh pirates": 24, "pirates": 24, "pit": 24,
    "st louis cardinals": 25, "st. louis cardinals": 25, "cardinals": 25, "stl": 25,
    # NL West
    "arizona diamondbacks": 26, "diamondbacks": 26, "dbacks": 26, "ari": 26, "az": 26,
    "colorado rockies": 27, "rockies": 27, "col": 27,
    "los angeles dodgers": 28, "la dodgers": 28, "dodgers": 28, "lad": 28, "la": 28,
    "san diego padres": 29, "padres": 29, "sd": 29, "sdp": 29,
    "san francisco giants": 30, "giants": 30, "sf": 30, "sfg": 30,
}

# PyBaseball team abbreviation mapping
PYBASEBALL_TEAM_MAP = {
    'NYY': 1, 'NYM': 2, 'BOS': 3, 'BAL': 4, 'TOR': 5, 'TB': 6, 'CWS': 7, 'CLE': 8, 
    'DET': 9, 'KC': 10, 'MIN': 11, 'HOU': 12, 'LAA': 13, 'OAK': 14, 'SEA': 15, 'TEX': 16,
    'ATL': 17, 'MIA': 18, 'PHI': 19, 'WSH': 20, 'CHC': 21, 'CIN': 22, 'MIL': 23, 'PIT': 24,
    'STL': 25, 'ARI': 26, 'COL': 27, 'LAD': 28, 'SD': 29, 'SF': 30
}


class EnhancedHybridMLBModel:
    """
    Enhanced Hybrid MLB Model with complete player stats integration including pitcher rotation.
    """
    
    def __init__(self, model_dir: Path = Path('models/mlb'), random_state: int = 42,
                 local_tz: str = "America/New_York"):
        """Initialize enhanced hybrid model with player stats integration."""
        logger.info("🚀 Initializing Enhanced Hybrid MLB Model...")
        logger.info("📊 Complete Player Stats Architecture:")
        logger.info("   🏟️ Training: API Sports historical data (your paid)")
        logger.info("   📈 Team Stats: PyBaseball real-time (free)")
        logger.info("   ⚾ Player Stats: PyBaseball individual performance")
        logger.info("   🎯 Pitcher Rotation: Starting pitcher prediction")
        logger.info("   💪 Hitting Matchups: Lineup vs pitcher analysis")
        logger.info("   📋 Name Validation: CSV files dictionary")
        logger.info("   💰 Live Games & Odds: Your paid Odds API")
        logger.info("   🏃 Runs Prediction: Enhanced with player matchups")
        
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        
        # Timezone for game time features and display
        self.local_tz = ZoneInfo(local_tz)
        
        # YOUR PAID DATA SOURCES (Priority #1)
        self.db = MLBDatabase() if COMPONENTS_AVAILABLE else None
        self.odds_api = OddsAPIClient() if COMPONENTS_AVAILABLE else None
        self.player_mapper = EnhancedPlayerMapper(sport='mlb', auto_build=True) if COMPONENTS_AVAILABLE else None
        
        # Model components for game outcome
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        
        # Model components for runs prediction
        self.runs_scaler = StandardScaler()
        self.runs_model = None
        self.runs_feature_names = None
        
        # Robust categorical encoding
        self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.categorical_cols = ['home_team_id', 'away_team_id']
        self.cat_encoder_fitted = False
        
        # Training configuration
        self.test_days = 45
        self.training_seasons = [2021, 2022, 2023, 2024, 2025]
        self.current_season = datetime.now().year
        
        # Player stats cache
        self.current_pitcher_stats = pd.DataFrame()
        self.current_batting_stats = pd.DataFrame()
        self.pitcher_rotations = {}
        
        # Learning system
        self.predictions_log = self.model_dir / 'predictions_log.json'
        self.performance_history = []
        
        # Model configurations
        self.model_configs = {
            'xgboost': {
                'model_class': xgb.XGBClassifier,
                'params': {'n_estimators': 500, 'max_depth': 8, 'learning_rate': 0.08, 
                           'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': random_state,
                           'eval_metric': 'logloss', 'use_label_encoder': False}
            },
            'lightgbm': {
                'model_class': lgb.LGBMClassifier,
                'params': {'n_estimators': 500, 'num_leaves': 100, 'learning_rate': 0.08, 
                           'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'random_state': random_state,
                           'verbosity': -1}
            },
            'random_forest': {
                'model_class': RandomForestClassifier,
                'params': {'n_estimators': 500, 'max_depth': 15, 'min_samples_split': 5,
                           'max_features': 'sqrt', 'class_weight': 'balanced', 'random_state': random_state,
                           'n_jobs': -1}
            }
        }
        
        # Runs prediction model configurations
        self.runs_model_configs = {
            'runs_rf': RandomForestRegressor(n_estimators=300, max_depth=12, random_state=random_state, n_jobs=-1),
            'runs_xgb': xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=random_state)
        }
        
        # Team name standardization
        self.team_name_dict = {}
        self.team_id_mapping = {}
        self._load_csv_team_dictionary()

        logger.info("✅ Enhanced Hybrid MLB Model initialized")

    def _load_csv_team_dictionary(self):
        """Load CSV files as team name dictionary for validation."""
        logger.info("📋 Loading CSV team dictionary...")
        
        if self.player_mapper and hasattr(self.player_mapper, 'team_map'):
            try:
                team_df = self.player_mapper.team_map
                
                if not team_df.empty:
                    for _, row in team_df.iterrows():
                        team_id = row.get('team_id', 1)
                        
                        if 'name' in row:
                            self.team_name_dict[row['name'].lower()] = team_id
                        if 'abbreviation' in row:
                            self.team_name_dict[row['abbreviation'].lower()] = team_id
                        if 'city' in row and 'name' in row:
                            full_name = f"{row['city']} {row['name']}".lower()
                            self.team_name_dict[full_name] = team_id
                    
                    logger.info(f"✅ Loaded {len(self.team_name_dict)} team name mappings from CSV")
                else:
                    self._create_fallback_team_dict()
            except Exception as e:
                logger.error(f"❌ Failed to load CSV team dict: {e}")
                self._create_fallback_team_dict()
        else:
            self._create_fallback_team_dict()

    def _create_fallback_team_dict(self):
        """Create fallback team dictionary."""
        self.team_name_dict = CANON_TEAM_MAP.copy()

    def load_current_player_stats(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load current season pitcher and batting stats from PyBaseball."""
        logger.info("⚾ Loading current player stats from PyBaseball...")
        
        if not PYBASEBALL_AVAILABLE:
            logger.warning("⚠️ PyBaseball not available, using fallback player stats")
            return self._get_fallback_player_stats()
        
        try:
            current_season = self.current_season
            
            # Load pitcher stats
            logger.info(f"   🎯 Loading {current_season} pitcher stats...")
            pitcher_stats = pyb.pitching_stats(current_season, qual=1)  # Min 1 inning
            
            # Load batting stats  
            logger.info(f"   💪 Loading {current_season} batting stats...")
            batting_stats = pyb.batting_stats(current_season, qual=1)   # Min 1 PA
            
            # Process pitcher stats
            if not pitcher_stats.empty:
                pitcher_stats = self._process_pitcher_stats(pitcher_stats)
                logger.info(f"   ✅ Processed {len(pitcher_stats)} pitcher records")
            
            # Process batting stats
            if not batting_stats.empty:
                batting_stats = self._process_batting_stats(batting_stats)
                logger.info(f"   ✅ Processed {len(batting_stats)} batting records")
            
            # Cache for quick access - ensure they're DataFrames
            if isinstance(pitcher_stats, pd.DataFrame):
                self.current_pitcher_stats = pitcher_stats
            else:
                self.current_pitcher_stats = pd.DataFrame()
                
            if isinstance(batting_stats, pd.DataFrame):
                self.current_batting_stats = batting_stats
            else:
                self.current_batting_stats = pd.DataFrame()
            
            return self.current_pitcher_stats, self.current_batting_stats
            
        except Exception as e:
            logger.error(f"❌ Failed to load player stats: {e}")
            pitcher_fallback, batting_fallback = self._get_fallback_player_stats()
            
            # Ensure fallback returns DataFrames
            if isinstance(pitcher_fallback, pd.DataFrame):
                self.current_pitcher_stats = pitcher_fallback
            else:
                self.current_pitcher_stats = pd.DataFrame()
                
            if isinstance(batting_fallback, pd.DataFrame):
                self.current_batting_stats = batting_fallback  
            else:
                self.current_batting_stats = pd.DataFrame()
                
            return self.current_pitcher_stats, self.current_batting_stats

    def _process_pitcher_stats(self, pitcher_df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean pitcher statistics."""
        processed = pitcher_df.copy()
        
        # Map team abbreviations to team IDs
        processed['team_id'] = processed['Team'].map(PYBASEBALL_TEAM_MAP)
        processed = processed.dropna(subset=['team_id'])
        processed['team_id'] = processed['team_id'].astype(int)
        
        # Essential pitcher metrics
        required_cols = ['Name', 'Team', 'team_id', 'IP', 'ERA', 'WHIP', 'SO', 'BB', 'W', 'L', 'SV']
        available_cols = [col for col in required_cols if col in processed.columns]
        processed = processed[available_cols].copy()
        
        # Fill missing values
        processed['IP'] = pd.to_numeric(processed['IP'], errors='coerce').fillna(50.0)
        processed['ERA'] = pd.to_numeric(processed['ERA'], errors='coerce').fillna(4.50)
        processed['WHIP'] = pd.to_numeric(processed['WHIP'], errors='coerce').fillna(1.35)
        processed['SO'] = pd.to_numeric(processed['SO'], errors='coerce').fillna(50)
        processed['BB'] = pd.to_numeric(processed['BB'], errors='coerce').fillna(20)
        processed['W'] = pd.to_numeric(processed['W'], errors='coerce').fillna(8)
        processed['L'] = pd.to_numeric(processed['L'], errors='coerce').fillna(8)
        processed['SV'] = pd.to_numeric(processed['SV'], errors='coerce').fillna(0)
        
        # Calculate advanced metrics
        processed['K9'] = (processed['SO'] / processed['IP'] * 9).replace([np.inf, -np.inf], 8.5)
        processed['BB9'] = (processed['BB'] / processed['IP'] * 9).replace([np.inf, -np.inf], 3.5)
        processed['K_BB_ratio'] = (processed['SO'] / processed['BB']).replace([np.inf, -np.inf], 2.5)
        processed['win_pct'] = (processed['W'] / (processed['W'] + processed['L'])).fillna(0.5)
        
        # Determine pitcher role
        processed['is_starter'] = ((processed['IP'] / processed['W'].fillna(1).clip(lower=1)) > 5).astype(int)
        processed['is_closer'] = (processed['SV'] >= 10).astype(int)
        
        return processed

    def _process_batting_stats(self, batting_df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean batting statistics."""
        processed = batting_df.copy()
        
        # Map team abbreviations to team IDs
        processed['team_id'] = processed['Team'].map(PYBASEBALL_TEAM_MAP)
        processed = processed.dropna(subset=['team_id'])
        processed['team_id'] = processed['team_id'].astype(int)
        
        # Essential batting metrics
        required_cols = ['Name', 'Team', 'team_id', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'BB', 'SO', 'AVG', 'OBP', 'SLG']
        available_cols = [col for col in required_cols if col in processed.columns]
        processed = processed[available_cols].copy()
        
        # Fill missing values with league averages
        for col in ['G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'BB', 'SO']:
            processed[col] = pd.to_numeric(processed[col], errors='coerce').fillna(0)
        
        processed['AVG'] = pd.to_numeric(processed['AVG'], errors='coerce').fillna(0.250)
        processed['OBP'] = pd.to_numeric(processed['OBP'], errors='coerce').fillna(0.320)
        processed['SLG'] = pd.to_numeric(processed['SLG'], errors='coerce').fillna(0.400)
        
        # Calculate advanced metrics
        processed['OPS'] = processed['OBP'] + processed['SLG']
        processed['ISO'] = processed['SLG'] - processed['AVG']
        processed['K_pct'] = (processed['SO'] / processed['PA']).replace([np.inf, -np.inf], 0.22).fillna(0.22)
        processed['BB_pct'] = (processed['BB'] / processed['PA']).replace([np.inf, -np.inf], 0.08).fillna(0.08)
        processed['power_rating'] = (processed['HR'] * 4 + processed['3B'] * 3 + processed['2B'] * 2 + processed['H']) / processed['AB'].clip(lower=1)
        processed['speed_rating'] = processed['SB'] / processed['G'].clip(lower=1)
        
        return processed

    def _get_fallback_player_stats(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create fallback player stats when PyBaseball unavailable."""
        # Create basic pitcher stats for each team
        pitcher_data = []
        batting_data = []
        
        for team_id in range(1, 31):
            # Create 5 pitchers per team (typical rotation)
            for i in range(5):
                pitcher_data.append({
                    'Name': f'Pitcher_{team_id}_{i}',
                    'Team': f'Team_{team_id}',
                    'team_id': team_id,
                    'IP': 150.0,
                    'ERA': np.random.normal(4.2, 0.8),
                    'WHIP': np.random.normal(1.3, 0.15),
                    'SO': np.random.normal(120, 30),
                    'BB': np.random.normal(45, 15),
                    'W': np.random.randint(8, 15),
                    'L': np.random.randint(6, 12),
                    'SV': 0 if i < 4 else np.random.randint(0, 30),
                    'K9': np.random.normal(8.5, 2.0),
                    'BB9': np.random.normal(3.2, 1.0),
                    'K_BB_ratio': np.random.normal(2.5, 0.8),
                    'win_pct': np.random.normal(0.5, 0.1),
                    'is_starter': 1 if i < 4 else 0,
                    'is_closer': 0 if i < 4 else 1
                })
            
            # Create 9 batters per team (typical lineup)
            for i in range(9):
                batting_data.append({
                    'Name': f'Batter_{team_id}_{i}',
                    'Team': f'Team_{team_id}',
                    'team_id': team_id,
                    'G': 140,
                    'PA': 550,
                    'AB': 500,
                    'R': np.random.randint(60, 100),
                    'H': np.random.randint(120, 180),
                    '2B': np.random.randint(25, 40),
                    '3B': np.random.randint(2, 8),
                    'HR': np.random.randint(15, 35),
                    'RBI': np.random.randint(60, 100),
                    'SB': np.random.randint(5, 25),
                    'BB': np.random.randint(40, 80),
                    'SO': np.random.randint(80, 140),
                    'AVG': np.random.normal(0.260, 0.035),
                    'OBP': np.random.normal(0.330, 0.040),
                    'SLG': np.random.normal(0.430, 0.060),
                    'OPS': np.random.normal(0.760, 0.080),
                    'ISO': np.random.normal(0.170, 0.040),
                    'K_pct': np.random.normal(0.22, 0.05),
                    'BB_pct': np.random.normal(0.08, 0.02),
                    'power_rating': np.random.normal(1.8, 0.4),
                    'speed_rating': np.random.normal(0.15, 0.08)
                })
        
        pitcher_df = pd.DataFrame(pitcher_data)
        batting_df = pd.DataFrame(batting_data)
        
        return pitcher_df, batting_df

    def predict_starting_pitchers(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Predict starting pitchers for today's games using rotation logic."""
        logger.info("🎯 Predicting starting pitchers using rotation logic...")
        
        games_with_pitchers = games_df.copy()
        
        if not isinstance(self.current_pitcher_stats, pd.DataFrame) or self.current_pitcher_stats.empty:
            logger.warning("⚠️ No pitcher stats available, using fallback")
            return self._add_fallback_pitchers(games_with_pitchers)
        
        # Get starters only
        starters = self.current_pitcher_stats[self.current_pitcher_stats['is_starter'] == 1].copy()
        
        pitcher_assignments = []
        
        for idx, game in games_df.iterrows():
            home_team_id = game.get('home_team_id')
            away_team_id = game.get('away_team_id')
            
            # Get team pitchers
            home_pitchers = starters[starters['team_id'] == home_team_id]
            away_pitchers = starters[starters['team_id'] == away_team_id]
            
            # Select best available pitcher (lowest ERA among starters)
            home_pitcher = self._select_best_pitcher(home_pitchers, home_team_id)
            away_pitcher = self._select_best_pitcher(away_pitchers, away_team_id)
            
            pitcher_assignments.append({
                'game_idx': idx,
                'home_pitcher_name': home_pitcher.get('Name', 'Unknown'),
                'home_pitcher_era': home_pitcher.get('ERA', 4.50),
                'home_pitcher_whip': home_pitcher.get('WHIP', 1.35),
                'home_pitcher_k9': home_pitcher.get('K9', 8.5),
                'home_pitcher_bb9': home_pitcher.get('BB9', 3.2),
                'home_pitcher_win_pct': home_pitcher.get('win_pct', 0.5),
                'away_pitcher_name': away_pitcher.get('Name', 'Unknown'),
                'away_pitcher_era': away_pitcher.get('ERA', 4.50),
                'away_pitcher_whip': away_pitcher.get('WHIP', 1.35),
                'away_pitcher_k9': away_pitcher.get('K9', 8.5),
                'away_pitcher_bb9': away_pitcher.get('BB9', 3.2),
                'away_pitcher_win_pct': away_pitcher.get('win_pct', 0.5)
            })
        
        # Add pitcher info to games
        pitcher_df = pd.DataFrame(pitcher_assignments)
        for col in pitcher_df.columns:
            if col != 'game_idx':
                games_with_pitchers[col] = pitcher_df[col]
        
        logger.info(f"✅ Added pitcher assignments for {len(games_with_pitchers)} games")
        return games_with_pitchers

    def _select_best_pitcher(self, team_pitchers: pd.DataFrame, team_id: int) -> Dict:
        """Select the best available pitcher for a team."""
        if team_pitchers.empty:
            return {
                'Name': f'Unknown_Pitcher_{team_id}',
                'ERA': 4.50,
                'WHIP': 1.35,
                'K9': 8.5,
                'BB9': 3.2,
                'win_pct': 0.5
            }
        
        # Simple rotation logic - select pitcher with best ERA
        best_pitcher = team_pitchers.loc[team_pitchers['ERA'].idxmin()]
        return best_pitcher.to_dict()

    def _add_fallback_pitchers(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Add fallback pitcher data when stats unavailable."""
        games_with_pitchers = games_df.copy()
        
        # Add average pitcher stats
        pitcher_cols = {
            'home_pitcher_name': 'Unknown Home Pitcher',
            'home_pitcher_era': 4.50,
            'home_pitcher_whip': 1.35,
            'home_pitcher_k9': 8.5,
            'home_pitcher_bb9': 3.2,
            'home_pitcher_win_pct': 0.5,
            'away_pitcher_name': 'Unknown Away Pitcher',
            'away_pitcher_era': 4.50,
            'away_pitcher_whip': 1.35,
            'away_pitcher_k9': 8.5,
            'away_pitcher_bb9': 3.2,
            'away_pitcher_win_pct': 0.5
        }
        
        for col, default_val in pitcher_cols.items():
            games_with_pitchers[col] = default_val
            
        return games_with_pitchers

    def calculate_hitting_matchups(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate hitting vs pitching matchups for run prediction."""
        logger.info("💪 Calculating hitting vs pitching matchups...")
        
        games_with_matchups = games_df.copy()
        
        if not isinstance(self.current_batting_stats, pd.DataFrame) or self.current_batting_stats.empty:
            logger.warning("⚠️ No batting stats available, using fallback")
            return self._add_fallback_hitting_stats(games_with_matchups)
        
        matchup_features = []
        
        for idx, game in games_df.iterrows():
            home_team_id = game.get('home_team_id')
            away_team_id = game.get('away_team_id')
            
            # Get team hitting stats
            home_hitters = self.current_batting_stats[self.current_batting_stats['team_id'] == home_team_id]
            away_hitters = self.current_batting_stats[self.current_batting_stats['team_id'] == away_team_id]
            
            # Calculate team hitting aggregates
            home_hitting = self._aggregate_team_hitting(home_hitters)
            away_hitting = self._aggregate_team_hitting(away_hitters)
            
            # Calculate matchup vs opposing pitcher
            home_vs_away_pitcher = self._calculate_pitcher_matchup(home_hitting, game, 'away')
            away_vs_home_pitcher = self._calculate_pitcher_matchup(away_hitting, game, 'home')
            
            matchup_features.append({
                'game_idx': idx,
                # Home team hitting
                'home_team_avg': home_hitting['avg'],
                'home_team_obp': home_hitting['obp'],
                'home_team_slg': home_hitting['slg'],
                'home_team_ops': home_hitting['ops'],
                'home_team_power': home_hitting['power_rating'],
                'home_team_k_pct': home_hitting['k_pct'],
                'home_expected_runs_vs_pitcher': home_vs_away_pitcher,
                # Away team hitting
                'away_team_avg': away_hitting['avg'],
                'away_team_obp': away_hitting['obp'],
                'away_team_slg': away_hitting['slg'],
                'away_team_ops': away_hitting['ops'],
                'away_team_power': away_hitting['power_rating'],
                'away_team_k_pct': away_hitting['k_pct'],
                'away_expected_runs_vs_pitcher': away_vs_home_pitcher,
            })
        
        # Add matchup features to games
        matchup_df = pd.DataFrame(matchup_features)
        for col in matchup_df.columns:
            if col != 'game_idx':
                games_with_matchups[col] = matchup_df[col]
        
        logger.info(f"✅ Added hitting matchup features for {len(games_with_matchups)} games")
        return games_with_matchups

    def _aggregate_team_hitting(self, team_hitters: pd.DataFrame) -> Dict:
        """Aggregate team hitting statistics."""
        if team_hitters.empty:
            return {
                'avg': 0.250, 'obp': 0.320, 'slg': 0.400, 'ops': 0.720,
                'power_rating': 1.5, 'k_pct': 0.22, 'bb_pct': 0.08
            }
        
        # Weight by plate appearances
        pa_weights = team_hitters['PA'] / team_hitters['PA'].sum()
        
        return {
            'avg': np.average(team_hitters['AVG'], weights=pa_weights),
            'obp': np.average(team_hitters['OBP'], weights=pa_weights),
            'slg': np.average(team_hitters['SLG'], weights=pa_weights),
            'ops': np.average(team_hitters['OPS'], weights=pa_weights),
            'power_rating': np.average(team_hitters['power_rating'], weights=pa_weights),
            'k_pct': np.average(team_hitters['K_pct'], weights=pa_weights),
            'bb_pct': np.average(team_hitters['BB_pct'], weights=pa_weights)
        }

    def _calculate_pitcher_matchup(self, hitting_stats: Dict, game: pd.Series, pitcher_side: str) -> float:
        """Calculate expected runs based on pitcher vs hitting matchup."""
        pitcher_era = game.get(f'{pitcher_side}_pitcher_era', 4.50)
        pitcher_whip = game.get(f'{pitcher_side}_pitcher_whip', 1.35)
        pitcher_k9 = game.get(f'{pitcher_side}_pitcher_k9', 8.5)
        
        # Basic runs expectation model
        base_runs = 4.5  # League average
        
        # Adjust for pitcher quality
        era_adjustment = (pitcher_era - 4.50) * -0.5  # Better ERA = more runs allowed reduced
        whip_adjustment = (pitcher_whip - 1.35) * -1.0  # Better WHIP = fewer baserunners
        k_adjustment = (pitcher_k9 - 8.5) * -0.1  # More strikeouts = fewer runs
        
        # Adjust for hitting quality
        ops_adjustment = (hitting_stats['ops'] - 0.720) * 2.0  # Better OPS = more runs
        power_adjustment = (hitting_stats['power_rating'] - 1.5) * 0.8
        
        expected_runs = base_runs + era_adjustment + whip_adjustment + k_adjustment + ops_adjustment + power_adjustment
        
        return max(expected_runs, 2.0)  # Minimum 2 runs

    def _add_fallback_hitting_stats(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Add fallback hitting stats when data unavailable."""
        games_with_hitting = games_df.copy()
        
        hitting_cols = {
            'home_team_avg': 0.250, 'home_team_obp': 0.320, 'home_team_slg': 0.400,
            'home_team_ops': 0.720, 'home_team_power': 1.5, 'home_team_k_pct': 0.22,
            'home_expected_runs_vs_pitcher': 4.5,
            'away_team_avg': 0.250, 'away_team_obp': 0.320, 'away_team_slg': 0.400,
            'away_team_ops': 0.720, 'away_team_power': 1.5, 'away_team_k_pct': 0.22,
            'away_expected_runs_vs_pitcher': 4.5
        }
        
        for col, default_val in hitting_cols.items():
            games_with_hitting[col] = default_val
            
        return games_with_hitting

    def load_api_sports_training_data(self) -> pd.DataFrame:
        """Load training data from API Sports with enhanced pitcher tracking."""
        logger.info("🏟️ Loading training data from API Sports...")
        
        if not self.db:
            raise ValueError("API Sports database not available")
        
        try:
            historical_games = self.db.get_historical_data(self.training_seasons)
            
            if historical_games.empty:
                logger.warning("⚠️ No data from API Sports, trying direct database query...")
                with self.db.get_connection() as conn:
                    historical_games = pd.read_sql_query(
                        "SELECT * FROM games WHERE status = 'Finished' ORDER BY date DESC LIMIT 10000", 
                        conn
                    )
            
            if historical_games.empty:
                raise ValueError("No training data available from API Sports")
            
            finished_games = historical_games[historical_games['status'] == 'Finished'].copy()
            finished_games = self._add_game_time_features(finished_games)
            
            # Try to add historical pitcher data if available
            finished_games = self._add_historical_pitcher_data(finished_games)
            
            logger.info(f"✅ Loaded {len(finished_games)} finished games from API Sports")
            logger.info(f"📅 Date range: {finished_games['date'].min()} to {finished_games['date'].max()}")
            
            return finished_games
            
        except Exception as e:
            logger.error(f"❌ Failed to load API Sports data: {e}")
            raise

    def _add_historical_pitcher_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add historical pitcher data if available in database."""
        try:
            # Try to get pitcher data from your database
            if self.db:
                pitcher_data = self.db.get_pitcher_matchups(df['game_id'].tolist())
                if not pitcher_data.empty:
                    df = df.merge(pitcher_data, on='game_id', how='left')
                    logger.info("✅ Added historical pitcher data from database")
        except Exception as e:
            logger.warning(f"⚠️ Could not load historical pitcher data: {e}")
        
        return df

    def _add_game_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add game time features using local timezone (America/New_York by default)."""
        out = df.copy()

        # 1) Parse a UTC timestamp from whichever column we have
        if 'commence_time' in out.columns:
            t_utc = pd.to_datetime(out['commence_time'], utc=True, errors='coerce')
        elif 'time' in out.columns:
            # If provider gave naive times, assume UTC, then localize
            t_utc = pd.to_datetime(out['time'], errors='coerce')
            if t_utc.dt.tz is None:
                t_utc = t_utc.dt.tz_localize('UTC')
            else:
                t_utc = t_utc.dt.tz_convert('UTC')
        else:
            # No time columns; fabricate a neutral evening time in UTC
            t_utc = pd.to_datetime('now', utc=True)  # fallback

        # 2) Convert to local tz
        t_local = t_utc.dt.tz_convert(self.local_tz)

        # 3) Persist both; keep `game_time` as LOCAL for backward-compat
        out['game_time_utc']   = t_utc
        out['game_time_local'] = t_local
        out['game_time']       = t_local.dt.tz_convert(self.local_tz)  # alias
        out['game_date_local'] = t_local.dt.date

        # 4) Build features off LOCAL time
        out['game_hour']        = t_local.dt.hour
        out['is_day_game']      = (out['game_hour'] < 18).astype(int)
        out['is_evening_game']  = (out['game_hour'] >= 18).astype(int)

        # Your code uses a naive 'date' for splitting; make it local calendar date (naive)
        out['date'] = pd.to_datetime(t_local.dt.date)

        # Doubleheader markers (unchanged)
        if 'date' in out.columns and 'home_team_id' in out.columns and 'away_team_id' in out.columns:
            out = out.sort_values(['date', 'home_team_id', 'away_team_id', 'game_time_local']).reset_index(drop=True)
            out['game_number'] = out.groupby(['date', 'home_team_id', 'away_team_id']).cumcount() + 1
            out['is_doubleheader'] = (out['game_number'] > 1).astype(int)
        else:
            out['game_number'] = 1
            out['is_doubleheader'] = 0

        return out

    def get_current_team_stats_pybaseball(self) -> pd.DataFrame:
        """Get current team stats from PyBaseball."""
        logger.info("📈 Getting current team stats from PyBaseball...")

        if not PYBASEBALL_AVAILABLE:
            return self._get_fallback_team_stats()

        try:
            current_season = self.current_season
            batting_stats = pyb.team_batting(current_season)
            pitching_stats = pyb.team_pitching(current_season)

            # Process team stats similar to original method
            required_batting_cols = ['Team', 'G', 'W', 'L', 'R']
            required_pitching_cols = ['Team', 'RA', 'ERA']

            for col in required_batting_cols:
                if col not in batting_stats.columns:
                    if col == 'R':
                        batting_stats['R'] = pd.to_numeric(batting_stats.get('Runs', np.nan), errors='coerce')
                    elif col == 'G':
                        batting_stats['G'] = pd.to_numeric(batting_stats.get('G', np.nan), errors='coerce')
                    elif col in ('W', 'L'):
                        batting_stats[col] = pd.to_numeric(batting_stats.get(col, np.nan), errors='coerce')

            for col in required_pitching_cols:
                if col not in pitching_stats.columns:
                    if col == 'RA':
                        pitching_stats['RA'] = pd.to_numeric(pitching_stats.get('RA', np.nan), errors='coerce')
                    elif col == 'ERA':
                        pitching_stats['ERA'] = pd.to_numeric(pitching_stats.get('ERA', np.nan), errors='coerce')

            batting_clean = batting_stats[['Team', 'G', 'W', 'L', 'R']].copy()
            pitching_clean = pitching_stats[['Team', 'RA', 'ERA']].copy()

            for c in ['G','W','L','R']:
                batting_clean[c] = pd.to_numeric(batting_clean[c], errors='coerce')
            for c in ['RA','ERA']:
                pitching_clean[c] = pd.to_numeric(pitching_clean[c], errors='coerce')

            team_stats = batting_clean.merge(pitching_clean, on='Team', how='outer')

            # Fill missing values
            team_stats['G'] = team_stats['G'].fillna(162)
            team_stats['W'] = team_stats['W'].fillna(81)
            team_stats['L'] = team_stats['L'].fillna(81)
            team_stats['R'] = team_stats['R'].fillna(4.5 * team_stats['G'])
            team_stats['RA'] = team_stats['RA'].fillna(4.5 * team_stats['G'])
            team_stats['ERA'] = team_stats['ERA'].fillna(4.00)

            # Map teams to IDs
            team_stats['team_id'] = team_stats['Team'].apply(
                lambda x: self._standardize_team_name(x, allow_hash_fallback=False)
            )
            team_stats = team_stats.dropna(subset=['team_id'])
            team_stats['team_id'] = team_stats['team_id'].astype(int)
            team_stats = team_stats.sort_values('Team').drop_duplicates('team_id', keep='first')

            # Calculate rates
            denom = (team_stats['W'] + team_stats['L']).replace(0, np.nan)
            team_stats['win_pct'] = (team_stats['W'] / denom).fillna(0.5)
            team_stats['runs_per_game'] = (team_stats['R'] / team_stats['G']).replace([np.inf, -np.inf], np.nan).fillna(4.5)
            team_stats['runs_allowed_per_game'] = (team_stats['RA'] / team_stats['G']).replace([np.inf, -np.inf], np.nan).fillna(4.5)
            team_stats['season'] = current_season

            final_stats = team_stats[['team_id','season','win_pct','runs_per_game','runs_allowed_per_game','ERA']].copy()

            return final_stats

        except Exception as e:
            logger.error(f"❌ PyBaseball stats failed: {e}")
            return self._get_fallback_team_stats()

    def _standardize_team_name(self, team_name: str, allow_hash_fallback: bool = False) -> Optional[int]:
        """Map team names to stable team IDs."""
        if not team_name:
            return None
        
        s = str(team_name).lower().strip().replace(".", "")
        
        if s in CANON_TEAM_MAP:
            return CANON_TEAM_MAP[s]
        
        for key in CANON_TEAM_MAP.keys():
            if s == key or s in key or key in s:
                return CANON_TEAM_MAP[key]
        
        if allow_hash_fallback:
            logger.warning(f"⚠️ Unknown team name (hash fallback): {team_name}")
            return hash(s) % 30 + 1
        else:
            logger.warning(f"⚠️ Unknown team name: {team_name}")
            return None

    def _get_fallback_team_stats(self) -> pd.DataFrame:
        """Fallback team stats when PyBaseball unavailable."""
        fallback_stats = []
        for team_id in range(1, 31):
            fallback_stats.append({
                'team_id': team_id,
                'season': self.current_season,
                'win_pct': 0.5,
                'runs_per_game': 4.5,
                'runs_allowed_per_game': 4.5,
                'ERA': 4.0
            })
        return pd.DataFrame(fallback_stats)

    def engineer_enhanced_hybrid_features(self, games_df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Engineer features for both game outcome and runs prediction."""
        logger.info("🔧 Engineering enhanced hybrid features with player stats...")
        
        # Defensive initialization - ensure player stats are DataFrames
        if not hasattr(self, 'current_pitcher_stats') or not isinstance(self.current_pitcher_stats, pd.DataFrame):
            self.current_pitcher_stats = pd.DataFrame()
        if not hasattr(self, 'current_batting_stats') or not isinstance(self.current_batting_stats, pd.DataFrame):
            self.current_batting_stats = pd.DataFrame()
        
        # Load current player stats
        if (is_training or 
            not hasattr(self, 'current_pitcher_stats') or 
            not isinstance(self.current_pitcher_stats, pd.DataFrame) or 
            self.current_pitcher_stats.empty):
            logger.info("🔄 Loading current player stats...")
            self.load_current_player_stats()
            logger.info(f"✅ Player stats loaded - Pitchers: {type(self.current_pitcher_stats)} ({len(self.current_pitcher_stats)} rows)")
            logger.info(f"✅ Player stats loaded - Batters: {type(self.current_batting_stats)} ({len(self.current_batting_stats)} rows)")
        
        # Start with team-level features
        features_df = self.engineer_hybrid_features(games_df, is_training)
        
        # Add pitcher assignments for prediction games
        if not is_training:
            games_with_pitchers = self.predict_starting_pitchers(games_df)
            games_with_hitting = self.calculate_hitting_matchups(games_with_pitchers)
        else:
            games_with_pitchers = games_df.copy()
            games_with_hitting = games_df.copy()
        
        # Merge pitcher and hitting features
        pitcher_hitting_features = self._extract_pitcher_hitting_features(games_with_hitting)
        
        # Combine all features
        enhanced_features = features_df.copy()
        for col in pitcher_hitting_features.columns:
            if col not in enhanced_features.columns:
                enhanced_features[col] = pitcher_hitting_features[col]
        
        # Create runs-specific features
        runs_features = self._create_runs_features(enhanced_features)
        
        logger.info(f"✅ Enhanced features: {len(enhanced_features.columns)} game features, {len(runs_features.columns)} runs features")
        
        return enhanced_features, runs_features

    def engineer_hybrid_features(self, games_df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Original hybrid feature engineering (team-level)."""
        logger.info("🔧 Engineering team-level hybrid features...")
        
        features_df = games_df.copy().reset_index(drop=True)
        original_length = len(features_df)
        
        # Basic features
        numeric_features = ['home_team_id', 'away_team_id', 'season']
        
        # Game time features
        time_features = ['game_hour', 'is_day_game', 'is_evening_game', 'game_number', 'is_doubleheader']
        for feat in time_features:
            if feat in features_df.columns:
                numeric_features.append(feat)
        
        # Get current team stats
        current_stats = self.get_current_team_stats_pybaseball()
        
        if not current_stats.empty:
            merge_cols = ['team_id', 'win_pct', 'runs_per_game', 'runs_allowed_per_game', 'ERA']
            
            # Merge team stats
            features_df = features_df.merge(
                current_stats[merge_cols],
                left_on='home_team_id', right_on='team_id',
                how='left', suffixes=('', '_home'), validate='many_to_one'
            ).drop('team_id', axis=1, errors='ignore')
            
            features_df = features_df.reset_index(drop=True)
            
            features_df = features_df.merge(
                current_stats[merge_cols],
                left_on='away_team_id', right_on='team_id',
                how='left', suffixes=('_home', '_away'), validate='many_to_one'
            ).drop('team_id', axis=1, errors='ignore')
            
            features_df = features_df.reset_index(drop=True)
            
            stat_features = ['win_pct_home', 'win_pct_away', 'runs_per_game_home', 'runs_per_game_away',
                        'runs_allowed_per_game_home', 'runs_allowed_per_game_away', 'ERA_home', 'ERA_away']
            numeric_features.extend([f for f in stat_features if f in features_df.columns])
        
        # Calculated features
        if 'win_pct_home' in features_df.columns and 'win_pct_away' in features_df.columns:
            features_df['win_pct_differential'] = features_df['win_pct_home'] - features_df['win_pct_away']
            numeric_features.append('win_pct_differential')
        
        if 'ERA_home' in features_df.columns and 'ERA_away' in features_df.columns:
            features_df['era_differential'] = features_df['ERA_home'] - features_df['ERA_away']
            features_df['era_advantage_home'] = (features_df['ERA_home'] < features_df['ERA_away']).astype(int)
            numeric_features.extend(['era_differential', 'era_advantage_home'])
        
        # Temporal features
        if 'date' in features_df.columns:
            features_df['date'] = pd.to_datetime(features_df['date'])
            features_df['month'] = features_df['date'].dt.month
            features_df['day_of_week'] = features_df['date'].dt.dayofweek
            features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
            numeric_features.extend(['month', 'day_of_week', 'is_weekend'])
        
        # Home field advantage
        features_df['home_field_advantage'] = 1
        numeric_features.append('home_field_advantage')
        
        # Get available features
        available_features = [feat for feat in numeric_features if feat in features_df.columns]
        X = features_df[available_features].copy()
        
        # Apply categorical encoding
        X = self._apply_categorical_encoding(X)
        
        # Fill missing values
        for col in X.columns:
            if pd.api.types.is_object_dtype(X[col].dtype):
                X[col] = pd.Categorical(X[col]).codes
            else:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                
                if 'era' in col.lower():
                    X[col] = X[col].fillna(4.00)
                elif 'win_pct' in col.lower():
                    X[col] = X[col].fillna(0.5)
                elif 'runs' in col.lower():
                    X[col] = X[col].fillna(4.5)
                else:
                    X[col] = X[col].fillna(X[col].median() if not X[col].isna().all() else 0)
        
        X = X.reset_index(drop=True)
        
        if len(X) != original_length:
            raise ValueError(f"Feature engineering changed number of rows: {original_length} -> {len(X)}")
        
        return X

    def _apply_categorical_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply categorical encoding."""
        X_fixed = X.copy()
        
        if all(col in X_fixed.columns for col in self.categorical_cols):
            if not self.cat_encoder_fitted:
                X_fixed[self.categorical_cols] = self.ordinal_encoder.fit_transform(
                    X_fixed[self.categorical_cols].astype(str)
                )
                self.cat_encoder_fitted = True
            else:
                X_fixed[self.categorical_cols] = self.ordinal_encoder.transform(
                    X_fixed[self.categorical_cols].astype(str)
                )
        
        return X_fixed

    def _extract_pitcher_hitting_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Extract pitcher and hitting features from games."""
        pitcher_hitting_features = games_df.copy()
        
        # Define pitcher and hitting feature columns
        pitcher_features = [
            'home_pitcher_era', 'home_pitcher_whip', 'home_pitcher_k9', 'home_pitcher_bb9', 'home_pitcher_win_pct',
            'away_pitcher_era', 'away_pitcher_whip', 'away_pitcher_k9', 'away_pitcher_bb9', 'away_pitcher_win_pct'
        ]
        
        hitting_features = [
            'home_team_avg', 'home_team_obp', 'home_team_slg', 'home_team_ops', 'home_team_power', 'home_team_k_pct',
            'away_team_avg', 'away_team_obp', 'away_team_slg', 'away_team_ops', 'away_team_power', 'away_team_k_pct',
            'home_expected_runs_vs_pitcher', 'away_expected_runs_vs_pitcher'
        ]
        
        # Fill missing pitcher features
        for feat in pitcher_features:
            if feat not in pitcher_hitting_features.columns:
                if 'era' in feat:
                    pitcher_hitting_features[feat] = 4.50
                elif 'whip' in feat:
                    pitcher_hitting_features[feat] = 1.35
                elif 'k9' in feat:
                    pitcher_hitting_features[feat] = 8.5
                elif 'bb9' in feat:
                    pitcher_hitting_features[feat] = 3.2
                elif 'win_pct' in feat:
                    pitcher_hitting_features[feat] = 0.5
        
        # Fill missing hitting features
        for feat in hitting_features:
            if feat not in pitcher_hitting_features.columns:
                if 'avg' in feat:
                    pitcher_hitting_features[feat] = 0.250
                elif 'obp' in feat:
                    pitcher_hitting_features[feat] = 0.320
                elif 'slg' in feat:
                    pitcher_hitting_features[feat] = 0.400
                elif 'ops' in feat:
                    pitcher_hitting_features[feat] = 0.720
                elif 'power' in feat:
                    pitcher_hitting_features[feat] = 1.5
                elif 'k_pct' in feat:
                    pitcher_hitting_features[feat] = 0.22
                elif 'expected_runs' in feat:
                    pitcher_hitting_features[feat] = 4.5
        
        # Calculate pitcher vs pitcher differentials
        if 'home_pitcher_era' in pitcher_hitting_features.columns and 'away_pitcher_era' in pitcher_hitting_features.columns:
            pitcher_hitting_features['pitcher_era_diff'] = pitcher_hitting_features['home_pitcher_era'] - pitcher_hitting_features['away_pitcher_era']
            pitcher_hitting_features['pitcher_whip_diff'] = pitcher_hitting_features['home_pitcher_whip'] - pitcher_hitting_features['away_pitcher_whip']
            pitcher_hitting_features['pitcher_k9_diff'] = pitcher_hitting_features['home_pitcher_k9'] - pitcher_hitting_features['away_pitcher_k9']
            pitcher_hitting_features['pitcher_quality_advantage_home'] = (
                (pitcher_hitting_features['home_pitcher_era'] < pitcher_hitting_features['away_pitcher_era']) &
                (pitcher_hitting_features['home_pitcher_whip'] < pitcher_hitting_features['away_pitcher_whip'])
            ).astype(int)
        
        # Calculate hitting vs hitting differentials
        if 'home_team_ops' in pitcher_hitting_features.columns and 'away_team_ops' in pitcher_hitting_features.columns:
            pitcher_hitting_features['hitting_ops_diff'] = pitcher_hitting_features['home_team_ops'] - pitcher_hitting_features['away_team_ops']
            pitcher_hitting_features['hitting_power_diff'] = pitcher_hitting_features['home_team_power'] - pitcher_hitting_features['away_team_power']
            pitcher_hitting_features['hitting_advantage_home'] = (pitcher_hitting_features['home_team_ops'] > pitcher_hitting_features['away_team_ops']).astype(int)
        
        # Keep only numeric pitcher and hitting columns
        feature_cols = pitcher_features + hitting_features + [
            'pitcher_era_diff', 'pitcher_whip_diff', 'pitcher_k9_diff', 'pitcher_quality_advantage_home',
            'hitting_ops_diff', 'hitting_power_diff', 'hitting_advantage_home'
        ]
        
        available_cols = [col for col in feature_cols if col in pitcher_hitting_features.columns]
        return pitcher_hitting_features[available_cols].fillna(0)

    def _create_runs_features(self, enhanced_features: pd.DataFrame) -> pd.DataFrame:
        """Create features specifically for runs prediction."""
        runs_features = enhanced_features.copy()
        
        # Focus on offensive/defensive features for runs prediction
        runs_specific_features = []
        
        for col in runs_features.columns:
            # Include features most relevant to run scoring
            if any(term in col.lower() for term in [
                'runs', 'era', 'whip', 'ops', 'avg', 'obp', 'slg', 'power', 'expected_runs',
                'pitcher', 'hitting', 'home_field', 'k9', 'bb9'
            ]):
                runs_specific_features.append(col)
        
        # Add interaction features for runs prediction
        if 'home_expected_runs_vs_pitcher' in runs_features.columns and 'away_expected_runs_vs_pitcher' in runs_features.columns:
            runs_features['total_expected_runs'] = runs_features['home_expected_runs_vs_pitcher'] + runs_features['away_expected_runs_vs_pitcher']
            runs_features['runs_differential'] = runs_features['home_expected_runs_vs_pitcher'] - runs_features['away_expected_runs_vs_pitcher']
            runs_specific_features.extend(['total_expected_runs', 'runs_differential'])
        
        # Park factors (simplified)
        runs_features['offensive_park_factor'] = 1.0  # Default neutral park
        runs_specific_features.append('offensive_park_factor')
        
        return runs_features[runs_specific_features].fillna(0)

    def train_enhanced_model(self):
        """Train enhanced model with player stats integration."""
        logger.info("🚀 Training Enhanced Hybrid MLB Model with Player Stats...")
        logger.info("=" * 70)
        
        try:
            # Load training data
            games_df = self.load_api_sports_training_data()
            
            # Create targets
            if 'home_score' in games_df.columns and 'away_score' in games_df.columns:
                y_outcome = (games_df['home_score'] > games_df['away_score']).astype(int)
                y_total_runs = games_df['home_score'] + games_df['away_score']
                y_home_runs = games_df['home_score']
                y_away_runs = games_df['away_score']
            else:
                raise ValueError("No score data available for training")
            
            # Reset indices
            games_df = games_df.reset_index(drop=True)
            y_outcome = y_outcome.reset_index(drop=True)
            y_total_runs = y_total_runs.reset_index(drop=True)
            
            logger.info(f"📊 Data loaded: {len(games_df)} games")
            logger.info(f"   🎯 Home win rate: {y_outcome.mean():.1%}")
            logger.info(f"   🏃 Avg total runs: {y_total_runs.mean():.1f}")
            
            # Engineer enhanced features
            X_outcome, X_runs = self.engineer_enhanced_hybrid_features(games_df, is_training=True)
            
            # Reset indices
            X_outcome = X_outcome.reset_index(drop=True)
            X_runs = X_runs.reset_index(drop=True)
            
            self.feature_names = X_outcome.columns.tolist()
            self.runs_feature_names = X_runs.columns.tolist()
            
            # Validate alignment
            if len(X_outcome) != len(y_outcome) or len(X_runs) != len(y_total_runs):
                raise ValueError("Data length mismatch after feature engineering")
            
            # Time-aware split
            X_outcome_train, X_outcome_test, y_outcome_train, y_outcome_test = self._split_last_45_days(
                X_outcome, y_outcome, games_df
            )
            X_runs_train, X_runs_test, y_runs_train, y_runs_test = self._split_last_45_days(
                X_runs, y_total_runs, games_df
            )
            
            # Train game outcome model
            logger.info("🎯 Training game outcome models...")
            X_outcome_train_scaled = self.scaler.fit_transform(X_outcome_train)
            X_outcome_test_scaled = self.scaler.transform(X_outcome_test)
            
            outcome_results = {}
            for model_name, config in self.model_configs.items():
                try:
                    result = self._train_single_model(
                        model_name, X_outcome_train_scaled, y_outcome_train, 
                        X_outcome_test_scaled, y_outcome_test
                    )
                    outcome_results[model_name] = result
                except Exception as e:
                    logger.error(f"❌ Failed to train {model_name}: {e}")
            
            # Select best outcome model
            best_outcome_name = max(outcome_results.keys(), key=lambda k: outcome_results[k]['metrics']['roc_auc'])
            self.best_model = outcome_results[best_outcome_name]['model']
            self.best_model_name = best_outcome_name
            
            # Train runs prediction model
            logger.info("🏃 Training runs prediction models...")
            X_runs_train_scaled = self.runs_scaler.fit_transform(X_runs_train)
            X_runs_test_scaled = self.runs_scaler.transform(X_runs_test)
            
            runs_results = {}
            for model_name, model in self.runs_model_configs.items():
                try:
                    model.fit(X_runs_train_scaled, y_runs_train)
                    y_runs_pred = model.predict(X_runs_test_scaled)
                    
                    mae = mean_absolute_error(y_runs_test, y_runs_pred)
                    rmse = np.sqrt(np.mean((y_runs_test - y_runs_pred) ** 2))
                    
                    runs_results[model_name] = {
                        'model': model,
                        'mae': mae,
                        'rmse': rmse,
                        'mean_prediction': np.mean(y_runs_pred),
                        'prediction_std': np.std(y_runs_pred)
                    }
                    
                    logger.info(f"   ✅ {model_name}: MAE {mae:.2f}, RMSE {rmse:.2f}, Mean {np.mean(y_runs_pred):.1f}")
                except Exception as e:
                    logger.error(f"❌ Failed to train runs model {model_name}: {e}")
            
            # Select best runs model
            if runs_results:
                best_runs_name = min(runs_results.keys(), key=lambda k: runs_results[k]['mae'])
                self.runs_model = runs_results[best_runs_name]['model']
                logger.info(f"🏆 Best runs model: {best_runs_name}")
            
            # Save models
            self._save_enhanced_model()
            
            # Log results
            self._log_enhanced_training_summary(outcome_results, runs_results, best_outcome_name)
            
        except Exception as e:
            logger.error(f"❌ Enhanced training failed: {e}")
            import traceback
            traceback.print_exc()

    def _split_last_45_days(self, X: pd.DataFrame, y: pd.Series, games_df: pd.DataFrame) -> Tuple:
        """Split data with last 45 days as test set."""
        if 'date' not in games_df.columns:
            return train_test_split(X, y, test_size=0.2, random_state=self.random_state)
        
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True) 
        games_df = games_df.reset_index(drop=True)
        
        if not (len(X) == len(y) == len(games_df)):
            logger.info("🔄 Falling back to random split due to length mismatch")
            return train_test_split(X, y, test_size=0.2, random_state=self.random_state)
        
        try:
            games_df['date'] = pd.to_datetime(games_df['date'])
            latest_date = games_df['date'].max()
            cutoff_date = latest_date - timedelta(days=self.test_days)
            
            train_mask = games_df['date'] <= cutoff_date
            test_mask = games_df['date'] > cutoff_date
            
            train_indices = games_df.index[train_mask].tolist()
            test_indices = games_df.index[test_mask].tolist()
            
            X_train = X.iloc[train_indices]
            X_test = X.iloc[test_indices]
            y_train = y.iloc[train_indices]
            y_test = y.iloc[test_indices]
            
            logger.info(f"📅 Time-aware split: {len(X_train)} train, {len(X_test)} test")
            
            if len(X_train) == 0 or len(X_test) == 0:
                return train_test_split(X, y, test_size=0.2, random_state=self.random_state)
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"❌ Time-aware split failed: {e}")
            return train_test_split(X, y, test_size=0.2, random_state=self.random_state)

    def _train_single_model(self, model_name: str, X_train, y_train, X_test, y_test) -> Dict:
        """Train a single classification model."""
        config = self.model_configs[model_name]
        model = config['model_class'](**config['params'])
        model.fit(X_train, y_train)
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'log_loss': log_loss(y_test, y_pred_proba),
            'prediction_variance': np.var(y_pred_proba)
        }
        
        return {'model': model, 'metrics': metrics}

    def predict_enhanced_games(self):
        """Predict today's games using enhanced model with player stats."""
        logger.info("🔮 Making enhanced predictions with player stats...")
        logger.info("💰 Getting live games from YOUR paid Odds API...")
        
        try:
            self._load_enhanced_model()
            
            todays_games = self._get_todays_games_from_odds_api()
            
            if todays_games.empty:
                logger.warning("⚪ No games available for enhanced prediction")
                return
            
            predictions = self._make_enhanced_predictions(todays_games)
            
            self._display_enhanced_predictions(predictions)
            
            self._save_predictions_for_learning(predictions)
            
        except Exception as e:
            logger.error(f"❌ Enhanced prediction failed: {e}")
            import traceback
            traceback.print_exc()

    def _get_todays_games_from_odds_api(self) -> pd.DataFrame:
        """Get today's games from odds API."""
        logger.info("💰 Fetching from YOUR paid Odds API...")
        
        if not self.odds_api:
            logger.error("❌ Your paid Odds API not available!")
            return self._create_sample_games()
        
        try:
            odds_data = self.odds_api.get_odds('mlb', markets=['h2h'])
            
            if odds_data.empty:
                logger.warning("⚪ No games from your Odds API today")
                return self._create_sample_games()
            
            games_df = self._process_odds_data(odds_data)
            games_df = self._add_team_ids_from_csv_dict(games_df)
            games_df = self._add_game_time_features(games_df)
            games_df['season'] = self.current_season
            
            logger.info(f"✅ Found {len(games_df)} games from YOUR paid Odds API")
            
            return games_df
            
        except Exception as e:
            logger.error(f"❌ Your paid Odds API failed: {e}")
            return self._create_sample_games()

    def _process_odds_data(self, odds_data: pd.DataFrame) -> pd.DataFrame:
        """Process odds data from API."""
        moneyline_data = odds_data[odds_data['market'].str.lower().isin(['h2h', 'moneyline'])].copy()
        
        if moneyline_data.empty:
            return pd.DataFrame()
        
        # Select bookmaker
        preferred_bookmakers = ['FanDuel', 'DraftKings', 'BetMGM', 'Caesars']
        available_bookmakers = moneyline_data['bookmaker'].unique()
        
        selected_bookmaker = next((b for b in preferred_bookmakers if b in available_bookmakers), available_bookmakers[0])
        
        single_book = moneyline_data[moneyline_data['bookmaker'] == selected_bookmaker]
        
        games_base = single_book.drop_duplicates('game_id')[
            ['game_id', 'home_team', 'away_team', 'commence_time']
        ].copy()
        
        home_odds = single_book[single_book['team'] == 'home'][['game_id', 'odds']]
        away_odds = single_book[single_book['team'] == 'away'][['game_id', 'odds']]
        
        games_with_odds = games_base.merge(home_odds, on='game_id', how='left')
        games_with_odds = games_with_odds.merge(away_odds, on='game_id', how='left', suffixes=('_home', '_away'))
        
        games_with_odds.columns = ['game_id', 'home_team_name', 'away_team_name', 'commence_time', 'home_odds', 'away_odds']
        
        return games_with_odds.dropna()

    def _add_team_ids_from_csv_dict(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Add team IDs using canonical mapping."""
        df = games_df.copy()
        
        df['home_team_id'] = df['home_team_name'].apply(
            lambda s: self._standardize_team_name(s, allow_hash_fallback=True)
        )
        df['away_team_id'] = df['away_team_name'].apply(
            lambda s: self._standardize_team_name(s, allow_hash_fallback=True)
        )
        
        return df

    def _create_sample_games(self) -> pd.DataFrame:
        """Create sample games for testing."""
        return pd.DataFrame({
            'game_id': ['sample_1', 'sample_2', 'sample_3'],
            'home_team_name': ['New York Yankees', 'Los Angeles Dodgers', 'Houston Astros'],
            'away_team_name': ['Boston Red Sox', 'San Francisco Giants', 'Seattle Mariners'],
            'commence_time': [pd.Timestamp.now() + pd.Timedelta(hours=i*3) for i in range(3)],
            'home_odds': [-150, -120, -180],
            'away_odds': [130, 100, 150]
        })

    def _make_enhanced_predictions(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Make enhanced predictions with player stats."""
        logger.info(f"🔮 Making enhanced predictions for {len(games_df)} games...")
        logger.info("📈 Using current PyBaseball team & player stats...")
        
        # Engineer enhanced features
        X_outcome, X_runs = self.engineer_enhanced_hybrid_features(games_df, is_training=False)
        
        # Ensure we have all features for outcome prediction
        missing_outcome_features = []
        for feature in self.feature_names:
            if feature not in X_outcome.columns:
                missing_outcome_features.append(feature)
                X_outcome[feature] = 0
        
        if missing_outcome_features:
            logger.warning(f"⚠️ Added {len(missing_outcome_features)} missing outcome features")
        
        X_outcome_selected = X_outcome[self.feature_names].copy()
        
        # Predict game outcomes
        X_outcome_scaled = self.scaler.transform(X_outcome_selected)
        outcome_probabilities = self.best_model.predict_proba(X_outcome_scaled)
        
        # Predict runs if runs model available
        runs_predictions = None
        if self.runs_model and self.runs_feature_names:
            missing_runs_features = []
            for feature in self.runs_feature_names:
                if feature not in X_runs.columns:
                    missing_runs_features.append(feature)
                    X_runs[feature] = 0
            
            if missing_runs_features:
                logger.warning(f"⚠️ Added {len(missing_runs_features)} missing runs features")
            
            X_runs_selected = X_runs[self.runs_feature_names].copy()
            X_runs_scaled = self.runs_scaler.transform(X_runs_selected)
            runs_predictions = self.runs_model.predict(X_runs_scaled)
        
        # Build enhanced results
        results = games_df.copy()
        home_probs = outcome_probabilities[:, 1]
        
        results['home_win_prob'] = home_probs
        results['away_win_prob'] = 1 - home_probs
        results['predicted_winner'] = np.where(home_probs > 0.5, results['home_team_name'], results['away_team_name'])
        results['confidence'] = np.abs(home_probs - 0.5) * 2
        
        # Add runs predictions
        if runs_predictions is not None:
            results['predicted_total_runs'] = np.maximum(runs_predictions, 4.0)  # Minimum 4 runs
            # Estimate individual team runs based on win probability and expected runs
            home_run_share = 0.45 + (home_probs - 0.5) * 0.2  # 45-55% range based on win prob
            results['predicted_home_runs'] = results['predicted_total_runs'] * home_run_share
            results['predicted_away_runs'] = results['predicted_total_runs'] * (1 - home_run_share)
        else:
            results['predicted_total_runs'] = 9.0
            results['predicted_home_runs'] = 4.5
            results['predicted_away_runs'] = 4.5
        
        # Add betting analysis
        if 'home_odds' in results.columns:
            results['home_ev'] = self._calculate_expected_value(home_probs, results['home_odds'])
            results['away_ev'] = self._calculate_expected_value(1 - home_probs, results['away_odds'])
            results['best_bet'] = results.apply(self._determine_best_bet, axis=1)
        
        # Enhanced prediction quality checks
        variance = np.var(home_probs)
        mean_home_prob = np.mean(home_probs)
        
        logger.info(f"🔍 Enhanced Model Quality:")
        logger.info(f"   Outcome variance: {variance:.4f}")
        logger.info(f"   Mean home prob: {mean_home_prob:.3f}")
        if runs_predictions is not None:
            logger.info(f"   Mean total runs: {np.mean(runs_predictions):.1f}")
            logger.info(f"   Runs std: {np.std(runs_predictions):.2f}")
        
        return results

    def _calculate_expected_value(self, win_prob: np.ndarray, odds: pd.Series) -> pd.Series:
        """Calculate expected value for betting."""
        def ev_single(prob, odd):
            if pd.isna(odd) or pd.isna(prob):
                return 0
            if odd > 0:
                return (prob * odd / 100) - (1 - prob)
            else:
                return (prob * 100 / abs(odd)) - (1 - prob)
        
        return pd.Series([ev_single(p, o) for p, o in zip(win_prob, odds)])

    def _determine_best_bet(self, row: pd.Series) -> str:
        """Determine best betting option."""
        if 'home_ev' in row and 'away_ev' in row:
            if row['home_ev'] > 0.05 and row['home_ev'] > row['away_ev']:
                return f"Home (+{row['home_ev']:.1%} EV)"
            elif row['away_ev'] > 0.05 and row['away_ev'] > row['home_ev']:
                return f"Away (+{row['away_ev']:.1%} EV)"
        return 'No Edge'

    def _display_enhanced_predictions(self, results: pd.DataFrame):
        """Display enhanced predictions with player stats insights."""
        print("\n" + "="*85)
        print("⚾ ENHANCED HYBRID MLB PREDICTIONS WITH PLAYER STATS")
        print("🏟️ Training: API Sports | 📈 Teams: PyBaseball | ⚾ Players: Individual Stats")
        print("🎯 Pitching: Rotation Prediction | 💪 Hitting: Matchup Analysis | 🏃 Runs: Player-Based")
        print("="*85)
        
        for idx, row in results.iterrows():
            print(f"\n🏟️ {row['away_team_name']} @ {row['home_team_name']}")
            
            # Game time
            if 'game_time_local' in row and pd.notna(row['game_time_local']):
                game_time = pd.to_datetime(row['game_time_local'])
            elif 'commence_time' in row and pd.notna(row['commence_time']):  # fallback
                game_time = pd.to_datetime(row['commence_time'], utc=True).tz_convert(self.local_tz)
            else:
                game_time = pd.Timestamp.now(tz=self.local_tz)

            tz_abbr = game_time.tzname()  # "EDT" in summer, "EST" in winter
            time_str = game_time.strftime('%I:%M %p').lstrip('0')
            print(f"  ⏰ Game Time: {time_str} {tz_abbr}", end="")
            if 'is_doubleheader' in row and row['is_doubleheader']:
                print(f" (Game #{row.get('game_number', 1)} - Doubleheader)")
            else:
                print()
            
            # Enhanced prediction with runs
            predicted_team = row['predicted_winner']
            if predicted_team == row['home_team_name']:
                predicted_prob = row['home_win_prob']
            else:
                predicted_prob = row['away_win_prob']
            
            print(f"  🎯 Prediction: {predicted_team} ({predicted_prob:.1%})")
            print(f"  📊 Probabilities: Home {row['home_win_prob']:.1%} | Away {row['away_win_prob']:.1%}")
            print(f"  📈 Confidence: {row['confidence']:.1%}")
            
            # Enhanced runs prediction
            if 'predicted_total_runs' in row:
                print(f"  🏃 Expected Runs: {row['predicted_total_runs']:.1f} total")
                print(f"     • Home: {row['predicted_home_runs']:.1f} | Away: {row['predicted_away_runs']:.1f}")
            
            # Pitcher matchup if available
            if 'home_pitcher_name' in row and 'away_pitcher_name' in row:
                print(f"  🎯 Starting Pitchers:")
                print(f"     • Home: {row['home_pitcher_name']} (ERA: {row.get('home_pitcher_era', 0):.2f})")
                print(f"     • Away: {row['away_pitcher_name']} (ERA: {row.get('away_pitcher_era', 0):.2f})")
            
            # Team hitting insights if available
            if 'home_team_ops' in row and 'away_team_ops' in row:
                print(f"  💪 Team Hitting (OPS):")
                print(f"     • Home: {row['home_team_ops']:.3f} | Away: {row['away_team_ops']:.3f}")
            
            # Odds and betting
            if 'home_odds' in row:
                print(f"  💰 Your Odds API: Home {row['home_odds']:+.0f} | Away {row['away_odds']:+.0f}")
                if 'home_ev' in row:
                    print(f"  📈 Expected Value: Home {row['home_ev']:.1%} | Away {row['away_ev']:.1%}")
                
                if 'best_bet' in row and row['best_bet'] != 'No Edge':
                    print(f"  ⭐ RECOMMENDED BET: {row['best_bet']}")
        
        # Enhanced model quality summary
        home_probs = results['home_win_prob']
        variance = np.var(home_probs)
        mean_home_prob = np.mean(home_probs)
        
        print(f"\n🔍 Enhanced Model Quality Assessment:")
        print(f"   📊 Outcome prediction variance: {variance:.4f}")
        print(f"   🏠 Mean home win probability: {mean_home_prob:.3f}")
        
        if 'predicted_total_runs' in results.columns:
            runs_mean = results['predicted_total_runs'].mean()
            runs_std = results['predicted_total_runs'].std()
            print(f"   🏃 Average predicted total runs: {runs_mean:.1f} ± {runs_std:.2f}")
            
            if runs_std > 0.5:
                print("   ✅ EXCELLENT runs variance - Player matchups working!")
            elif runs_std > 0.3:
                print("   ✅ GOOD runs variance - Meaningful differentiation")
            else:
                print("   ⚠️ LOW runs variance - Check player stats quality")
        
        print(f"\n📊 Enhanced Data Sources:")
        print(f"   🏟️ Training Data: API Sports historical")
        print(f"   📈 Team Stats: PyBaseball current season")
        print(f"   ⚾ Player Stats: PyBaseball individual performance")
        print(f"   🎯 Pitcher Rotation: Starting pitcher prediction")
        print(f"   💪 Hitting Analysis: Team vs pitcher matchups")
        print(f"   💰 Live Odds: Your paid Odds API")
        print(f"   🏃 Runs Model: Enhanced player-based prediction")

    def _save_predictions_for_learning(self, predictions: pd.DataFrame):
        """Save enhanced predictions for learning."""
        predictions_dict = predictions.copy()
        
        # Convert Timestamps to strings
        for col in predictions_dict.columns:
            if pd.api.types.is_datetime64_any_dtype(predictions_dict[col]):
                predictions_dict[col] = predictions_dict[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        prediction_record = {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'enhanced_hybrid_with_players',
            'predictions': predictions_dict.to_dict('records'),
            'outcome_model': self.best_model_name,
            'runs_model': 'enhanced' if self.runs_model else 'fallback'
        }
        
        # Load existing predictions
        if self.predictions_log.exists():
            try:
                with open(self.predictions_log, 'r') as f:
                    existing_predictions = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                existing_predictions = []
        else:
            existing_predictions = []
        
        existing_predictions.append(prediction_record)
        
        # Save updated predictions
        try:
            with open(self.predictions_log, 'w') as f:
                json.dump(existing_predictions, f, indent=2, default=str)
            
            logger.info(f"💾 Saved enhanced predictions for learning: {self.predictions_log}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save predictions: {e}")

    def _save_enhanced_model(self):
        """Save the enhanced model with player stats."""
        model_data = {
            'outcome_model': self.best_model,
            'runs_model': self.runs_model,
            'scaler': self.scaler,
            'runs_scaler': self.runs_scaler,
            'feature_names': self.feature_names,
            'runs_feature_names': self.runs_feature_names,
            'ordinal_encoder': self.ordinal_encoder,
            'cat_encoder_fitted': self.cat_encoder_fitted,
            'categorical_cols': self.categorical_cols,
            'best_model_name': self.best_model_name,
            'team_name_dict': self.team_name_dict,
            'training_date': datetime.now().isoformat(),
            'model_type': 'enhanced_hybrid_with_players'
        }
        
        with open(self.model_dir / 'enhanced_hybrid_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"💾 Enhanced hybrid model saved")

    def _load_enhanced_model(self):
        """Load the enhanced model with player stats."""
        model_path = self.model_dir / 'enhanced_hybrid_model.pkl'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Enhanced model not found. Train first: python {__file__} --train")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.best_model = model_data['outcome_model']
        self.runs_model = model_data.get('runs_model')
        self.scaler = model_data['scaler']
        self.runs_scaler = model_data.get('runs_scaler')
        self.feature_names = model_data['feature_names']
        self.runs_feature_names = model_data.get('runs_feature_names', [])
        self.ordinal_encoder = model_data['ordinal_encoder']
        self.cat_encoder_fitted = model_data['cat_encoder_fitted']
        self.categorical_cols = model_data['categorical_cols']
        self.best_model_name = model_data['best_model_name']
        self.team_name_dict = model_data.get('team_name_dict', {})
        
        logger.info(f"✅ Loaded enhanced hybrid model: {self.best_model_name}")
        if self.runs_model:
            logger.info(f"✅ Runs prediction model loaded")

    def _log_enhanced_training_summary(self, outcome_results: dict, runs_results: dict, best_outcome_name: str):
        """Log enhanced training summary."""
        logger.info("=" * 70)
        logger.info("🏆 ENHANCED HYBRID MODEL TRAINING RESULTS:")
        
        # Outcome models comparison
        if outcome_results:
            outcome_df = pd.DataFrame({
                name: result['metrics'] for name, result in outcome_results.items()
            }).T
            
            logger.info(f"\n🎯 GAME OUTCOME MODELS:")
            logger.info(f"{outcome_df[['accuracy', 'roc_auc', 'prediction_variance']].round(4)}")
            
            best_outcome_metrics = outcome_results[best_outcome_name]['metrics']
            logger.info(f"\n🥇 BEST OUTCOME MODEL: {best_outcome_name}")
            logger.info(f"   Accuracy: {best_outcome_metrics['accuracy']:.3f}")
            logger.info(f"   ROC-AUC: {best_outcome_metrics['roc_auc']:.3f}")
            logger.info(f"   Variance: {best_outcome_metrics['prediction_variance']:.4f}")
        
        # Runs models comparison
        if runs_results:
            logger.info(f"\n🏃 RUNS PREDICTION MODELS:")
            for name, result in runs_results.items():
                logger.info(f"   {name}: MAE {result['mae']:.2f}, RMSE {result['rmse']:.2f}")
        
        logger.info("\n✅ ENHANCED DATA SOURCES:")
        logger.info("   🏟️ Training: API Sports historical")
        logger.info("   📈 Team Stats: PyBaseball current")
        logger.info("   ⚾ Player Stats: Individual performance")
        logger.info("   🎯 Pitcher Rotation: Starting pitcher prediction")
        logger.info("   💪 Hitting Matchups: Team vs pitcher analysis")
        logger.info("   📋 Name Dict: CSV files validation")
        logger.info("   💰 Live Odds: Your paid Odds API")
        logger.info("   🏃 Runs Model: Player-based prediction")
        logger.info("=" * 70)


def main():
    """Main function for enhanced model."""
    parser = argparse.ArgumentParser(description='Enhanced Hybrid MLB Model with Player Stats & Timezone Support')
    parser.add_argument('--train', action='store_true', help='Train enhanced model with player stats')
    parser.add_argument('--predict', action='store_true', help='Predict with enhanced player analysis')
    parser.add_argument('--install-pybaseball', action='store_true', help='Install PyBaseball')
    parser.add_argument('--timezone', default='America/New_York', help='Local timezone for game times (default: America/New_York)')
    args = parser.parse_args()
    
    if args.install_pybaseball:
        import subprocess
        print("📦 Installing PyBaseball...")
        subprocess.check_call(['pip', 'install', 'pybaseball'])
        print("✅ PyBaseball installed!")
        return
    
    model = EnhancedHybridMLBModel(local_tz=args.timezone)
    
    if args.train:
        model.train_enhanced_model()
    elif args.predict:
        model.predict_enhanced_games()
    else:
        print("🚀 Enhanced Hybrid MLB Model with Timezone Support")
        print("Perfect player stats architecture:")
        print("  🏟️ Training: API Sports historical (your paid investment)")
        print("  📈 Team Stats: PyBaseball current season (free)")
        print("  ⚾ Player Stats: PyBaseball individual performance (free)")
        print("  🎯 Pitcher Rotation: Starting pitcher prediction")
        print("  💪 Hitting Matchups: Lineup vs pitcher analysis")
        print("  📋 Name Dictionary: CSV files validation")
        print("  💰 Live Odds: Your paid Odds API")
        print("  🏃 Runs Prediction: Enhanced with player matchups")
        print("  ⏰ Game Time: Timezone-aware (US/Eastern by default)")
        print("  🧠 Learning: Track prediction accuracy")
        print("\nCommands:")
        print("  --install-pybaseball  Install PyBaseball")
        print("  --train              Train enhanced model with player stats")
        print("  --predict            Predict with enhanced player analysis")
        print("  --timezone TZ        Set timezone (default: America/New_York)")
        print("                       Examples: America/Chicago, America/Los_Angeles")

if __name__ == "__main__":
    main()
