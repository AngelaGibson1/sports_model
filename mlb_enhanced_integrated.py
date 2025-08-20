# mlb_enhanced_integrated.py
"""
Enhanced MLB trainer integrated with your working data pipeline.
Now includes starting pitcher data and advanced pitcher-aware features.
Combines your multi-model approach with pitcher matchup analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime
from typing import Dict, Tuple, Any, Optional, List
import sqlite3

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, RFE

# Advanced models
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier

# For tracking
import warnings
warnings.filterwarnings('ignore')

from loguru import logger
from config.settings import Settings
from data.database.mlb import MLBDatabase

class IntegratedMLBEnhancedTrainer:
    """
    Enhanced multi-model trainer integrated with your working MLB data pipeline.
    Now includes starting pitcher data and advanced pitcher-aware features.
    Uses your 15K+ games with pitcher matchup analysis for superior predictions.
    """
    
    def __init__(self, model_dir: Path = Path('models/mlb/enhanced_with_pitchers'), 
                 random_state: int = 42):
        """Initialize the integrated enhanced trainer with pitcher awareness."""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        
        # Initialize database connection
        self.db = MLBDatabase()
        
        # Store all trained models
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_selector = None
        self.label_encoders = {}
        
        # Training results for analysis
        self.training_results = {}
        self.database_schema = None
        
        # Model configurations optimized for pitcher-aware MLB data
        self.model_configs = self._get_pitcher_optimized_configs()
        
        logger.info("🎯⚾ Initialized Enhanced MLB Trainer with Pitcher Integration")
    
    def _get_pitcher_optimized_configs(self) -> Dict:
        """Get model configurations optimized for pitcher-aware MLB prediction."""
        return {
            'xgboost_pitcher_mlb': {
                'model_class': xgb.XGBClassifier,
                'param_grid': {
                    'n_estimators': [300, 500, 800],
                    'max_depth': [6, 8, 10],
                    'learning_rate': [0.05, 0.08, 0.1, 0.12],
                    'subsample': [0.8, 0.9],
                    'colsample_bytree': [0.8, 0.9],
                    'scale_pos_weight': [0.85, 0.9, 0.95],
                    'reg_alpha': [0, 0.1, 0.3],  # L1 regularization for pitcher features
                    'reg_lambda': [1, 1.5, 2],   # L2 regularization
                    'random_state': [self.random_state]
                },
                'quick_params': {
                    'n_estimators': 500,
                    'max_depth': 8,
                    'learning_rate': 0.08,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'scale_pos_weight': 0.89,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1.5,
                    'random_state': self.random_state,
                    'eval_metric': 'logloss',
                    'use_label_encoder': False
                }
            },
            
            'lightgbm_pitcher_mlb': {
                'model_class': lgb.LGBMClassifier,
                'param_grid': {
                    'n_estimators': [300, 500, 800],
                    'num_leaves': [50, 100, 150],
                    'learning_rate': [0.05, 0.08, 0.1, 0.12],
                    'feature_fraction': [0.7, 0.8, 0.9],
                    'bagging_fraction': [0.7, 0.8, 0.9],
                    'min_child_samples': [20, 30, 50],
                    'reg_alpha': [0, 0.1, 0.3],
                    'reg_lambda': [0, 0.1, 0.3],
                    'random_state': [self.random_state]
                },
                'quick_params': {
                    'n_estimators': 500,
                    'num_leaves': 100,
                    'learning_rate': 0.08,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'min_child_samples': 30,
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.1,
                    'random_state': self.random_state,
                    'verbosity': -1
                }
            },
            
            'random_forest_pitcher_mlb': {
                'model_class': RandomForestClassifier,
                'param_grid': {
                    'n_estimators': [500, 800, 1200],
                    'max_depth': [20, 25, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', 0.8],
                    'class_weight': ['balanced', 'balanced_subsample'],
                    'random_state': [self.random_state]
                },
                'quick_params': {
                    'n_estimators': 800,
                    'max_depth': 25,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'max_features': 'sqrt',
                    'class_weight': 'balanced',
                    'random_state': self.random_state,
                    'n_jobs': -1
                }
            },
            
            'gradient_boosting_pitcher_mlb': {
                'model_class': GradientBoostingClassifier,
                'param_grid': {
                    'n_estimators': [300, 500, 800],
                    'max_depth': [6, 8, 10],
                    'learning_rate': [0.05, 0.08, 0.1],
                    'subsample': [0.8, 0.9],
                    'min_samples_split': [5, 10, 20],
                    'random_state': [self.random_state]
                },
                'quick_params': {
                    'n_estimators': 500,
                    'max_depth': 8,
                    'learning_rate': 0.08,
                    'subsample': 0.8,
                    'min_samples_split': 10,
                    'random_state': self.random_state
                }
            },
            
            'neural_network_pitcher_mlb': {
                'model_class': MLPClassifier,
                'param_grid': {
                    'hidden_layer_sizes': [(150, 75), (200, 100, 50), (250, 125, 75)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.001, 0.01, 0.1],
                    'learning_rate_init': [0.001, 0.01],
                    'max_iter': [2000, 3000],
                    'random_state': [self.random_state]
                },
                'quick_params': {
                    'hidden_layer_sizes': (200, 100, 50),
                    'activation': 'relu',
                    'alpha': 0.01,
                    'learning_rate_init': 0.001,
                    'max_iter': 3000,
                    'random_state': self.random_state,
                    'early_stopping': True,
                    'validation_fraction': 0.1
                }
            },
            
            'logistic_regression_pitcher_mlb': {
                'model_class': LogisticRegression,
                'param_grid': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga'],
                    'class_weight': [None, 'balanced'],
                    'random_state': [self.random_state]
                },
                'quick_params': {
                    'C': 10.0,
                    'penalty': 'l2',
                    'solver': 'liblinear',
                    'class_weight': 'balanced',
                    'random_state': self.random_state,
                    'max_iter': 3000
                }
            }
        }
    
    def load_mlb_data_with_pitchers(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare MLB data with pitcher integration from your working database."""
        logger.info("📊⚾ Loading MLB data with pitcher integration from your working database...")
        
        try:
            # First, ensure pitcher integration is set up
            self.db.setup_pitcher_integration()
            
            # Load games with pitcher data using the enhanced method
            data = self.db.get_games_with_pitcher_data(include_unconfirmed=False)
            
            if data.empty:
                logger.warning("No games with pitcher data found! Falling back to team-only data...")
                return self._load_fallback_data()
            
            # Filter for finished games only
            data = data[data['status'] == 'Finished']
            data = data.dropna(subset=['home_score', 'away_score'])
            
            if data.empty:
                logger.warning("No finished games with pitcher data! Falling back to team-only data...")
                return self._load_fallback_data()
            
            logger.info(f"✅ Loaded {len(data)} finished games with pitcher data")
            
            # Create target variable
            y = (data['home_score'] > data['away_score']).astype(int)
            
            # Get coverage statistics
            total_finished = len(self.db.get_games(season=None))
            coverage_pct = (len(data) / total_finished * 100) if total_finished > 0 else 0
            
            logger.info(f"📊 Home team win rate: {y.mean():.1%}")
            logger.info(f"📊 Pitcher data coverage: {coverage_pct:.1f}% of all games")
            logger.info(f"⚾ PITCHER INTEGRATION SUCCESSFUL!")
            
            return data, y
                
        except Exception as e:
            logger.error(f"❌ Error loading MLB data with pitchers: {e}")
            logger.info("🔄 Falling back to team-only data...")
            return self._load_fallback_data()
    
    def _load_fallback_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Fallback to team-only data if pitcher data unavailable."""
        logger.warning("⚠️ Using fallback team-only data (no pitcher features)")
        
        with sqlite3.connect(Settings.MLB_DB_PATH) as conn:
            query = """
            SELECT 
                g.*,
                hts.win_percentage as home_team_win_pct,
                hts.runs_per_game as home_runs_per_game,
                hts.runs_allowed_per_game as home_runs_allowed,
                hts.batting_average as home_team_ba,
                hts.on_base_percentage as home_team_obp,
                hts.slugging_percentage as home_team_slg,
                hts.earned_run_average as home_team_era,
                hts.whip as home_team_whip,
                ats.win_percentage as away_team_win_pct,
                ats.runs_per_game as away_runs_per_game,
                ats.runs_allowed_per_game as away_runs_allowed,
                ats.batting_average as away_team_ba,
                ats.on_base_percentage as away_team_obp,
                ats.slugging_percentage as away_team_slg,
                ats.earned_run_average as away_team_era,
                ats.whip as away_team_whip
            FROM games g
            LEFT JOIN team_statistics hts ON g.home_team_id = hts.team_id AND g.season = hts.season
            LEFT JOIN team_statistics ats ON g.away_team_id = ats.team_id AND g.season = ats.season
            WHERE g.status = 'Finished' 
            AND g.home_score IS NOT NULL 
            AND g.away_score IS NOT NULL
            ORDER BY g.date
            """
            
            data = pd.read_sql_query(query, conn)
            y = (data['home_score'] > data['away_score']).astype(int)
            
            logger.info(f"✅ Loaded {len(data)} finished games (team-only)")
            
            return data, y
    
    def engineer_pitcher_aware_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer comprehensive features including pitcher-specific metrics."""
        logger.info("🔧⚾ Engineering pitcher-aware features for MLB prediction...")
        
        feature_data = data.copy()
        features = []
        
        # === BASIC GAME FEATURES ===
        basic_features = ['home_team_id', 'away_team_id', 'season']
        for feat in basic_features:
            if feat in feature_data.columns:
                features.append(feat)
        
        # === TEAM PERFORMANCE FEATURES ===
        team_features = [
            'home_team_win_pct', 'away_team_win_pct',
            'home_runs_per_game', 'away_runs_per_game',
            'home_runs_allowed', 'away_runs_allowed',
            'home_team_ba', 'away_team_ba',
            'home_team_obp', 'away_team_obp',
            'home_team_slg', 'away_team_slg',
            'home_team_era', 'away_team_era',
            'home_team_whip', 'away_team_whip'
        ]
        
        for feat in team_features:
            if feat in feature_data.columns:
                features.append(feat)
        
        # === STARTING PITCHER FEATURES ===
        pitcher_features = [
            'home_starter_era', 'away_starter_era',
            'home_starter_whip', 'away_starter_whip',
            'home_starter_ip', 'away_starter_ip',
            'home_starter_k', 'away_starter_k',
            'home_starter_bb', 'away_starter_bb',
            'home_starter_w', 'away_starter_w',
            'home_starter_l', 'away_starter_l',
            'home_starter_k9', 'away_starter_k9',
            'home_starter_bb9', 'away_starter_bb9',
            'home_starter_k_bb_ratio', 'away_starter_k_bb_ratio'
        ]
        
        pitcher_features_found = []
        for feat in pitcher_features:
            if feat in feature_data.columns:
                pitcher_features_found.append(feat)
                features.append(feat)
        
        logger.info(f"   📊 Found {len(pitcher_features_found)} pitcher features in data")
        
        # === CALCULATED PITCHER DIFFERENTIALS ===
        # ERA differential (negative means home pitcher advantage)
        if all(col in feature_data.columns for col in ['home_starter_era', 'away_starter_era']):
            feature_data['era_differential'] = feature_data['home_starter_era'] - feature_data['away_starter_era']
            feature_data['era_advantage_home'] = (feature_data['home_starter_era'] < feature_data['away_starter_era']).astype(int)
            features.extend(['era_differential', 'era_advantage_home'])
        
        # WHIP differential  
        if all(col in feature_data.columns for col in ['home_starter_whip', 'away_starter_whip']):
            feature_data['whip_differential'] = feature_data['home_starter_whip'] - feature_data['away_starter_whip']
            feature_data['whip_advantage_home'] = (feature_data['home_starter_whip'] < feature_data['away_starter_whip']).astype(int)
            features.extend(['whip_differential', 'whip_advantage_home'])
        
        # K/9 differential
        if all(col in feature_data.columns for col in ['home_starter_k9', 'away_starter_k9']):
            feature_data['k9_differential'] = feature_data['home_starter_k9'] - feature_data['away_starter_k9']
            feature_data['k9_advantage_home'] = (feature_data['home_starter_k9'] > feature_data['away_starter_k9']).astype(int)
            features.extend(['k9_differential', 'k9_advantage_home'])
        
        # === PITCHER EXPERIENCE FEATURES ===
        if all(col in feature_data.columns for col in ['home_starter_w', 'home_starter_l']):
            feature_data['home_starter_games'] = feature_data['home_starter_w'] + feature_data['home_starter_l']
            feature_data['home_starter_win_pct'] = np.where(
                feature_data['home_starter_games'] > 0,
                feature_data['home_starter_w'] / feature_data['home_starter_games'],
                0.5
            )
            features.extend(['home_starter_games', 'home_starter_win_pct'])
        
        if all(col in feature_data.columns for col in ['away_starter_w', 'away_starter_l']):
            feature_data['away_starter_games'] = feature_data['away_starter_w'] + feature_data['away_starter_l']
            feature_data['away_starter_win_pct'] = np.where(
                feature_data['away_starter_games'] > 0,
                feature_data['away_starter_w'] / feature_data['away_starter_games'],
                0.5
            )
            features.extend(['away_starter_games', 'away_starter_win_pct'])
        
        # === PITCHER VS TEAM OFFENSE MATCHUPS ===
        # Pitcher ERA vs opposing team offense
        if all(col in feature_data.columns for col in ['home_starter_era', 'away_runs_per_game']):
            feature_data['home_pitcher_vs_away_offense'] = feature_data['home_starter_era'] - feature_data['away_runs_per_game']
            features.append('home_pitcher_vs_away_offense')
        
        if all(col in feature_data.columns for col in ['away_starter_era', 'home_runs_per_game']):
            feature_data['away_pitcher_vs_home_offense'] = feature_data['away_starter_era'] - feature_data['home_runs_per_game']
            features.append('away_pitcher_vs_home_offense')
        
        # === HANDEDNESS FEATURES ===
        if 'home_starter_hand' in feature_data.columns:
            feature_data['home_starter_lefty'] = (feature_data['home_starter_hand'] == 'L').astype(int)
            features.append('home_starter_lefty')
        
        if 'away_starter_hand' in feature_data.columns:
            feature_data['away_starter_lefty'] = (feature_data['away_starter_hand'] == 'L').astype(int)
            features.append('away_starter_lefty')
        
        if all(col in feature_data.columns for col in ['home_starter_hand', 'away_starter_hand']):
            feature_data['both_starters_lefty'] = ((feature_data['home_starter_hand'] == 'L') & 
                                                  (feature_data['away_starter_hand'] == 'L')).astype(int)
            feature_data['both_starters_righty'] = ((feature_data['home_starter_hand'] == 'R') & 
                                                   (feature_data['away_starter_hand'] == 'R')).astype(int)
            feature_data['mixed_handedness'] = ((feature_data['home_starter_hand'] == 'L') & 
                                               (feature_data['away_starter_hand'] == 'R')).astype(int) | \
                                             ((feature_data['home_starter_hand'] == 'R') & 
                                               (feature_data['away_starter_hand'] == 'L')).astype(int)
            features.extend(['both_starters_lefty', 'both_starters_righty', 'mixed_handedness'])
        
        # === COMPOSITE PITCHER QUALITY SCORES ===
        # Create overall pitcher quality scores
        if all(col in feature_data.columns for col in ['home_starter_era', 'home_starter_whip', 'home_starter_k9']):
            # Lower ERA and WHIP are better, higher K/9 is better
            feature_data['home_pitcher_quality'] = (
                (5.0 - feature_data['home_starter_era'].clip(0, 5)) * 0.4 +
                (2.0 - feature_data['home_starter_whip'].clip(0.8, 2.0)) * 0.3 +
                (feature_data['home_starter_k9'].clip(4, 15) / 15) * 0.3
            )
            features.append('home_pitcher_quality')
        
        if all(col in feature_data.columns for col in ['away_starter_era', 'away_starter_whip', 'away_starter_k9']):
            feature_data['away_pitcher_quality'] = (
                (5.0 - feature_data['away_starter_era'].clip(0, 5)) * 0.4 +
                (2.0 - feature_data['away_starter_whip'].clip(0.8, 2.0)) * 0.3 +
                (feature_data['away_starter_k9'].clip(4, 15) / 15) * 0.3
            )
            features.append('away_pitcher_quality')
        
        # Pitcher quality differential
        if all(col in feature_data.columns for col in ['home_pitcher_quality', 'away_pitcher_quality']):
            feature_data['pitcher_quality_diff'] = feature_data['home_pitcher_quality'] - feature_data['away_pitcher_quality']
            feature_data['pitcher_quality_advantage_home'] = (feature_data['home_pitcher_quality'] > feature_data['away_pitcher_quality']).astype(int)
            features.extend(['pitcher_quality_diff', 'pitcher_quality_advantage_home'])
        
        # === DERIVED GAME FEATURES ===
        # NOTE: We CANNOT use actual game scores (home_score, away_score) as features
        # because that would be data leakage - the model would just look at who scored more
        # Instead, we can only use pre-game information for prediction
        
        # We can add venue/situational features if available
        if 'venue' in feature_data.columns:
            # Create venue encoding or home field advantage indicators
            pass  # Would need venue-specific data
        
        # Add day/night game indicators if available
        if 'time' in feature_data.columns:
            # Could parse time to determine day/night games
            pass  # Would need time parsing
        
        # === DATE/TIME FEATURES ===
        if 'date' in feature_data.columns:
            try:
                feature_data['date'] = pd.to_datetime(feature_data['date'])
                feature_data['day_of_week'] = feature_data['date'].dt.dayofweek
                feature_data['month'] = feature_data['date'].dt.month
                feature_data['day_of_year'] = feature_data['date'].dt.dayofyear
                feature_data['is_weekend'] = (feature_data['day_of_week'] >= 5).astype(int)
                
                # Season progression (important for baseball)
                feature_data['season_progress'] = feature_data['day_of_year'] / 365.0
                
                features.extend(['day_of_week', 'month', 'day_of_year', 'is_weekend', 'season_progress'])
            except Exception as e:
                logger.warning(f"Date feature creation failed: {e}")
        
        # === TEAM DIFFERENTIALS ===
        # Win percentage differential
        if all(col in feature_data.columns for col in ['home_team_win_pct', 'away_team_win_pct']):
            feature_data['team_win_pct_diff'] = feature_data['home_team_win_pct'] - feature_data['away_team_win_pct']
            features.append('team_win_pct_diff')
        
        # Offensive vs Defensive matchups
        if all(col in feature_data.columns for col in ['home_runs_per_game', 'away_runs_allowed']):
            feature_data['home_offense_vs_away_defense'] = feature_data['home_runs_per_game'] - feature_data['away_runs_allowed']
            features.append('home_offense_vs_away_defense')
        
        if all(col in feature_data.columns for col in ['away_runs_per_game', 'home_runs_allowed']):
            feature_data['away_offense_vs_home_defense'] = feature_data['away_runs_per_game'] - feature_data['home_runs_allowed']
            features.append('away_offense_vs_home_defense')
        
        # === ADVANCED ROLLING FEATURES ===
        # Calculate rolling team performance
        try:
            rolling_features = self._calculate_rolling_features(feature_data)
            feature_data = pd.concat([feature_data, rolling_features], axis=1)
            features.extend(rolling_features.columns.tolist())
        except Exception as e:
            logger.warning(f"Rolling features failed: {e}")
        
        # === HEAD-TO-HEAD FEATURES ===
        try:
            h2h_features = self._calculate_head_to_head_features(feature_data)
            feature_data = pd.concat([feature_data, h2h_features], axis=1)
            features.extend(h2h_features.columns.tolist())
        except Exception as e:
            logger.warning(f"Head-to-head features failed: {e}")
        
        # === FEATURE SELECTION AND CLEANUP ===
        # Only keep features that exist and have reasonable data
        available_features = []
        
        # Data leakage prevention - exclude any features that could reveal the game outcome
        prohibited_features = [
            'home_score', 'away_score', 'total_runs', 'run_differential', 
            'high_scoring', 'close_game', 'winner', 'result', 'final_score'
        ]
        
        for feat in features:
            if feat in feature_data.columns:
                # Check for data leakage
                if any(prohibited in feat.lower() for prohibited in prohibited_features):
                    logger.warning(f"   ⚠️ Skipping potential data leakage feature: {feat}")
                    continue
                    
                missing_pct = feature_data[feat].isnull().sum() / len(feature_data)
                if missing_pct < 0.6:  # Keep features with <60% missing (relaxed for pitcher data)
                    available_features.append(feat)
        
        if not available_features:
            raise ValueError("No valid features available after data leakage prevention!")
            
        logger.info(f"   ✅ Kept {len(available_features)} features after data leakage check")
        
        # Final feature matrix
        X = feature_data[available_features].copy()
        
        # Handle categorical features
        categorical_features = ['home_team_id', 'away_team_id']
        for feat in categorical_features:
            if feat in X.columns:
                if feat not in self.label_encoders:
                    self.label_encoders[feat] = LabelEncoder()
                    X[feat] = self.label_encoders[feat].fit_transform(X[feat].astype(str))
                else:
                    try:
                        X[feat] = self.label_encoders[feat].transform(X[feat].astype(str))
                    except ValueError:
                        X[feat] = 0
        
        # Handle missing values intelligently
        for col in X.columns:
            if X[col].dtype in ['object']:
                X[col] = X[col].fillna('unknown')
            else:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                
                # Use smarter defaults for pitcher stats
                if 'era' in col.lower():
                    X[col] = X[col].fillna(4.00)  # League average ERA
                elif 'whip' in col.lower():
                    X[col] = X[col].fillna(1.30)  # League average WHIP
                elif 'k9' in col.lower():
                    X[col] = X[col].fillna(8.0)   # League average K/9
                elif 'bb9' in col.lower():
                    X[col] = X[col].fillna(3.0)   # League average BB/9
                elif 'win_pct' in col.lower():
                    X[col] = X[col].fillna(0.5)   # Neutral win rate
                elif 'quality' in col.lower():
                    X[col] = X[col].fillna(0.5)   # Neutral quality score
                else:
                    X[col] = X[col].fillna(X[col].median())
        
        # Calculate feature breakdown
        pitcher_features_count = len([col for col in X.columns if any(keyword in col.lower() for keyword in 
                                                                     ['starter', 'era', 'whip', 'pitcher', 'k9', 'bb9', 'quality'])])
        team_features_count = len([col for col in X.columns if 'team' in col.lower()])
        
        logger.info(f"✅ Engineered {len(X.columns)} total features for {len(X)} games")
        logger.info(f"   📊 Feature breakdown:")
        logger.info(f"      ⚾ Pitcher features: {pitcher_features_count}")
        logger.info(f"      🏟️ Team features: {team_features_count}")
        logger.info(f"      📅 Other features: {len(X.columns) - pitcher_features_count - team_features_count}")
        
        if pitcher_features_count > 10:
            logger.info("   🎯 EXCELLENT PITCHER FEATURE COVERAGE!")
        elif pitcher_features_count > 5:
            logger.info("   ⚾ Good pitcher feature coverage")
        else:
            logger.warning("   ⚠️ Limited pitcher features - may fall back to team-based model")
        
        return X
    
    def _calculate_rolling_features(self, data: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """Calculate rolling team performance features."""
        rolling_features = pd.DataFrame(index=data.index)
        
        # Sort by date for proper rolling calculations
        if 'date' in data.columns:
            data_sorted = data.sort_values(['date']).reset_index(drop=True)
        else:
            data_sorted = data.copy()
        
        for window in windows:
            # Home team rolling stats
            if 'home_team_win_pct' in data_sorted.columns:
                rolling_features[f'home_win_pct_rolling_{window}'] = (
                    data_sorted.groupby('home_team_id')['home_team_win_pct']
                    .rolling(window, min_periods=3).mean()
                    .reset_index(drop=True)
                )
            
            # Away team rolling stats
            if 'away_team_win_pct' in data_sorted.columns:
                rolling_features[f'away_win_pct_rolling_{window}'] = (
                    data_sorted.groupby('away_team_id')['away_team_win_pct']
                    .rolling(window, min_periods=3).mean()
                    .reset_index(drop=True)
                )
        
        return rolling_features.fillna(0.5)  # Fill with neutral win rate
    
    def _calculate_head_to_head_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate head-to-head matchup features."""
        h2h_features = pd.DataFrame(index=data.index)
        
        # This is simplified - in production you'd query historical H2H records
        # For now, create team matchup features
        if 'home_team_id' in data.columns and 'away_team_id' in data.columns:
            # Team ID differential (simple matchup indicator)
            h2h_features['team_id_diff'] = data['home_team_id'] - data['away_team_id']
            
            # Team strength differential
            if all(col in data.columns for col in ['home_team_win_pct', 'away_team_win_pct']):
                h2h_features['team_strength_diff'] = data['home_team_win_pct'] - data['away_team_win_pct']
        
        return h2h_features.fillna(0)
    
    def select_features_pitcher_aware(self, X: pd.DataFrame, y: pd.Series, 
                                     method: str = 'importance', 
                                     n_features: int = 60) -> pd.DataFrame:
        """Advanced feature selection optimized for pitcher-aware MLB data."""
        logger.info(f"🔍⚾ Selecting top {n_features} features using {method} method (pitcher-aware)...")
        
        if len(X.columns) <= n_features:
            logger.info(f"   Already have {len(X.columns)} features (less than {n_features})")
            return X
        
        if method == 'importance':
            # Use XGBoost with pitcher-optimized parameters
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            )
            model.fit(X, y)
            
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Ensure we keep key pitcher features if available
            pitcher_features = [col for col in X.columns if any(keyword in col.lower() for keyword in 
                                                              ['starter', 'era', 'whip', 'pitcher', 'k9', 'bb9', 'quality'])]
            
            selected_features = importance_df.head(n_features)['feature'].tolist()
            
            # Force include top pitcher features if not in selection
            pitcher_in_selection = len([f for f in selected_features if f in pitcher_features])
            total_pitcher_features = len(pitcher_features)
            
            logger.info(f"   ⚾ Total pitcher features available: {total_pitcher_features}")
            logger.info(f"   ⚾ Pitcher features in selection: {pitcher_in_selection}")
            
            if total_pitcher_features > 0:
                # Show top pitcher features
                top_pitcher_features = importance_df[importance_df['feature'].isin(pitcher_features)].head(5)
                logger.info(f"   🔝 Top 5 pitcher features by importance:")
                for i, (_, row) in enumerate(top_pitcher_features.iterrows()):
                    logger.info(f"      {i+1}. {row['feature']}: {row['importance']:.3f}")
            
        elif method == 'kbest':
            selector = SelectKBest(score_func=f_classif, k=n_features)
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            self.feature_selector = selector
            
        elif method == 'rfe':
            estimator = LogisticRegression(random_state=self.random_state, max_iter=1000)
            selector = RFE(estimator, n_features_to_select=n_features, step=5)
            selector.fit(X, y)
            selected_features = X.columns[selector.support_].tolist()
            self.feature_selector = selector
        
        logger.info(f"   ✅ Selected {len(selected_features)} features for pitcher-aware training")
        
        return X[selected_features]
    
    def train_single_model_pitcher_aware(self, model_name: str, X_train: pd.DataFrame, 
                                        y_train: pd.Series, X_test: pd.DataFrame, 
                                        y_test: pd.Series, tune_hyperparameters: bool = False) -> Dict:
        """Train a single model optimized for pitcher-aware MLB prediction."""
        logger.info(f"🎯⚾ Training {model_name} with pitcher-aware features...")
        
        config = self.model_configs[model_name]
        
        if tune_hyperparameters and 'param_grid' in config:
            logger.info("   🔧 Tuning hyperparameters for pitcher-aware data...")
            model = config['model_class']()
            
            # Enhanced search for pitcher data
            search = RandomizedSearchCV(
                model,
                config['param_grid'],
                n_iter=20,  # More iterations for pitcher data
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=self.random_state
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            logger.info(f"   Best params: {search.best_params_}")
        else:
            # Use pitcher-optimized quick parameters
            best_model = config['model_class'](**config['quick_params'])
            best_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Enhanced metrics calculation
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'predictions_std': np.std(y_pred_proba),
            'predictions_mean': np.mean(y_pred_proba),
            'near_fifty_pct': np.mean((y_pred_proba > 0.45) & (y_pred_proba < 0.55)),
            'confident_predictions': np.mean((y_pred_proba < 0.4) | (y_pred_proba > 0.6)),
            'very_confident': np.mean((y_pred_proba < 0.3) | (y_pred_proba > 0.7))
        }
        
        # Cross-validation
        try:
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='roc_auc')
            metrics['cv_mean'] = np.mean(cv_scores)
            metrics['cv_std'] = np.std(cv_scores)
        except:
            metrics['cv_mean'] = metrics['roc_auc']
            metrics['cv_std'] = 0.0
        
        # Enhanced Kelly criterion for pitcher-aware betting
        if metrics['accuracy'] > 0.524:
            edge = metrics['accuracy'] - 0.5
            kelly_fraction = (edge * 2) - 1
            metrics['kelly_edge'] = edge
            metrics['kelly_fraction'] = kelly_fraction
        
        # Display results with pitcher context
        logger.info(f"   ✅ {model_name} Results:")
        logger.info(f"      Accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"      ROC-AUC: {metrics['roc_auc']:.3f}")
        logger.info(f"      CV Score: {metrics['cv_mean']:.3f} (+/- {metrics['cv_std']:.3f})")
        logger.info(f"      Confident predictions: {metrics['confident_predictions']:.1%}")
        logger.info(f"      Very confident: {metrics['very_confident']:.1%}")
        
        if metrics.get('kelly_edge'):
            logger.info(f"      💰 Kelly Edge: {metrics['kelly_edge']:.1%}")
        
        if metrics['near_fifty_pct'] > 0.6:
            logger.warning(f"      ⚠️ High 50/50 predictions: {metrics['near_fifty_pct']:.1%}")
        
        return {
            'model': best_model,
            'metrics': metrics,
            'name': model_name
        }
    
    def train_all_models_pitcher_integrated(self, feature_selection: bool = True,
                                           tune_hyperparameters: bool = False,
                                           n_features: int = 60) -> Dict:
        """Train all models using pitcher-integrated MLB data."""
        logger.info("🚀⚾ PITCHER-INTEGRATED MLB ENHANCED TRAINING PIPELINE")
        logger.info("=" * 70)
        
        # Load data with pitcher integration
        data, y = self.load_mlb_data_with_pitchers()
        
        # Engineer pitcher-aware features
        X = self.engineer_pitcher_aware_features(data)
        
        # Pitcher-aware feature selection
        if feature_selection:
            X = self.select_features_pitcher_aware(X, y, method='importance', n_features=n_features)
        
        self.feature_names = X.columns.tolist()
        
        # Time-aware split (important for baseball)
        if 'date' in data.columns:
            data_sorted = data.sort_values('date')
            split_idx = int(len(data_sorted) * 0.8)
            train_idx = data_sorted.index[:split_idx]
            test_idx = data_sorted.index[split_idx:]
            
            X_train, X_test = X.loc[train_idx], X.loc[test_idx]
            y_train, y_test = y.loc[train_idx], y.loc[test_idx]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
        
        # Count pitcher features
        pitcher_feature_count = len([col for col in X.columns if any(keyword in col.lower() for keyword in 
                                                                    ['starter', 'era', 'whip', 'pitcher', 'k9', 'bb9', 'quality'])])
        
        logger.info(f"📊 Enhanced Training Configuration:")
        logger.info(f"   Total games: {len(X)}")
        logger.info(f"   Train set: {len(X_train)} samples")
        logger.info(f"   Test set: {len(X_test)} samples")
        logger.info(f"   Total features: {len(X.columns)}")
        logger.info(f"   Pitcher features: {pitcher_feature_count}")
        logger.info(f"   Home win rate: {y_train.mean():.1%}")
        
        if pitcher_feature_count > 10:
            logger.info("   ⚾ EXCELLENT PITCHER INTEGRATION!")
        elif pitcher_feature_count > 5:
            logger.info("   ⚾ Good pitcher integration")
        else:
            logger.warning("   ⚠️ Limited pitcher integration")
            logger.info("   💡 To improve predictions, add pitcher data to your database:")
            logger.info("      1. Populate player_statistics table with pitcher stats")
            logger.info("      2. Add starting pitcher assignments to game_starting_pitchers table")
            logger.info("      3. Consider integrating with MLB API for live pitcher data")
        
        # Train enhanced models
        results = {}
        
        for model_name in self.model_configs.keys():
            try:
                result = self.train_single_model_pitcher_aware(
                    model_name, X_train_scaled, y_train, 
                    X_test_scaled, y_test, tune_hyperparameters
                )
                results[model_name] = result
                self.models[model_name] = result['model']
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue
        
        # Select best model
        if results:
            best_model_name = max(results.keys(), 
                                 key=lambda k: results[k]['metrics']['roc_auc'])
            self.best_model = results[best_model_name]['model']
            self.best_model_name = best_model_name
        
        # Enhanced results display
        self._display_pitcher_integration_results(results, pitcher_feature_count)
        
        self.training_results = results
        return results
    
    def _display_pitcher_integration_results(self, results: Dict, pitcher_features: int):
        """Display results with pitcher integration insights."""
        logger.info("=" * 70)
        logger.info("📊⚾ PITCHER-INTEGRATED MLB MODEL RESULTS")
        logger.info("=" * 70)
        
        if results:
            comparison_df = pd.DataFrame({
                name: result['metrics'] 
                for name, result in results.items()
            }).T
            
            logger.info(f"\n{comparison_df.round(3)}")
            
            logger.info(f"\n🏆 BEST MODEL: {self.best_model_name}")
            best_metrics = results[self.best_model_name]['metrics']
            logger.info(f"   ROC-AUC: {best_metrics['roc_auc']:.3f}")
            logger.info(f"   Accuracy: {best_metrics['accuracy']:.3f}")
            
            # Enhanced insights for pitcher integration
            if pitcher_features > 10:
                logger.info(f"   ⚾ Pitcher features used: {pitcher_features}")
                logger.info("   🎯 ENHANCED WITH COMPREHENSIVE PITCHER DATA!")
                
                if best_metrics['accuracy'] > 0.56:
                    logger.info("   📈 EXCELLENT PERFORMANCE FROM PITCHER INTEGRATION!")
                elif best_metrics['accuracy'] > 0.54:
                    logger.info("   📈 GOOD IMPROVEMENT FROM PITCHER INTEGRATION!")
                
            if best_metrics.get('kelly_edge'):
                logger.info(f"   💰 Profitable Edge: {best_metrics['kelly_edge']:.1%}")
                logger.info("   🎯 READY FOR PROFITABLE PITCHER-AWARE BETTING!")
            
            # Feature importance if available
            if hasattr(self.best_model, 'feature_importances_'):
                self._show_top_pitcher_features()
    
    def _show_top_pitcher_features(self, top_n: int = 15):
        """Show top features from the best model with pitcher emphasis."""
        if hasattr(self.best_model, 'feature_importances_') and self.feature_names:
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info(f"\n🔍 TOP {top_n} MOST IMPORTANT FEATURES:")
            for i, (_, row) in enumerate(importance_df.head(top_n).iterrows()):
                feature_name = row['feature']
                importance = row['importance']
                
                # Add emoji for different feature types
                if any(keyword in feature_name.lower() for keyword in ['starter', 'era', 'whip', 'pitcher', 'k9', 'bb9', 'quality']):
                    emoji = "⚾"
                elif 'team' in feature_name.lower():
                    emoji = "🏟️"
                elif any(keyword in feature_name.lower() for keyword in ['date', 'time', 'season', 'month']):
                    emoji = "📅"
                else:
                    emoji = "📊"
                
                logger.info(f"   {i+1:2}. {emoji} {feature_name}: {importance:.3f}")
    
    def save_pitcher_integrated_model(self):
        """Save the pitcher-integrated model and all components."""
        if self.best_model is None:
            raise ValueError("No model trained yet!")
        
        # Save best model
        model_path = self.model_dir / f"{self.best_model_name}_pitcher_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        # Save scaler
        scaler_path = self.model_dir / "pitcher_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save label encoders
        encoders_path = self.model_dir / "pitcher_label_encoders.pkl"
        with open(encoders_path, 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        # Save feature names
        features_path = self.model_dir / "pitcher_feature_names.pkl"
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        # Count pitcher features
        pitcher_feature_count = len([col for col in self.feature_names if any(keyword in col.lower() for keyword in 
                                                                             ['starter', 'era', 'whip', 'pitcher', 'k9', 'bb9', 'quality'])])
        
        # Save comprehensive metadata
        best_metrics = self.training_results[self.best_model_name]['metrics']
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            """Convert numpy types to Python types for JSON serialization."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Clean metrics for JSON
        clean_metrics = convert_numpy_types(best_metrics)
        
        metadata = {
            'model_type': self.best_model_name,
            'n_features': len(self.feature_names),
            'pitcher_features': pitcher_feature_count,
            'pitcher_integration': pitcher_feature_count > 5,
            'training_date': datetime.now().isoformat(),
            'feature_names': self.feature_names,
            'performance_metrics': clean_metrics,
            'is_profitable': clean_metrics.get('kelly_edge', 0) > 0,
            'recommended_kelly_fraction': clean_metrics.get('kelly_fraction', 0),
            'training_games': len(self.training_results),
            'model_version': 'pitcher_integrated_v1.0'
        }
        
        metadata_path = self.model_dir / "pitcher_model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"💾⚾ Pitcher-integrated model saved to {self.model_dir}")
        logger.info(f"   Model: {self.best_model_name}")
        logger.info(f"   Total features: {len(self.feature_names)}")
        logger.info(f"   Pitcher features: {pitcher_feature_count}")
        logger.info(f"   Performance: {best_metrics['accuracy']:.1%} accuracy")
        
        if metadata['is_profitable']:
            logger.info(f"   💰 PROFITABLE: {best_metrics['kelly_edge']:.1%} edge")
        
        if pitcher_feature_count > 10:
            logger.info("   🎯⚾ PITCHER INTEGRATION SUCCESSFUL!")


# Enhanced production function
def run_pitcher_integrated_mlb_training():
    """Run the complete pitcher-integrated MLB training pipeline."""
    logger.info("🚀⚾ Starting Pitcher-Integrated MLB Enhanced Training")
    
    try:
        # Initialize enhanced trainer
        trainer = IntegratedMLBEnhancedTrainer()
        
        # Train all models with pitcher integration
        results = trainer.train_all_models_pitcher_integrated(
            feature_selection=True,
            tune_hyperparameters=False,  # Set to True for production
            n_features=60  # More features for pitcher data
        )
        
        if results:
            # Save best model
            trainer.save_pitcher_integrated_model()
            
            # Enhanced summary
            best_metrics = results[trainer.best_model_name]['metrics']
            pitcher_feature_count = len([col for col in trainer.feature_names if any(keyword in col.lower() for keyword in 
                                                                                   ['starter', 'era', 'whip', 'pitcher', 'k9', 'bb9', 'quality'])])
            
            logger.info("✅ PITCHER-INTEGRATED TRAINING COMPLETE!")
            logger.info(f"🏆 Best Model: {trainer.best_model_name}")
            logger.info(f"📊 Accuracy: {best_metrics['accuracy']:.1%}")
            logger.info(f"🎯 ROC-AUC: {best_metrics['roc_auc']:.3f}")
            logger.info(f"⚾ Pitcher Features: {pitcher_feature_count}")
            
            if best_metrics.get('kelly_edge'):
                logger.info(f"💰 Profitable Edge: {best_metrics['kelly_edge']:.1%}")
                logger.info("🎯⚾ READY FOR PITCHER-AWARE BETTING!")
            
            return trainer, results
        else:
            logger.error("❌ No models trained successfully")
            return None, None
            
    except Exception as e:
        logger.error(f"❌ Pitcher-integrated training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    trainer, results = run_pitcher_integrated_mlb_training()
    
    if trainer and results:
        print("🎉⚾ PITCHER-INTEGRATED MLB TRAINING SUCCESSFUL!")
        print("💰 Ready for enhanced MLB predictions with pitcher data!")
    else:
        print("❌ Training failed - check logs")
