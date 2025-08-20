# models/mlb/mlb_enhanced_integrated.py
"""
Enhanced MLB trainer integrated with your working data pipeline.
Combines your multi-model approach with our working 15K+ game dataset.
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

class IntegratedMLBEnhancedTrainer:
    """
    Enhanced multi-model trainer integrated with your working MLB data pipeline.
    Uses your 15K+ games with advanced model comparison and selection.
    """
    
    def __init__(self, model_dir: Path = Path('models/mlb/enhanced'), 
                 random_state: int = 42):
        """Initialize the integrated enhanced trainer."""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        
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
        
        # Model configurations optimized for your MLB data
        self.model_configs = self._get_mlb_optimized_configs()
        
        logger.info("🎯 Initialized Enhanced MLB Trainer")
    
    def _get_mlb_optimized_configs(self) -> Dict:
        """Get model configurations optimized for MLB prediction."""
        return {
            'xgboost_mlb': {
                'model_class': xgb.XGBClassifier,
                'param_grid': {
                    'n_estimators': [200, 300, 500],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'subsample': [0.8, 0.9],
                    'colsample_bytree': [0.8, 0.9],
                    'scale_pos_weight': [0.85, 0.9, 0.95],  # Adjust for 52.9% home win rate
                    'random_state': [self.random_state]
                },
                'quick_params': {
                    'n_estimators': 300,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'scale_pos_weight': 0.89,  # For 52.9% home win rate
                    'random_state': self.random_state,
                    'eval_metric': 'logloss',
                    'use_label_encoder': False
                }
            },
            
            'lightgbm_mlb': {
                'model_class': lgb.LGBMClassifier,
                'param_grid': {
                    'n_estimators': [200, 300, 500],
                    'num_leaves': [31, 50, 100],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'feature_fraction': [0.8, 0.9],
                    'bagging_fraction': [0.8, 0.9],
                    'min_child_samples': [20, 30, 50],
                    'random_state': [self.random_state]
                },
                'quick_params': {
                    'n_estimators': 300,
                    'num_leaves': 50,
                    'learning_rate': 0.1,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'min_child_samples': 30,
                    'random_state': self.random_state,
                    'verbosity': -1
                }
            },
            
            'random_forest_mlb': {
                'model_class': RandomForestClassifier,
                'param_grid': {
                    'n_estimators': [300, 500, 1000],
                    'max_depth': [15, 20, 25],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2'],
                    'class_weight': ['balanced', 'balanced_subsample'],
                    'random_state': [self.random_state]
                },
                'quick_params': {
                    'n_estimators': 500,
                    'max_depth': 20,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'max_features': 'sqrt',
                    'class_weight': 'balanced',
                    'random_state': self.random_state,
                    'n_jobs': -1
                }
            },
            
            'gradient_boosting_mlb': {
                'model_class': GradientBoostingClassifier,
                'param_grid': {
                    'n_estimators': [200, 300, 500],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'subsample': [0.8, 0.9],
                    'min_samples_split': [5, 10, 20],
                    'random_state': [self.random_state]
                },
                'quick_params': {
                    'n_estimators': 300,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'min_samples_split': 10,
                    'random_state': self.random_state
                }
            },
            
            'neural_network_mlb': {
                'model_class': MLPClassifier,
                'param_grid': {
                    'hidden_layer_sizes': [(100, 50), (150, 75), (200, 100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.001, 0.01, 0.1],
                    'learning_rate_init': [0.001, 0.01],
                    'max_iter': [1000, 2000],
                    'random_state': [self.random_state]
                },
                'quick_params': {
                    'hidden_layer_sizes': (150, 75),
                    'activation': 'relu',
                    'alpha': 0.01,
                    'learning_rate_init': 0.001,
                    'max_iter': 2000,
                    'random_state': self.random_state,
                    'early_stopping': True,
                    'validation_fraction': 0.1
                }
            },
            
            'logistic_regression_mlb': {
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
                    'max_iter': 2000
                }
            }
        }
    
    def load_mlb_data_from_database(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare MLB data from your working database."""
        logger.info("📊 Loading MLB data from your working database...")
        
        try:
            with sqlite3.connect(Settings.MLB_DB_PATH) as conn:
                # Load finished games with comprehensive data
                query = """
                SELECT 
                    g.*,
                    hts.win_percentage as home_team_win_pct,
                    hts.runs_per_game as home_runs_per_game,
                    hts.runs_allowed_per_game as home_runs_allowed,
                    ats.win_percentage as away_team_win_pct,
                    ats.runs_per_game as away_runs_per_game,
                    ats.runs_allowed_per_game as away_runs_allowed
                FROM games g
                LEFT JOIN team_statistics hts ON g.home_team_id = hts.team_id AND g.season = hts.season
                LEFT JOIN team_statistics ats ON g.away_team_id = ats.team_id AND g.season = ats.season
                WHERE g.status = 'Finished' 
                AND g.home_score IS NOT NULL 
                AND g.away_score IS NOT NULL
                ORDER BY g.date
                """
                
                data = pd.read_sql_query(query, conn)
                
                if data.empty:
                    raise ValueError("No finished games found in database!")
                
                logger.info(f"✅ Loaded {len(data)} finished games")
                
                # Create target variable
                y = (data['home_score'] > data['away_score']).astype(int)
                
                logger.info(f"📊 Home team win rate: {y.mean():.1%}")
                
                return data, y
                
        except Exception as e:
            logger.error(f"❌ Error loading MLB data: {e}")
            raise
    
    def engineer_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer advanced features for MLB prediction."""
        logger.info("🔧 Engineering advanced MLB features...")
        
        feature_data = data.copy()
        
        # === BASIC GAME FEATURES ===
        features = []
        
        # Team IDs (encode them)
        if 'home_team_id' in feature_data.columns:
            features.append('home_team_id')
        if 'away_team_id' in feature_data.columns:
            features.append('away_team_id')
        
        # Season
        if 'season' in feature_data.columns:
            features.append('season')
        
        # === DERIVED FEATURES ===
        # Total runs and run differential
        if 'home_score' in feature_data.columns and 'away_score' in feature_data.columns:
            feature_data['total_runs'] = feature_data['home_score'] + feature_data['away_score']
            feature_data['run_differential'] = feature_data['home_score'] - feature_data['away_score']
            feature_data['high_scoring'] = (feature_data['total_runs'] > 10).astype(int)
            feature_data['close_game'] = (abs(feature_data['run_differential']) <= 2).astype(int)
            features.extend(['total_runs', 'run_differential', 'high_scoring', 'close_game'])
        
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
        
        # === TEAM PERFORMANCE FEATURES ===
        # Win percentages
        if 'home_team_win_pct' in feature_data.columns:
            features.append('home_team_win_pct')
        if 'away_team_win_pct' in feature_data.columns:
            features.append('away_team_win_pct')
        
        # Win percentage differential
        if all(col in feature_data.columns for col in ['home_team_win_pct', 'away_team_win_pct']):
            feature_data['win_pct_diff'] = feature_data['home_team_win_pct'] - feature_data['away_team_win_pct']
            features.append('win_pct_diff')
        
        # Offensive metrics
        if 'home_runs_per_game' in feature_data.columns:
            features.append('home_runs_per_game')
        if 'away_runs_per_game' in feature_data.columns:
            features.append('away_runs_per_game')
        
        # Defensive metrics
        if 'home_runs_allowed' in feature_data.columns:
            features.append('home_runs_allowed')
        if 'away_runs_allowed' in feature_data.columns:
            features.append('away_runs_allowed')
        
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
        
        # === FEATURE SELECTION ===
        # Only keep features that exist and have reasonable data
        available_features = []
        for feat in features:
            if feat in feature_data.columns:
                missing_pct = feature_data[feat].isnull().sum() / len(feature_data)
                if missing_pct < 0.5:  # Keep features with <50% missing
                    available_features.append(feat)
        
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
                        # Handle unseen categories
                        X[feat] = 0
        
        # Handle missing values
        for col in X.columns:
            if X[col].dtype in ['object']:
                X[col] = X[col].fillna('unknown')
            else:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                X[col] = X[col].fillna(X[col].median())
        
        logger.info(f"✅ Engineered {len(X.columns)} features for {len(X)} games")
        
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
    
    def select_features_advanced(self, X: pd.DataFrame, y: pd.Series, 
                                method: str = 'importance', 
                                n_features: int = 50) -> pd.DataFrame:
        """Advanced feature selection optimized for MLB data."""
        logger.info(f"🔍 Selecting top {n_features} features using {method} method...")
        
        if len(X.columns) <= n_features:
            logger.info(f"   Already have {len(X.columns)} features (less than {n_features})")
            return X
        
        if method == 'importance':
            # Use XGBoost feature importance (better for baseball)
            model = xgb.XGBClassifier(
                n_estimators=100, 
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            )
            model.fit(X, y)
            
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            selected_features = importance_df.head(n_features)['feature'].tolist()
            
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
        
        logger.info(f"   ✅ Selected {len(selected_features)} features")
        
        # Show top features
        logger.info(f"   📊 Top 10 selected features:")
        for i, feat in enumerate(selected_features[:10]):
            logger.info(f"      {i+1}. {feat}")
        
        return X[selected_features]
    
    def train_single_model_mlb(self, model_name: str, X_train: pd.DataFrame, 
                              y_train: pd.Series, X_test: pd.DataFrame, 
                              y_test: pd.Series, tune_hyperparameters: bool = False) -> Dict:
        """Train a single model optimized for MLB prediction."""
        logger.info(f"🎯 Training {model_name} for MLB...")
        
        config = self.model_configs[model_name]
        
        if tune_hyperparameters and 'param_grid' in config:
            logger.info("   🔧 Tuning hyperparameters...")
            model = config['model_class']()
            
            # Use RandomizedSearchCV for faster tuning on large dataset
            search = RandomizedSearchCV(
                model,
                config['param_grid'],
                n_iter=15,  # Balanced speed vs thoroughness
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=self.random_state
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            logger.info(f"   Best params: {search.best_params_}")
        else:
            # Use optimized quick parameters
            best_model = config['model_class'](**config['quick_params'])
            best_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Calculate comprehensive metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'predictions_std': np.std(y_pred_proba),
            'predictions_mean': np.mean(y_pred_proba),
            'near_fifty_pct': np.mean((y_pred_proba > 0.45) & (y_pred_proba < 0.55)),
            'confident_predictions': np.mean((y_pred_proba < 0.4) | (y_pred_proba > 0.6))
        }
        
        # Cross-validation for robustness
        try:
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='roc_auc')
            metrics['cv_mean'] = np.mean(cv_scores)
            metrics['cv_std'] = np.std(cv_scores)
        except:
            metrics['cv_mean'] = metrics['roc_auc']
            metrics['cv_std'] = 0.0
        
        # Kelly criterion edge for betting
        if metrics['accuracy'] > 0.524:  # Profitable threshold
            edge = metrics['accuracy'] - 0.5
            kelly_fraction = (edge * 2) - 1  # Simplified Kelly
            metrics['kelly_edge'] = edge
            metrics['kelly_fraction'] = kelly_fraction
        
        logger.info(f"   ✅ {model_name} Results:")
        logger.info(f"      Accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"      ROC-AUC: {metrics['roc_auc']:.3f}")
        logger.info(f"      CV Score: {metrics['cv_mean']:.3f} (+/- {metrics['cv_std']:.3f})")
        logger.info(f"      Confident predictions: {metrics['confident_predictions']:.1%}")
        
        if metrics.get('kelly_edge'):
            logger.info(f"      💰 Kelly Edge: {metrics['kelly_edge']:.1%}")
        
        if metrics['near_fifty_pct'] > 0.6:
            logger.warning(f"      ⚠️ High 50/50 predictions: {metrics['near_fifty_pct']:.1%}")
        
        return {
            'model': best_model,
            'metrics': metrics,
            'name': model_name
        }
    
    def train_all_models_integrated(self, feature_selection: bool = True,
                                   tune_hyperparameters: bool = False,
                                   n_features: int = 50) -> Dict:
        """Train all models using your integrated MLB data."""
        logger.info("🚀 INTEGRATED MLB ENHANCED TRAINING PIPELINE")
        logger.info("=" * 60)
        
        # Load data from your working database
        data, y = self.load_mlb_data_from_database()
        
        # Engineer advanced features
        X = self.engineer_advanced_features(data)
        
        # Feature selection
        if feature_selection:
            X = self.select_features_advanced(X, y, method='importance', n_features=n_features)
        
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
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
        
        logger.info(f"📊 Training Configuration:")
        logger.info(f"   Total games: {len(X)}")
        logger.info(f"   Train set: {len(X_train)} samples")
        logger.info(f"   Test set: {len(X_test)} samples")
        logger.info(f"   Features: {len(X.columns)}")
        logger.info(f"   Home win rate: {y_train.mean():.1%}")
        
        # Train all models
        results = {}
        
        for model_name in self.model_configs.keys():
            try:
                result = self.train_single_model_mlb(
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
        
        # Display results
        logger.info("=" * 60)
        logger.info("📊 MLB MODEL COMPARISON RESULTS")
        logger.info("=" * 60)
        
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
            
            if best_metrics.get('kelly_edge'):
                logger.info(f"   💰 Profitable Edge: {best_metrics['kelly_edge']:.1%}")
                logger.info("   🎯 READY FOR PROFITABLE BETTING!")
        
        self.training_results = results
        return results
    
    def save_integrated_model(self):
        """Save the integrated model and all components."""
        if self.best_model is None:
            raise ValueError("No model trained yet!")
        
        # Save best model
        model_path = self.model_dir / f"{self.best_model_name}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        # Save scaler
        scaler_path = self.model_dir / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save label encoders
        encoders_path = self.model_dir / "label_encoders.pkl"
        with open(encoders_path, 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        # Save feature names
        features_path = self.model_dir / "feature_names.pkl"
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        # Save comprehensive metadata
        best_metrics = self.training_results[self.best_model_name]['metrics']
        metadata = {
            'model_type': self.best_model_name,
            'n_features': len(self.feature_names),
            'training_date': datetime.now().isoformat(),
            'feature_names': self.feature_names,
            'performance_metrics': best_metrics,
            'is_profitable': best_metrics.get('kelly_edge', 0) > 0,
            'recommended_kelly_fraction': best_metrics.get('kelly_fraction', 0),
            'training_games': best_metrics.get('training_samples', 0)
        }
        
        metadata_path = self.model_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"💾 Integrated model saved to {self.model_dir}")
        logger.info(f"   Model: {self.best_model_name}")
        logger.info(f"   Features: {len(self.feature_names)}")
        logger.info(f"   Performance: {best_metrics['accuracy']:.1%} accuracy")
        
        if metadata['is_profitable']:
            logger.info(f"   💰 PROFITABLE: {best_metrics['kelly_edge']:.1%} edge")


# Production integration function
def run_integrated_mlb_training():
    """Run the complete integrated MLB training pipeline."""
    logger.info("🚀 Starting Integrated MLB Enhanced Training")
    
    try:
        # Initialize trainer
        trainer = IntegratedMLBEnhancedTrainer()
        
        # Train all models
        results = trainer.train_all_models_integrated(
            feature_selection=True,
            tune_hyperparameters=False,  # Set to True for production
            n_features=50
        )
        
        if results:
            # Save best model
            trainer.save_integrated_model()
            
            # Summary
            best_metrics = results[trainer.best_model_name]['metrics']
            
            logger.info("✅ INTEGRATED TRAINING COMPLETE!")
            logger.info(f"🏆 Best Model: {trainer.best_model_name}")
            logger.info(f"📊 Accuracy: {best_metrics['accuracy']:.1%}")
            logger.info(f"🎯 ROC-AUC: {best_metrics['roc_auc']:.3f}")
            
            if best_metrics.get('kelly_edge'):
                logger.info(f"💰 Profitable Edge: {best_metrics['kelly_edge']:.1%}")
                logger.info("🎯 READY FOR LIVE BETTING!")
            
            return trainer, results
        else:
            logger.error("❌ No models trained successfully")
            return None, None
            
    except Exception as e:
        logger.error(f"❌ Integrated training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    trainer, results = run_integrated_mlb_training()
    
    if trainer and results:
        print("🎉 INTEGRATED MLB ENHANCED TRAINING SUCCESSFUL!")
        print("💰 Ready for profitable MLB predictions!")
    else:
        print("❌ Training failed - check logs")
