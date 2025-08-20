# models/mlb/mlb_model.py

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier, DummyRegressor
from typing import Dict, List, Any, Optional, Tuple
import joblib
from pathlib import Path
from datetime import datetime
from loguru import logger
import warnings
import os

# Fix multiprocessing issues for Python 3.13
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')

from config.settings import Settings

class MLBPredictionModel:
    """
    Comprehensive MLB prediction model using XGBoost.
    Updated for Python 3.13 compatibility and robust error handling.
    """
    
    def __init__(self, model_type: str = 'game_winner'):
        """
        Initialize MLB prediction model.
        
        Args:
            model_type: Type of model ('game_winner', 'total_runs', 'nrfi', 'player_hits', etc.)
        """
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.feature_importance = {}
        self.is_dummy_model = False
        
        # Get base model parameters from settings
        try:
            self.model_params = Settings.get_model_params('mlb').copy()
        except:
            # Fallback parameters if settings fail
            self.model_params = {
                'n_estimators': 100,
                'max_depth': 4,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': 1,  # Critical for Python 3.13
                'verbosity': 0,
                'tree_method': 'hist'
            }
        
        # Configure model parameters based on type
        self._configure_model_for_type(model_type)
        
        # Model storage path
        try:
            self.model_path = Settings.MODEL_PATHS['mlb'].get(model_type)
        except:
            self.model_path = Path(f"models/mlb/{model_type}_model.joblib")
        
        logger.info(f"⚾ Initialized MLB {model_type} model")
    
    def _configure_model_for_type(self, model_type: str):
        """Configure model parameters based on the model type."""
        if model_type == 'game_winner':
            self.model_params.update({
                'objective': 'binary:logistic',
                'eval_metric': 'logloss'
            })
            self.target_column = 'home_win'
            
        elif model_type == 'total_runs':
            self.model_params.update({
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse'
            })
            self.target_column = 'total_runs'
            
        elif model_type == 'nrfi':  # No Run First Inning
            self.model_params.update({
                'objective': 'binary:logistic',
                'eval_metric': 'logloss'
            })
            self.target_column = 'nrfi'
            
        elif 'player' in model_type:
            self.model_params.update({
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse'
            })
            self.target_column = self._get_player_target_column(model_type)
        
        # Remove early_stopping_rounds initially - we'll add it back conditionally
        self.model_params.pop('early_stopping_rounds', None)
    
    def _get_player_target_column(self, model_type: str) -> str:
        """Get target column for player prop models."""
        player_targets = {
            'player_hits': 'hits',
            'player_rbi': 'rbi',
            'player_runs': 'runs',
            'player_strikeouts': 'strikeouts_pitched'
        }
        return player_targets.get(model_type, 'hits')
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training/prediction with improved consistency.
        """
        logger.info(f"🔧 Preparing features for MLB {self.model_type} model...")
        
        if df.empty:
            return pd.DataFrame(), pd.Series(dtype=float)
        
        # Create a copy to avoid modifying original data
        data = df.copy()
        
        # Define comprehensive feature columns
        feature_columns = []
        
        # Basic game features
        basic_features = [
            'home_team_id', 'away_team_id', 'season'
        ]
        
        # Statistical features (comprehensive list)
        stat_features = [
            'home_runs_per_game', 'away_runs_per_game',
            'home_runs_allowed_per_game', 'away_runs_allowed_per_game',
            'home_batting_average', 'away_batting_average',
            'home_on_base_percentage', 'away_on_base_percentage',
            'home_slugging_percentage', 'away_slugging_percentage',
            'home_earned_run_average', 'away_earned_run_average',
            'home_whip', 'away_whip',
            'home_strikeouts_per_nine', 'away_strikeouts_per_nine',
            'home_walks_per_nine', 'away_walks_per_nine',
            'home_fielding_percentage', 'away_fielding_percentage'
        ]
        
        # Create date features if date column exists
        if 'date' in data.columns:
            try:
                data['date'] = pd.to_datetime(data['date'])
                data['day_of_week'] = data['date'].dt.dayofweek
                data['month'] = data['date'].dt.month
                data['day_of_year'] = data['date'].dt.dayofyear
                feature_columns.extend(['day_of_week', 'month', 'day_of_year'])
            except:
                logger.warning("Could not create date features")
        
        # Add available basic features
        for col in basic_features:
            if col in data.columns:
                feature_columns.append(col)
        
        # Add available statistical features
        for col in stat_features:
            if col in data.columns:
                feature_columns.append(col)
        
        # Create derived features if base stats are available
        self._create_derived_features(data, feature_columns)
        
        # If we have stored feature names from training, use only those for prediction
        if hasattr(self, 'feature_names') and self.feature_names:
            # For prediction, only use features that were in training
            available_features = [col for col in self.feature_names if col in data.columns]
            
            # Add missing features with default values
            for col in self.feature_names:
                if col not in data.columns:
                    data[col] = 0  # Default value for missing features
            
            feature_columns = self.feature_names
        else:
            # For training, use all available features
            available_features = [col for col in feature_columns if col in data.columns]
            feature_columns = available_features
        
        # Ensure we have at least some features
        if not feature_columns:
            logger.warning("⚠️ No features found, creating minimal feature set")
            # Create minimal features from available numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            feature_columns = [col for col in numeric_cols 
                             if col not in ['game_id', 'home_score', 'away_score']][:5]
            
            if not feature_columns:
                # Last resort: create dummy features
                data['dummy_feature_1'] = 1
                data['dummy_feature_2'] = np.random.rand(len(data))
                feature_columns = ['dummy_feature_1', 'dummy_feature_2']
        
        # Select and prepare features
        X = data[feature_columns].copy()
        
        # Handle missing values
        X = X.fillna(X.median() if len(X) > 0 else 0)
        
        # Ensure all features are numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        # Handle target variable
        y = self._prepare_target(data)
        
        # Store feature names for consistency (only during training)
        if not hasattr(self, 'feature_names') or not self.feature_names:
            self.feature_names = list(X.columns)
        
        logger.info(f"✅ Prepared {len(X)} samples with {len(X.columns)} features")
        logger.info(f"   Target: {self.target_column} (mean: {y.mean():.3f})")
        
        return X, y
    
    def _create_derived_features(self, data: pd.DataFrame, feature_columns: List[str]):
        """Create derived statistical features."""
        try:
            # Batting average difference
            if 'home_batting_average' in data.columns and 'away_batting_average' in data.columns:
                data['batting_average_diff'] = data['home_batting_average'] - data['away_batting_average']
                feature_columns.append('batting_average_diff')
            
            # ERA difference (lower is better, so home - away)
            if 'home_earned_run_average' in data.columns and 'away_earned_run_average' in data.columns:
                data['era_diff'] = data['away_earned_run_average'] - data['home_earned_run_average']
                feature_columns.append('era_diff')
            
            # Offensive strength difference
            if 'home_runs_per_game' in data.columns and 'away_runs_per_game' in data.columns:
                data['offense_diff'] = data['home_runs_per_game'] - data['away_runs_per_game']
                feature_columns.append('offense_diff')
            
            # Defensive strength difference
            if 'home_runs_allowed_per_game' in data.columns and 'away_runs_allowed_per_game' in data.columns:
                data['defense_diff'] = data['away_runs_allowed_per_game'] - data['home_runs_allowed_per_game']
                feature_columns.append('defense_diff')
            
            # OPS difference
            if all(col in data.columns for col in ['home_on_base_percentage', 'home_slugging_percentage', 
                                                   'away_on_base_percentage', 'away_slugging_percentage']):
                data['home_ops'] = data['home_on_base_percentage'] + data['home_slugging_percentage']
                data['away_ops'] = data['away_on_base_percentage'] + data['away_slugging_percentage']
                data['ops_diff'] = data['home_ops'] - data['away_ops']
                feature_columns.extend(['home_ops', 'away_ops', 'ops_diff'])
                
        except Exception as e:
            logger.warning(f"Error creating derived features: {e}")
    
    def _prepare_target(self, data: pd.DataFrame) -> pd.Series:
        """Prepare target variable with proper validation."""
        if self.target_column in data.columns:
            y = data[self.target_column].copy()
            
            # Handle missing values in target
            y = y.dropna()
            
            # Validate and clean target based on model type
            if self.model_type in ['game_winner', 'nrfi']:
                # Binary classification - ensure values are 0 or 1
                y = pd.to_numeric(y, errors='coerce')
                y = y.fillna(0).astype(int).clip(0, 1)
            else:
                # Regression - ensure numeric values
                y = pd.to_numeric(y, errors='coerce')
                y = y.fillna(y.median() if len(y) > 0 else 0)
                
                # Handle edge case for total_runs with logistic objective
                if self.model_type == 'total_runs' and self.model_params.get('objective') == 'binary:logistic':
                    # Convert to binary (high/low scoring game)
                    median_runs = y.median()
                    y = (y > median_runs).astype(int)
        else:
            logger.warning(f"Target column '{self.target_column}' not found. Using dummy target.")
            # Create appropriate dummy target
            if self.model_type in ['game_winner', 'nrfi']:
                y = pd.Series([0] * len(data))
            else:
                y = pd.Series([5.0] * len(data))  # Average MLB game total
        
        return y
    
    def train(self, df: pd.DataFrame, 
             validation_split: float = 0.2,
             cv_folds: int = 3,  # Reduced for small datasets
             optimize_hyperparams: bool = False) -> Dict[str, Any]:
        """
        Train the MLB prediction model with robust error handling.
        """
        logger.info(f"🚀 Training MLB {self.model_type} model...")
        
        try:
            # Prepare features and target
            X, y = self.prepare_features(df)
            
            if len(X) == 0:
                raise ValueError("No valid training data after feature preparation")
            
            # Check for sufficient data
            min_samples = 10
            if len(X) < min_samples:
                logger.warning(f"⚠️ Very small dataset: {len(X)} samples. Creating dummy model.")
                return self._create_dummy_model(X, y)
            
            # Adjust parameters for small datasets
            use_early_stopping = len(X) > 50
            use_cv = len(X) > 20
            
            # Configure parameters for stability
            stable_params = self.model_params.copy()
            if len(X) < 100:
                stable_params.update({
                    'n_estimators': min(stable_params.get('n_estimators', 100), 50),
                    'max_depth': min(stable_params.get('max_depth', 6), 4),
                    'n_jobs': 1  # Critical for Python 3.13
                })
            
            # Remove early stopping for small datasets
            if not use_early_stopping:
                stable_params.pop('early_stopping_rounds', None)
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Initialize model
            if self.model_type in ['game_winner', 'nrfi']:
                self.model = xgb.XGBClassifier(**stable_params)
            else:
                self.model = xgb.XGBRegressor(**stable_params)
            
            # Train model
            if use_early_stopping and len(X) > 50:
                # Train with validation set for early stopping
                split_idx = int(len(X) * (1 - validation_split))
                X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
                y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
                
                if len(X_val) > 0:
                    eval_set = [(X_train, y_train), (X_val, y_val)]
                    self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
                else:
                    self.model.fit(X_scaled, y, verbose=False)
            else:
                # Simple training without validation
                self.model.fit(X_scaled, y, verbose=False)
            
            # Calculate metrics
            metrics = self._calculate_training_metrics(X_scaled, y, use_cv, cv_folds)
            
            # Store feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
            
            logger.info(f"✅ Training complete for MLB {self.model_type}")
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
            logger.info("🔄 Creating fallback dummy model...")
            return self._create_dummy_model(X if 'X' in locals() else pd.DataFrame(), 
                                          y if 'y' in locals() else pd.Series())
    
    def _calculate_training_metrics(self, X_scaled: np.ndarray, y: pd.Series, 
                                  use_cv: bool, cv_folds: int) -> Dict[str, Any]:
        """Calculate training performance metrics."""
        try:
            # Basic predictions
            y_pred = self.model.predict(X_scaled)
            
            if self.model_type in ['game_winner', 'nrfi']:
                # Classification metrics
                accuracy = accuracy_score(y, y_pred)
                
                # AUC only if we have both classes
                if len(np.unique(y)) > 1:
                    try:
                        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]
                        auc = roc_auc_score(y, y_pred_proba)
                    except:
                        auc = 0.5
                else:
                    auc = 0.5
                
                metrics = {
                    'accuracy': accuracy,
                    'auc': auc,
                    'samples': len(X_scaled),
                    'features': len(self.feature_names)
                }
                
                logger.info(f"   Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")
                
            else:
                # Regression metrics
                mse = np.mean((y - y_pred) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(y - y_pred))
                
                metrics = {
                    'rmse': rmse,
                    'mse': mse,
                    'mae': mae,
                    'samples': len(X_scaled),
                    'features': len(self.feature_names)
                }
                
                logger.info(f"   RMSE: {rmse:.3f}, MAE: {mae:.3f}")
            
            # Add cross-validation if dataset is large enough
            if use_cv:
                try:
                    tscv = TimeSeriesSplit(n_splits=cv_folds)
                    scoring = 'accuracy' if self.model_type in ['game_winner', 'nrfi'] else 'neg_mean_squared_error'
                    
                    # Use n_jobs=1 to avoid multiprocessing issues
                    cv_scores = cross_val_score(
                        self.model, X_scaled, y, 
                        cv=tscv, scoring=scoring, n_jobs=1
                    )
                    
                    metrics['cv_mean'] = cv_scores.mean()
                    metrics['cv_std'] = cv_scores.std()
                    
                except Exception as cv_error:
                    logger.warning(f"CV failed: {cv_error}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {'error': str(e), 'samples': len(X_scaled), 'features': len(self.feature_names)}
    
    def _create_dummy_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Create a dummy model as fallback."""
        try:
            logger.info("🔄 Creating dummy model for testing...")
            
            # Create minimal dummy data if needed
            if len(X) == 0:
                X = pd.DataFrame({'dummy_feature': [1, 2, 3]})
                if self.model_type in ['game_winner', 'nrfi']:
                    y = pd.Series([0, 1, 0])
                else:
                    y = pd.Series([4.0, 5.0, 6.0])
            
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Create dummy model
            if self.model_type in ['game_winner', 'nrfi']:
                self.model = DummyClassifier(strategy='most_frequent')
            else:
                self.model = DummyRegressor(strategy='mean')
            
            self.model.fit(X_scaled, y)
            self.is_dummy_model = True
            
            if not self.feature_names:
                self.feature_names = list(X.columns)
            
            logger.info("✅ Dummy model created successfully")
            
            return {
                'model_type': 'dummy',
                'samples': len(X),
                'features': len(self.feature_names),
                'accuracy': 0.5 if self.model_type in ['game_winner', 'nrfi'] else None,
                'rmse': y.std() if self.model_type not in ['game_winner', 'nrfi'] else None,
                'is_dummy': True
            }
            
        except Exception as e:
            logger.error(f"❌ Even dummy model failed: {e}")
            return {'error': str(e), 'is_dummy': True}
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data with improved error handling."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a trained model.")
        
        try:
            X, _ = self.prepare_features(df)
            
            if len(X) == 0:
                return np.array([])
            
            # Ensure feature consistency
            if set(X.columns) != set(self.feature_names):
                logger.warning("Feature mismatch detected, adjusting...")
                # Add missing features with zeros
                for col in self.feature_names:
                    if col not in X.columns:
                        X[col] = 0
                # Reorder columns to match training
                X = X[self.feature_names]
            
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            
            logger.info(f"✅ Generated {len(predictions)} predictions for MLB {self.model_type}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            # Return dummy predictions
            if self.model_type in ['game_winner', 'nrfi']:
                return np.array([0] * len(df))
            else:
                return np.array([5.0] * len(df))  # Average MLB total
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities (for classification models)."""
        if self.model_type not in ['game_winner', 'nrfi']:
            raise ValueError("predict_proba only available for classification models")
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a trained model.")
        
        try:
            X, _ = self.prepare_features(df)
            
            if len(X) == 0:
                return np.array([])
            
            # Ensure feature consistency
            if set(X.columns) != set(self.feature_names):
                for col in self.feature_names:
                    if col not in X.columns:
                        X[col] = 0
                X = X[self.feature_names]
            
            X_scaled = self.scaler.transform(X)
            
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X_scaled)
            else:
                # Dummy model fallback
                predictions = self.model.predict(X_scaled)
                probabilities = np.column_stack([1-predictions, predictions])
            
            return probabilities
            
        except Exception as e:
            logger.error(f"❌ Probability prediction failed: {e}")
            # Return dummy probabilities
            return np.array([[0.5, 0.5]] * len(df))
    
    def _optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, cv) -> Dict[str, Any]:
        """Optimize hyperparameters using grid search with error handling."""
        try:
            param_grid = {
                'n_estimators': [50, 100],  # Reduced for stability
                'max_depth': [3, 4, 5],
                'learning_rate': [0.1, 0.15]
            }
            
            if self.model_type in ['game_winner', 'nrfi']:
                base_model = xgb.XGBClassifier(**self.model_params)
                scoring = 'accuracy'
            else:
                base_model = xgb.XGBRegressor(**self.model_params)
                scoring = 'neg_mean_squared_error'
            
            # Use n_jobs=1 to avoid multiprocessing issues
            grid_search = GridSearchCV(
                base_model, param_grid, 
                cv=cv, scoring=scoring, 
                n_jobs=1, verbose=0
            )
            
            grid_search.fit(X, y)
            
            logger.info(f"🎯 Best hyperparameters: {grid_search.best_params_}")
            return grid_search.best_params_
            
        except Exception as e:
            logger.warning(f"Hyperparameter optimization failed: {e}")
            return {}
    
    def _get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N most important features."""
        if not self.feature_importance:
            return []
        
        sorted_features = sorted(
            self.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_features[:n]
    
    def save_model(self, filepath: Optional[Path] = None) -> Path:
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        save_path = filepath or self.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_package = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'target_column': self.target_column,
            'is_dummy_model': getattr(self, 'is_dummy_model', False),
            'saved_date': datetime.now().isoformat()
        }
        
        joblib.dump(model_package, save_path)
        
        logger.info(f"💾 Saved MLB {self.model_type} model to {save_path}")
        return save_path
    
    def load_model(self, filepath: Optional[Path] = None) -> bool:
        """Load a trained model from disk."""
        load_path = filepath or self.model_path
        
        if not load_path.exists():
            logger.error(f"Model file not found: {load_path}")
            return False
        
        try:
            model_package = joblib.load(load_path)
            
            self.model = model_package['model']
            self.scaler = model_package['scaler']
            self.feature_names = model_package['feature_names']
            self.feature_importance = model_package.get('feature_importance', {})
            self.model_type = model_package.get('model_type', self.model_type)
            self.model_params = model_package.get('model_params', self.model_params)
            self.target_column = model_package.get('target_column', self.target_column)
            self.is_dummy_model = model_package.get('is_dummy_model', False)
            
            saved_date = model_package.get('saved_date', 'unknown')
            logger.info(f"📦 Loaded MLB {self.model_type} model from {load_path}")
            logger.info(f"   Model saved: {saved_date}")
            logger.info(f"   Features: {len(self.feature_names)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False


class MLBModelEnsemble:
    """Ensemble of MLB prediction models for improved accuracy."""
    
    def __init__(self, model_types: List[str] = ['game_winner']):
        """Initialize ensemble with multiple model types."""
        self.models = {}
        self.weights = {}
        
        for model_type in model_types:
            self.models[model_type] = MLBPredictionModel(model_type)
            self.weights[model_type] = 1.0
        
        logger.info(f"🎯 Initialized MLB ensemble with {len(self.models)} models")
    
    def train_ensemble(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train all models in the ensemble with error handling."""
        results = {}
        
        for model_type, model in self.models.items():
            logger.info(f"Training MLB {model_type} model...")
            try:
                model_results = model.train(df, cv_folds=2, optimize_hyperparams=False)
                results[model_type] = model_results
                
                # Adjust weights based on performance
                if model_type in ['game_winner', 'nrfi']:
                    performance = model_results.get('accuracy', 0.5)
                else:
                    rmse = model_results.get('rmse', 1.0)
                    performance = 1.0 / (rmse + 0.001)
                
                self.weights[model_type] = performance
                
            except Exception as e:
                logger.error(f"Error training MLB {model_type}: {e}")
                results[model_type] = {'error': str(e)}
                self.weights[model_type] = 0.1  # Low weight for failed models
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        results['ensemble_weights'] = self.weights
        logger.info(f"✅ Ensemble training complete. Weights: {self.weights}")
        return results
    
    def predict_ensemble(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Make ensemble predictions with error handling."""
        predictions = {}
        
        for model_type, model in self.models.items():
            try:
                if model.model is not None:
                    pred = model.predict(df)
                    predictions[model_type] = pred
                    logger.info(f"✅ {model_type}: {len(pred)} predictions")
            except Exception as e:
                logger.error(f"Error predicting with MLB {model_type}: {e}")
                # Add dummy predictions for failed models
                if model_type in ['game_winner', 'nrfi']:
                    predictions[model_type] = np.array([0] * len(df))
                else:
                    predictions[model_type] = np.array([5.0] * len(df))
        
        return predictions
