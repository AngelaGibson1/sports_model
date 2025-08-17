# models/nfl/nfl_model.py

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any, Optional, Tuple
import joblib
from pathlib import Path
from datetime import datetime
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from config.settings import Settings

class NFLPredictionModel:
    """
    Comprehensive NFL prediction model using XGBoost.
    Supports game outcome, total points, and player prop predictions.
    """
    
    def __init__(self, model_type: str = 'game_winner'):
        """
        Initialize NFL prediction model.
        
        Args:
            model_type: Type of model ('game_winner', 'total_points', 'qb_passing_yards', etc.)
        """
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.feature_importance = {}
        
        # Get model parameters from settings
        self.model_params = Settings.get_model_params('nfl')
        
        # Adjust parameters based on model type
        if model_type == 'game_winner':
            self.model_params['objective'] = 'binary:logistic'
            self.model_params['eval_metric'] = 'logloss'
            self.target_column = 'home_win'
        elif model_type == 'total_points':
            self.model_params['objective'] = 'reg:squarederror'
            self.model_params['eval_metric'] = 'rmse'
            self.target_column = 'total_points'
        elif 'qb' in model_type:
            self.model_params['objective'] = 'reg:squarederror'
            self.model_params['eval_metric'] = 'rmse'
            self.target_column = self._get_player_target_column(model_type)
        
        # Model storage path
        self.model_path = Settings.MODEL_PATHS['nfl'].get(model_type)
        
        logger.info(f"🏈 Initialized NFL {model_type} model")
    
    def _get_player_target_column(self, model_type: str) -> str:
        """Get target column for player prop models."""
        player_targets = {
            'qb_passing_yards': 'passing_yards',
            'qb_touchdowns': 'passing_touchdowns',
            'rb_rushing_yards': 'rushing_yards',
            'wr_receiving_yards': 'receiving_yards'
        }
        return player_targets.get(model_type, 'passing_yards')
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training/prediction.
        
        Args:
            df: Input DataFrame with engineered features
            
        Returns:
            Tuple of (features_df, target_series)
        """
        logger.info(f"🔧 Preparing features for NFL {self.model_type} model...")
        
        # Remove non-feature columns
        exclude_columns = [
            'game_id', 'date', 'Date', 'season', 'week', 'home_team_id', 'away_team_id',
            'home_team_name', 'away_team_name', 'home_score', 'away_score',
            'status', 'venue', 'city', 'created_at', 'updated_at',
            'player_id', 'player_name', 'team_id', 'season_type'
        ]
        
        # Also exclude target columns we're not predicting
        target_columns = ['home_win', 'total_points', 'passing_yards', 'passing_touchdowns', 'rushing_yards', 'receiving_yards']
        exclude_columns.extend([col for col in target_columns if col != self.target_column])
        
        # Select feature columns
        feature_columns = [col for col in df.columns 
                          if col not in exclude_columns and df[col].dtype in ['int64', 'float64']]
        
        # Prepare features
        X = df[feature_columns].copy()
        
        # Handle missing values in features
        X = X.fillna(X.median())
        
        # Prepare target
        if self.target_column in df.columns:
            y = df[self.target_column].copy()
            
            # Handle missing values in target
            valid_idx = y.notna()
            X = X[valid_idx]
            y = y[valid_idx]
        else:
            logger.warning(f"Target column '{self.target_column}' not found. Using dummy target.")
            y = pd.Series([0] * len(X))
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        logger.info(f"✅ Prepared {len(X)} samples with {len(self.feature_names)} features")
        logger.info(f"   Target: {self.target_column} (mean: {y.mean():.3f})")
        
        return X, y
    
    def train(self, df: pd.DataFrame, 
             validation_split: float = 0.2,
             cv_folds: int = 5,
             optimize_hyperparams: bool = False) -> Dict[str, Any]:
        """
        Train the NFL prediction model.
        
        Args:
            df: Training DataFrame with features and target
            validation_split: Fraction of data to use for validation
            cv_folds: Number of cross-validation folds
            optimize_hyperparams: Whether to perform hyperparameter optimization
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"🚀 Training NFL {self.model_type} model...")
        
        # Prepare features and target
        X, y = self.prepare_features(df)
        
        if len(X) == 0:
            raise ValueError("No valid training data after feature preparation")
        
        # Time series split for proper validation (respects temporal order)
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Hyperparameter optimization if requested
        if optimize_hyperparams:
            logger.info("🔍 Optimizing hyperparameters...")
            best_params = self._optimize_hyperparameters(X, y, tscv)
            self.model_params.update(best_params)
        
        # Initialize model
        if self.model_type == 'game_winner':
            self.model = xgb.XGBClassifier(**self.model_params)
        else:
            self.model = xgb.XGBRegressor(**self.model_params)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model with early stopping
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            self.model, X_scaled, y, 
            cv=tscv, 
            scoring='accuracy' if self.model_type == 'game_winner' else 'neg_mean_squared_error',
            n_jobs=-1
        )
        
        # Calculate feature importance
        self.feature_importance = dict(zip(
            self.feature_names, 
            self.model.feature_importances_
        ))
        
        # Validation predictions and metrics
        if self.model_type == 'game_winner':
            val_pred = self.model.predict(X_val)
            val_pred_proba = self.model.predict_proba(X_val)[:, 1]
            val_accuracy = accuracy_score(y_val, val_pred)
            val_auc = roc_auc_score(y_val, val_pred_proba)
            
            performance_metrics = {
                'validation_accuracy': val_accuracy,
                'validation_auc': val_auc,
                'cv_mean_accuracy': cv_scores.mean(),
                'cv_std_accuracy': cv_scores.std()
            }
        else:
            val_pred = self.model.predict(X_val)
            val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
            val_mae = np.mean(np.abs(y_val - val_pred))
            
            performance_metrics = {
                'validation_rmse': val_rmse,
                'validation_mae': val_mae,
                'cv_mean_rmse': np.sqrt(-cv_scores.mean()),
                'cv_std_rmse': np.sqrt(cv_scores.std())
            }
        
        training_results = {
            'model_type': self.model_type,
            'training_samples': len(X),
            'features_count': len(self.feature_names),
            'cv_folds': cv_folds,
            **performance_metrics,
            'top_features': self._get_top_features(10),
            'training_date': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Training complete for NFL {self.model_type}")
        for metric, value in performance_metrics.items():
            logger.info(f"   {metric}: {value:.4f}")
        
        return training_results
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a trained model.")
        
        X, _ = self.prepare_features(df)
        
        if len(X) == 0:
            return np.array([])
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        logger.info(f"✅ Generated {len(predictions)} predictions for NFL {self.model_type}")
        
        return predictions
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities (for classification models)."""
        if self.model_type != 'game_winner':
            raise ValueError("predict_proba only available for classification models")
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a trained model.")
        
        X, _ = self.prepare_features(df)
        
        if len(X) == 0:
            return np.array([])
        
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)
        
        return probabilities
    
    def _optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, cv) -> Dict[str, Any]:
        """Optimize hyperparameters using grid search."""
        param_grid = {
            'n_estimators': [150, 200, 250],
            'max_depth': [5, 7, 9],
            'learning_rate': [0.1, 0.15, 0.2],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
        
        if self.model_type == 'game_winner':
            base_model = xgb.XGBClassifier(**self.model_params)
            scoring = 'accuracy'
        else:
            base_model = xgb.XGBRegressor(**self.model_params)
            scoring = 'neg_mean_squared_error'
        
        grid_search = GridSearchCV(
            base_model, param_grid, 
            cv=cv, scoring=scoring, 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        logger.info(f"🎯 Best hyperparameters: {grid_search.best_params_}")
        logger.info(f"🎯 Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_
    
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
            'saved_date': datetime.now().isoformat()
        }
        
        joblib.dump(model_package, save_path)
        
        logger.info(f"💾 Saved NFL {self.model_type} model to {save_path}")
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
            
            saved_date = model_package.get('saved_date', 'unknown')
            logger.info(f"📦 Loaded NFL {self.model_type} model from {load_path}")
            logger.info(f"   Model saved: {saved_date}")
            logger.info(f"   Features: {len(self.feature_names)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False


class NFLModelEnsemble:
    """Ensemble of NFL prediction models for improved accuracy."""
    
    def __init__(self, model_types: List[str] = ['game_winner']):
        """Initialize ensemble with multiple model types."""
        self.models = {}
        self.weights = {}
        
        for model_type in model_types:
            self.models[model_type] = NFLPredictionModel(model_type)
            self.weights[model_type] = 1.0
        
        logger.info(f"🎯 Initialized NFL ensemble with {len(self.models)} models")
    
    def train_ensemble(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train all models in the ensemble."""
        results = {}
        
        for model_type, model in self.models.items():
            logger.info(f"Training NFL {model_type} model...")
            try:
                model_results = model.train(df)
                results[model_type] = model_results
                
                # Adjust weights based on performance
                if model_type == 'game_winner':
                    performance = model_results.get('validation_accuracy', 0.5)
                else:
                    performance = 1.0 / (model_results.get('validation_rmse', 1.0) + 0.001)
                
                self.weights[model_type] = performance
                
            except Exception as e:
                logger.error(f"Error training NFL {model_type}: {e}")
                results[model_type] = {'error': str(e)}
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        results['ensemble_weights'] = self.weights
        return results
    
    def predict_ensemble(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Make ensemble predictions."""
        predictions = {}
        
        for model_type, model in self.models.items():
            try:
                if model.model is not None:
                    pred = model.predict(df)
                    predictions[model_type] = pred
            except Exception as e:
                logger.error(f"Error predicting with NFL {model_type}: {e}")
        
        return predictions
