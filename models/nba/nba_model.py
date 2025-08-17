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

class NBAPredictionModel:
    """
    Comprehensive NBA prediction model using XGBoost.
    Supports both game outcome and player prop predictions.
    """
    
    def __init__(self, model_type: str = 'game_winner'):
        """
        Initialize NBA prediction model.
        
        Args:
            model_type: Type of model ('game_winner', 'total_points', 'player_points', etc.)
        """
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.feature_importance = {}
        
        # Get model parameters from settings
        self.model_params = Settings.get_model_params('nba')
        
        # Adjust parameters based on model type
        if model_type == 'game_winner':
            self.model_params['objective'] = 'binary:logistic'
            self.model_params['eval_metric'] = 'logloss'
            self.target_column = 'home_win'
        elif model_type == 'total_points':
            self.model_params['objective'] = 'reg:squarederror'
            self.model_params['eval_metric'] = 'rmse'
            self.target_column = 'total_points'
        elif 'player' in model_type:
            self.model_params['objective'] = 'reg:squarederror'
            self.model_params['eval_metric'] = 'rmse'
            self.target_column = self._get_player_target_column(model_type)
        
        # Model storage path
        self.model_path = Settings.MODEL_PATHS['nba'].get(model_type)
        
        logger.info(f"🏀 Initialized NBA {model_type} model")
    
    def _get_player_target_column(self, model_type: str) -> str:
        """Get target column for player prop models."""
        player_targets = {
            'player_points': 'points',
            'player_rebounds': 'rebounds',
            'player_assists': 'assists',
            'player_threes': 'three_pointers_made'
        }
        return player_targets.get(model_type, 'points')
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training/prediction.
        
        Args:
            df: Input DataFrame with engineered features
            
        Returns:
            Tuple of (features_df, target_series)
        """
        logger.info(f"🔧 Preparing features for {self.model_type} model...")
        
        # Remove non-feature columns
        exclude_columns = [
            'game_id', 'date', 'Date', 'season', 'home_team_id', 'away_team_id',
            'home_team_name', 'away_team_name', 'home_score', 'away_score',
            'status', 'venue', 'city', 'created_at', 'updated_at',
            'player_id', 'player_name', 'team_id'
        ]
        
        # Also exclude target columns we're not predicting
        target_columns = ['home_win', 'total_points', 'points', 'rebounds', 'assists']
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
        
        logger.info(f"✅ Training complete for {self.model_type}")
        for metric, value in performance_metrics.items():
            logger.info(f"   {metric}: {value:.4f}")
        
        return training_results
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            df: DataFrame with features (same structure as training data)
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a trained model.")
        
        # Prepare features (without target)
        X, _ = self.prepare_features(df)
        
        if len(X) == 0:
            return np.array([])
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        logger.info(f"✅ Generated {len(predictions)} predictions for {self.model_type}")
        
        return predictions
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities (for classification models).
        
        Args:
            df: DataFrame with features
            
        Returns:
            Array of prediction probabilities
        """
        if self.model_type != 'game_winner':
            raise ValueError("predict_proba only available for classification models")
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a trained model.")
        
        # Prepare features
        X, _ = self.prepare_features(df)
        
        if len(X) == 0:
            return np.array([])
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities
        probabilities = self.model.predict_proba(X_scaled)
        
        return probabilities
    
    def _optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, cv) -> Dict[str, Any]:
        """Optimize hyperparameters using grid search."""
        param_grid = {
            'n_estimators': [200, 300, 400],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.08, 0.1, 0.12],
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
        """
        Save the trained model to disk.
        
        Args:
            filepath: Optional custom file path
            
        Returns:
            Path where model was saved
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        save_path = filepath or self.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create model package
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
        
        # Save using joblib
        joblib.dump(model_package, save_path)
        
        logger.info(f"💾 Saved {self.model_type} model to {save_path}")
        return save_path
    
    def load_model(self, filepath: Optional[Path] = None) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Optional custom file path
            
        Returns:
            True if successful, False otherwise
        """
        load_path = filepath or self.model_path
        
        if not load_path.exists():
            logger.error(f"Model file not found: {load_path}")
            return False
        
        try:
            # Load model package
            model_package = joblib.load(load_path)
            
            # Restore model components
            self.model = model_package['model']
            self.scaler = model_package['scaler']
            self.feature_names = model_package['feature_names']
            self.feature_importance = model_package.get('feature_importance', {})
            self.model_type = model_package.get('model_type', self.model_type)
            self.model_params = model_package.get('model_params', self.model_params)
            self.target_column = model_package.get('target_column', self.target_column)
            
            saved_date = model_package.get('saved_date', 'unknown')
            logger.info(f"📦 Loaded {self.model_type} model from {load_path}")
            logger.info(f"   Model saved: {saved_date}")
            logger.info(f"   Features: {len(self.feature_names)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def evaluate_model(self, test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Args:
            test_df: Test DataFrame with features and target
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a trained model.")
        
        # Prepare test data
        X_test, y_test = self.prepare_features(test_df)
        
        if len(X_test) == 0:
            return {'error': 'No valid test data'}
        
        # Make predictions
        predictions = self.predict(test_df)
        
        if self.model_type == 'game_winner':
            # Classification metrics
            accuracy = accuracy_score(y_test, predictions)
            probabilities = self.predict_proba(test_df)[:, 1]
            auc = roc_auc_score(y_test, probabilities)
            
            # Detailed classification report
            class_report = classification_report(y_test, predictions, output_dict=True)
            
            evaluation_results = {
                'test_samples': len(y_test),
                'accuracy': accuracy,
                'auc': auc,
                'precision': class_report['weighted avg']['precision'],
                'recall': class_report['weighted avg']['recall'],
                'f1_score': class_report['weighted avg']['f1-score'],
                'home_win_rate': y_test.mean(),
                'predicted_home_win_rate': predictions.mean()
            }
        else:
            # Regression metrics
            rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
            mae = np.mean(np.abs(y_test - predictions))
            mape = np.mean(np.abs((y_test - predictions) / (y_test + 0.001))) * 100
            r2 = 1 - (np.sum((y_test - predictions) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
            
            evaluation_results = {
                'test_samples': len(y_test),
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'r2_score': r2,
                'target_mean': y_test.mean(),
                'target_std': y_test.std(),
                'prediction_mean': predictions.mean(),
                'prediction_std': predictions.std()
            }
        
        logger.info(f"📊 Model evaluation complete:")
        for metric, value in evaluation_results.items():
            if isinstance(value, (int, float)):
                logger.info(f"   {metric}: {value:.4f}")
        
        return evaluation_results
    
    def get_feature_importance_analysis(self) -> Dict[str, Any]:
        """
        Get detailed feature importance analysis.
        
        Returns:
            Dictionary with feature importance insights
        """
        if not self.feature_importance:
            return {'error': 'No feature importance data available'}
        
        # Sort features by importance
        sorted_features = sorted(
            self.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Calculate statistics
        importances = list(self.feature_importance.values())
        total_importance = sum(importances)
        
        # Top features analysis
        top_10 = sorted_features[:10]
        top_10_importance = sum([imp for _, imp in top_10])
        
        # Feature categories
        categories = {
            'differentials': [f for f in self.feature_names if '_diff' in f],
            'rolling_stats': [f for f in self.feature_names if 'roll_' in f],
            'form_features': [f for f in self.feature_names if 'form' in f or 'trend' in f],
            'pace_features': [f for f in self.feature_names if 'pace' in f],
            'advanced_features': [f for f in self.feature_names if any(x in f for x in ['pythag', 'efg', 'net_rating'])],
            'situational': [f for f in self.feature_names if any(x in f for x in ['home_', 'weekend', 'rest'])]
        }
        
        # Calculate category importance
        category_importance = {}
        for category, features in categories.items():
            cat_importance = sum(self.feature_importance.get(f, 0) for f in features)
            category_importance[category] = cat_importance
        
        analysis = {
            'total_features': len(self.feature_names),
            'top_10_features': top_10,
            'top_10_importance_share': top_10_importance / total_importance if total_importance > 0 else 0,
            'feature_importance_stats': {
                'mean': np.mean(importances),
                'std': np.std(importances),
                'max': np.max(importances),
                'min': np.min(importances)
            },
            'category_importance': category_importance,
            'low_importance_features': [f for f, imp in sorted_features if imp < np.mean(importances) * 0.1]
        }
        
        return analysis
    
    def predict_with_confidence(self, df: pd.DataFrame, confidence_threshold: float = 0.6) -> pd.DataFrame:
        """
        Make predictions with confidence scores.
        
        Args:
            df: DataFrame with features
            confidence_threshold: Minimum confidence for high-confidence predictions
            
        Returns:
            DataFrame with predictions and confidence scores
        """
        if self.model_type != 'game_winner':
            predictions = self.predict(df)
            
            # For regression, use prediction interval as confidence proxy
            results = pd.DataFrame({
                'prediction': predictions,
                'confidence': 0.5,  # Placeholder for regression confidence
                'high_confidence': False
            })
        else:
            probabilities = self.predict_proba(df)
            predictions = (probabilities[:, 1] > 0.5).astype(int)
            
            # Confidence is the maximum probability
            confidence = np.max(probabilities, axis=1)
            high_confidence = confidence >= confidence_threshold
            
            results = pd.DataFrame({
                'prediction': predictions,
                'home_win_probability': probabilities[:, 1],
                'away_win_probability': probabilities[:, 0],
                'confidence': confidence,
                'high_confidence': high_confidence
            })
        
        return results
    
    def retrain_with_new_data(self, new_df: pd.DataFrame, retrain_from_scratch: bool = False) -> Dict[str, Any]:
        """
        Update model with new data.
        
        Args:
            new_df: New training data
            retrain_from_scratch: Whether to retrain completely or update incrementally
            
        Returns:
            Dictionary with retraining results
        """
        if retrain_from_scratch or self.model is None:
            # Complete retraining
            return self.train(new_df)
        else:
            # Incremental learning (simplified approach)
            logger.info("🔄 Performing incremental model update...")
            
            X_new, y_new = self.prepare_features(new_df)
            
            if len(X_new) == 0:
                return {'error': 'No valid new data for retraining'}
            
            # Scale new features
            X_new_scaled = self.scaler.transform(X_new)
            
            # For XGBoost, we need to retrain (no true incremental learning)
            # In practice, you might want to combine old and new data
            logger.warning("XGBoost doesn't support true incremental learning. Consider retraining from scratch.")
            
            return {'status': 'incremental_update_simulated', 'new_samples': len(X_new)}

class NBAModelEnsemble:
    """Ensemble of NBA prediction models for improved accuracy."""
    
    def __init__(self, model_types: List[str] = ['game_winner']):
        """Initialize ensemble with multiple model types."""
        self.models = {}
        self.weights = {}
        
        for model_type in model_types:
            self.models[model_type] = NBAPredictionModel(model_type)
            self.weights[model_type] = 1.0  # Equal weights initially
        
        logger.info(f"🎯 Initialized NBA ensemble with {len(self.models)} models")
    
    def train_ensemble(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train all models in the ensemble."""
        results = {}
        
        for model_type, model in self.models.items():
            logger.info(f"Training {model_type} model...")
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
                logger.error(f"Error training {model_type}: {e}")
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
                logger.error(f"Error predicting with {model_type}: {e}")
        
        return predictions
    
    def save_ensemble(self, base_path: Path):
        """Save all models in the ensemble."""
        for model_type, model in self.models.items():
            model_path = base_path / f"{model_type}_model.joblib"
            model.save_model(model_path)
        
        # Save ensemble metadata
        ensemble_meta = {
            'weights': self.weights,
            'model_types': list(self.models.keys()),
            'saved_date': datetime.now().isoformat()
        }
        
        meta_path = base_path / "ensemble_metadata.joblib"
        joblib.dump(ensemble_meta, meta_path)
        
        logger.info(f"💾 Saved NBA ensemble to {base_path}")
    
    def load_ensemble(self, base_path: Path):
        """Load all models in the ensemble."""
        # Load ensemble metadata
        meta_path = base_path / "ensemble_metadata.joblib"
        if meta_path.exists():
            ensemble_meta = joblib.load(meta_path)
            self.weights = ensemble_meta.get('weights', {})
        
        # Load individual models
        for model_type, model in self.models.items():
            model_path = base_path / f"{model_type}_model.joblib"
            model.load_model(model_path)
        
        logger.info(f"📦 Loaded NBA ensemble from {base_path}")
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
        Train the NBA prediction model.
        
        Args:
            df: Training DataFrame with features and target
            validation_split: Fraction of data to use for validation
            cv_folds: Number of cross-validation folds
            optimize_hyperparams: Whether to perform hyperparameter optimization
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"🚀 Training NBA {self.model_type} model...")
        
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
        
        # Scale features if needed (especially for neural net-like models)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model with early stopping
        # Split data for early stopping
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
        
        # Validation predictions
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
        else
