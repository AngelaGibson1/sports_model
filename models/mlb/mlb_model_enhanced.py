# models/mlb/mlb_model_enhanced.py
# Enhanced MLB Prediction Model with Ensemble Methods & Kelly Integration

import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from loguru import logger
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, classification_report
from sklearn.calibration import CalibratedClassifierCV

# XGBoost and LightGBM (with fallbacks if not installed)
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    logger.warning("XGBoost not available - using sklearn models only")
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    logger.warning("LightGBM not available - using sklearn models only")
    LGB_AVAILABLE = False

from config.settings import Settings
from data.database.mlb import MLBDatabase
from data.features.mlb_features import EnhancedMLBFeatureEngineer

class EnhancedMLBPredictionModel:
    """
    Enhanced MLB prediction model with ensemble methods and betting integration.
    Includes probability calibration for accurate Kelly criterion calculations.
    """
    
    def __init__(self, 
                 model_type: str = 'game_winner',
                 use_ensemble: bool = True,
                 calibrate_probabilities: bool = True):
        """
        Initialize enhanced MLB prediction model.
        
        Args:
            model_type: Type of prediction ('game_winner', 'total_runs', 'nrfi', etc.)
            use_ensemble: Whether to use ensemble of multiple models
            calibrate_probabilities: Whether to calibrate probability outputs
        """
        self.model_type = model_type
        self.use_ensemble = use_ensemble
        self.calibrate_probabilities = calibrate_probabilities
        
        # Model configuration
        self.models = {}
        self.ensemble_model = None
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.probability_calibrator = None
        
        # Feature tracking
        self.feature_names = []
        self.feature_importance = {}
        
        # Model performance tracking
        self.training_history = []
        self.performance_metrics = {}
        
        # Target variable mapping
        self.target_column = self._get_target_column()
        
        # Model paths
        self.model_paths = getattr(Settings, 'MODEL_PATHS', {}).get('mlb', {})
        self.model_path = self.model_paths.get(model_type, Path(f"models/mlb/{model_type}_enhanced.joblib"))
        
        logger.info(f"⚾ Enhanced MLB {model_type} model initialized")
    
    def _get_target_column(self) -> str:
        """Get target column based on model type."""
        target_mapping = {
            'game_winner': 'home_win',
            'total_runs': 'total_runs',
            'nrfi': 'no_runs_first_inning',
            'home_score': 'home_score',
            'away_score': 'away_score',
            'run_differential': 'run_differential'
        }
        return target_mapping.get(self.model_type, 'home_win')
    
    def _get_base_models(self) -> Dict[str, Any]:
        """Get base models for ensemble."""
        base_models = {}
        
        # XGBoost (if available)
        if XGB_AVAILABLE:
            if self.model_type in ['total_runs', 'home_score', 'away_score', 'run_differential']:
                base_models['xgboost'] = {
                    'model': xgb.XGBRegressor(
                        n_estimators=300, max_depth=6, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8, random_state=42
                    ),
                    'params': {
                        'n_estimators': [200, 300, 400],
                        'max_depth': [4, 6, 8],
                        'learning_rate': [0.05, 0.1, 0.15],
                        'subsample': [0.7, 0.8, 0.9]
                    }
                }
            else:
                base_models['xgboost'] = {
                    'model': xgb.XGBClassifier(
                        n_estimators=300, max_depth=6, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8, random_state=42,
                        eval_metric='logloss'
                    ),
                    'params': {
                        'n_estimators': [200, 300, 400],
                        'max_depth': [4, 6, 8],
                        'learning_rate': [0.05, 0.1, 0.15],
                        'subsample': [0.7, 0.8, 0.9]
                    }
                }
        
        # LightGBM (if available)
        if LGB_AVAILABLE:
            if self.model_type in ['total_runs', 'home_score', 'away_score', 'run_differential']:
                base_models['lightgbm'] = {
                    'model': lgb.LGBMRegressor(
                        n_estimators=300, max_depth=6, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1
                    ),
                    'params': {
                        'n_estimators': [200, 300, 400],
                        'max_depth': [4, 6, 8],
                        'learning_rate': [0.05, 0.1, 0.15]
                    }
                }
            else:
                base_models['lightgbm'] = {
                    'model': lgb.LGBMClassifier(
                        n_estimators=300, max_depth=6, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1
                    ),
                    'params': {
                        'n_estimators': [200, 300, 400],
                        'max_depth': [4, 6, 8],
                        'learning_rate': [0.05, 0.1, 0.15]
                    }
                }
        
        # Sklearn models (always available)
        if self.model_type in ['total_runs', 'home_score', 'away_score', 'run_differential']:
            # Regression models
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            
            base_models.update({
                'random_forest': {
                    'model': RandomForestRegressor(
                        n_estimators=300, max_depth=8, min_samples_split=5,
                        min_samples_leaf=2, random_state=42, n_jobs=-1
                    ),
                    'params': {
                        'n_estimators': [200, 300, 500],
                        'max_depth': [6, 8, 10],
                        'min_samples_split': [2, 5, 10]
                    }
                },
                'gradient_boosting': {
                    'model': GradientBoostingRegressor(
                        n_estimators=200, max_depth=6, learning_rate=0.1,
                        subsample=0.8, random_state=42
                    ),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [4, 6, 8],
                        'learning_rate': [0.05, 0.1, 0.15]
                    }
                }
            })
        else:
            # Classification models
            base_models.update({
                'random_forest': {
                    'model': RandomForestClassifier(
                        n_estimators=300, max_depth=8, min_samples_split=5,
                        min_samples_leaf=2, random_state=42, n_jobs=-1
                    ),
                    'params': {
                        'n_estimators': [200, 300, 500],
                        'max_depth': [6, 8, 10],
                        'min_samples_split': [2, 5, 10]
                    }
                },
                'gradient_boosting': {
                    'model': GradientBoostingClassifier(
                        n_estimators=200, max_depth=6, learning_rate=0.1,
                        subsample=0.8, random_state=42
                    ),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [4, 6, 8],
                        'learning_rate': [0.05, 0.1, 0.15]
                    }
                },
                'logistic_regression': {
                    'model': LogisticRegression(
                        C=1.0, penalty='l2', solver='liblinear',
                        random_state=42, max_iter=2000
                    ),
                    'params': {
                        'C': [0.1, 1.0, 10.0, 100.0],
                        'penalty': ['l1', 'l2']
                    }
                }
            })
        
        return base_models
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training/prediction.
        Enhanced with automatic feature engineering.
        """
        logger.info(f"🔧 Preparing features for {self.model_type} model...")
        
        if df.empty:
            return pd.DataFrame(), pd.Series(dtype=float)
        
        # Create enhanced features if not already present
        if len(df.columns) < 20:  # Indicates basic data, needs feature engineering
            logger.info("🔧 Creating enhanced features...")
            feature_engineer = EnhancedMLBFeatureEngineer()
            df = feature_engineer.engineer_comprehensive_features(df)
        
        # Prepare target variable
        if self.target_column in df.columns:
            y = df[self.target_column].copy()
        else:
            # Create target based on model type
            if self.model_type == 'game_winner' and 'home_score' in df.columns and 'away_score' in df.columns:
                y = (df['home_score'] > df['away_score']).astype(int)
            elif self.model_type == 'total_runs' and 'home_score' in df.columns and 'away_score' in df.columns:
                y = df['home_score'] + df['away_score']
            elif self.model_type == 'nrfi':
                # Simulate NRFI data (No Runs First Inning)
                y = np.random.choice([0, 1], len(df), p=[0.6, 0.4])
            else:
                logger.warning(f"Cannot create target for {self.model_type}, using dummy data")
                y = pd.Series(np.random.choice([0, 1], len(df)), index=df.index)
        
        # Select features (exclude non-feature columns)
        exclude_columns = [
            'game_id', 'date', 'time', 'status', 'season',
            'home_team_name', 'away_team_name', 'venue',
            'home_score', 'away_score', 'total_runs', 'home_win',
            self.target_column
        ]
        
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        X = df[feature_columns].copy()
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            X[col] = pd.Categorical(X[col]).codes
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        logger.info(f"✅ Features prepared: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def train_ensemble_model(self, 
                           X: pd.DataFrame, 
                           y: pd.Series,
                           tune_hyperparameters: bool = False,
                           cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train ensemble model with multiple base models.
        
        Args:
            X: Feature matrix
            y: Target variable
            tune_hyperparameters: Whether to tune hyperparameters
            cv_folds: Number of cross-validation folds
        
        Returns:
            Training results and metrics
        """
        logger.info(f"🤖 Training ensemble {self.model_type} model...")
        
        if X.empty or len(y) == 0:
            raise ValueError("No data provided for training")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, 
            stratify=y if self.model_type == 'game_winner' else None
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        # Train base models
        base_models = self._get_base_models()
        trained_models = {}
        model_scores = {}
        
        for model_name, model_config in base_models.items():
            logger.info(f"   Training {model_name}...")
            
            try:
                model = model_config['model']
                
                # Hyperparameter tuning if requested
                if tune_hyperparameters and 'params' in model_config:
                    param_grid = model_config['params']
                    
                    # Use appropriate scoring metric
                    scoring = 'neg_log_loss' if self.model_type == 'game_winner' else 'neg_mean_squared_error'
                    
                    search = RandomizedSearchCV(
                        model, param_grid, n_iter=20, cv=cv_folds,
                        scoring=scoring, n_jobs=-1, random_state=42
                    )
                    search.fit(X_train_scaled, y_train)
                    model = search.best_estimator_
                    logger.info(f"      Best params: {search.best_params_}")
                else:
                    model.fit(X_train_scaled, y_train)
                
                # Evaluate model
                if self.model_type == 'game_winner':
                    train_score = model.score(X_train_scaled, y_train)
                    test_score = model.score(X_test_scaled, y_test)
                    
                    # Probability predictions for AUC
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                    auc_score = roc_auc_score(y_test, y_pred_proba)
                    
                    model_scores[model_name] = {
                        'train_accuracy': train_score,
                        'test_accuracy': test_score,
                        'auc_score': auc_score,
                        'cv_score': np.mean(cross_val_score(model, X_train_scaled, y_train, cv=cv_folds))
                    }
                else:
                    # Regression metrics
                    train_score = model.score(X_train_scaled, y_train)
                    test_score = model.score(X_test_scaled, y_test)
                    
                    model_scores[model_name] = {
                        'train_r2': train_score,
                        'test_r2': test_score,
                        'cv_score': np.mean(cross_val_score(model, X_train_scaled, y_train, cv=cv_folds))
                    }
                
                trained_models[model_name] = model
                logger.info(f"      ✅ {model_name} trained successfully")
                
            except Exception as e:
                logger.error(f"      ❌ {model_name} training failed: {e}")
                continue
        
        if not trained_models:
            raise RuntimeError("No models trained successfully")
        
        # Create ensemble
        if self.use_ensemble and len(trained_models) > 1:
            logger.info("🎯 Creating ensemble model...")
            
            # Weight models by performance
            if self.model_type == 'game_winner':
                weights = [model_scores[name]['auc_score'] for name in trained_models.keys()]
            else:
                weights = [model_scores[name]['test_r2'] for name in trained_models.keys()]
            
            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # Create voting ensemble
            estimators = [(name, model) for name, model in trained_models.items()]
            
            if self.model_type == 'game_winner':
                self.ensemble_model = VotingClassifier(
                    estimators=estimators, voting='soft', weights=weights
                )
            else:
                from sklearn.ensemble import VotingRegressor
                self.ensemble_model = VotingRegressor(
                    estimators=estimators, weights=weights
                )
            
            self.ensemble_model.fit(X_train_scaled, y_train)
            
            # Evaluate ensemble
            if self.model_type == 'game_winner':
                ensemble_accuracy = self.ensemble_model.score(X_test_scaled, y_test)
                ensemble_proba = self.ensemble_model.predict_proba(X_test_scaled)[:, 1]
                ensemble_auc = roc_auc_score(y_test, ensemble_proba)
                
                logger.info(f"   🎯 Ensemble accuracy: {ensemble_accuracy:.3f}")
                logger.info(f"   🎯 Ensemble AUC: {ensemble_auc:.3f}")
                
                model_scores['ensemble'] = {
                    'test_accuracy': ensemble_accuracy,
                    'auc_score': ensemble_auc
                }
            else:
                ensemble_r2 = self.ensemble_model.score(X_test_scaled, y_test)
                logger.info(f"   🎯 Ensemble R²: {ensemble_r2:.3f}")
                
                model_scores['ensemble'] = {
                    'test_r2': ensemble_r2
                }
        
        # Select best single model as backup
        if self.model_type == 'game_winner':
            best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x].get('auc_score', 0))
        else:
            best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x].get('test_r2', 0))
        
        self.models = trained_models
        self.best_single_model = trained_models.get(best_model_name.replace('ensemble', list(trained_models.keys())[0]))
        
        # Probability calibration for classification
        if self.calibrate_probabilities and self.model_type == 'game_winner':
            logger.info("🎯 Calibrating probabilities...")
            
            model_to_calibrate = self.ensemble_model if self.ensemble_model else self.best_single_model
            
            self.probability_calibrator = CalibratedClassifierCV(
                model_to_calibrate, method='isotonic', cv=3
            )
            self.probability_calibrator.fit(X_train_scaled, y_train)
            
            # Test calibration
            calibrated_proba = self.probability_calibrator.predict_proba(X_test_scaled)[:, 1]
            calibrated_auc = roc_auc_score(y_test, calibrated_proba)
            logger.info(f"   📊 Calibrated AUC: {calibrated_auc:.3f}")
        
        # Feature importance
        if hasattr(self.best_single_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.best_single_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.feature_importance = importance_df.set_index('feature')['importance'].to_dict()
            
            # Log top features
            logger.info("🔍 Top 10 most important features:")
            for i, (feature, importance) in enumerate(importance_df.head(10).values):
                logger.info(f"   {i+1:2}. {feature}: {importance:.3f}")
        
        # Training results
        training_results = {
            'model_scores': model_scores,
            'best_model': best_model_name,
            'ensemble_used': self.ensemble_model is not None,
            'calibrated': self.probability_calibrator is not None,
            'feature_count': len(self.feature_names),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        self.performance_metrics = model_scores
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'results': training_results
        })
        
        logger.info("✅ Ensemble model training completed")
        return training_results
    
    def predict_probability(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities with calibrated ensemble model.
        Critical for accurate Kelly criterion calculations.
        """
        if X.empty:
            return np.array([])
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        # Use calibrated model if available
        if self.probability_calibrator:
            probabilities = self.probability_calibrator.predict_proba(X_scaled)
            return probabilities[:, 1] if len(probabilities.shape) > 1 else probabilities
        
        # Use ensemble model
        elif self.ensemble_model:
            probabilities = self.ensemble_model.predict_proba(X_scaled)
            return probabilities[:, 1] if len(probabilities.shape) > 1 else probabilities
        
        # Fallback to best single model
        elif self.best_single_model:
            probabilities = self.best_single_model.predict_proba(X_scaled)
            return probabilities[:, 1] if len(probabilities.shape) > 1 else probabilities
        
        # No trained model
        else:
            logger.warning("No trained model available, returning default probabilities")
            return np.full(len(X), 0.5)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the best available model."""
        if X.empty:
            return np.array([])
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        # Use ensemble model
        if self.ensemble_model:
            return self.ensemble_model.predict(X_scaled)
        
        # Fallback to best single model
        elif self.best_single_model:
            return self.best_single_model.predict(X_scaled)
        
        # No trained model
        else:
            logger.warning("No trained model available, returning default predictions")
            return np.full(len(X), 0 if self.model_type == 'game_winner' else 9.0)
    
    def save_model(self, filepath: Optional[Path] = None) -> bool:
        """Save the complete model package."""
        save_path = filepath or self.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            model_package = {
                'ensemble_model': self.ensemble_model,
                'models': self.models,
                'best_single_model': self.best_single_model,
                'scaler': self.scaler,
                'probability_calibrator': self.probability_calibrator,
                'feature_names': self.feature_names,
                'feature_importance': self.feature_importance,
                'performance_metrics': self.performance_metrics,
                'model_type': self.model_type,
                'use_ensemble': self.use_ensemble,
                'calibrate_probabilities': self.calibrate_probabilities,
                'training_history': self.training_history,
                'saved_date': datetime.now().isoformat(),
                'version': '2.0_enhanced'
            }
            
            joblib.dump(model_package, save_path)
            logger.info(f"💾 Enhanced model saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")
            return False
    
    def load_model(self, filepath: Optional[Path] = None) -> bool:
        """Load the complete model package."""
        load_path = filepath or self.model_path
        
        if not load_path.exists():
            logger.error(f"Model file not found: {load_path}")
            return False
        
        try:
            model_package = joblib.load(load_path)
            
            self.ensemble_model = model_package.get('ensemble_model')
            self.models = model_package.get('models', {})
            self.best_single_model = model_package.get('best_single_model')
            self.scaler = model_package.get('scaler')
            self.probability_calibrator = model_package.get('probability_calibrator')
            self.feature_names = model_package.get('feature_names', [])
            self.feature_importance = model_package.get('feature_importance', {})
            self.performance_metrics = model_package.get('performance_metrics', {})
            self.training_history = model_package.get('training_history', [])
            
            saved_date = model_package.get('saved_date', 'unknown')
            version = model_package.get('version', '1.0')
            
            logger.info(f"📦 Enhanced model loaded from {load_path}")
            logger.info(f"   Version: {version}, Saved: {saved_date}")
            logger.info(f"   Features: {len(self.feature_names)}")
            logger.info(f"   Ensemble: {self.ensemble_model is not None}")
            logger.info(f"   Calibrated: {self.probability_calibrator is not None}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False


# Model factory for easy creation
class MLBModelFactory:
    """Factory for creating different types of MLB models."""
    
    @staticmethod
    def create_game_winner_model() -> EnhancedMLBPredictionModel:
        """Create a game winner prediction model."""
        return EnhancedMLBPredictionModel(
            model_type='game_winner',
            use_ensemble=True,
            calibrate_probabilities=True
        )
    
    @staticmethod
    def create_total_runs_model() -> EnhancedMLBPredictionModel:
        """Create a total runs prediction model."""
        return EnhancedMLBPredictionModel(
            model_type='total_runs',
            use_ensemble=True,
            calibrate_probabilities=False  # Regression model
        )
    
    @staticmethod
    def create_nrfi_model() -> EnhancedMLBPredictionModel:
        """Create a No Runs First Inning (NRFI) prediction model."""
        return EnhancedMLBPredictionModel(
            model_type='nrfi',
            use_ensemble=True,
            calibrate_probabilities=True
        )


# Example usage
if __name__ == "__main__":
    print("🧪 Testing Enhanced MLB Prediction Model")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'game_id': range(1000),
        'date': pd.date_range('2024-04-01', periods=1000),
        'home_team_id': np.random.choice(range(1, 31), 1000),
        'away_team_id': np.random.choice(range(1, 31), 1000),
        'home_score': np.random.poisson(4.5, 1000),
        'away_score': np.random.poisson(4.5, 1000),
        'season': 2024
    })
    
    # Test game winner model
    model = MLBModelFactory.create_game_winner_model()
    
    # Prepare features
    X, y = model.prepare_features(sample_data)
    
    # Train model
    results = model.train_ensemble_model(X, y, tune_hyperparameters=False)
    
    print(f"✅ Model trained with {results['feature_count']} features")
    print(f"🎯 Best model: {results['best_model']}")
    print(f"📊 Performance: {results['model_scores']}")
    
    # Test prediction
    test_proba = model.predict_probability(X.head(5))
    print(f"🔮 Sample probabilities: {test_proba}")
