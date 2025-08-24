# pipeline/nfl_model_pipeline.py
# Enhanced NFL Training Pipeline
# Orchestrates: Data Ingestion → Feature Engineering → Model Training → Prediction

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from loguru import logger
import warnings
from pathlib import Path
import sys
warnings.filterwarnings('ignore')

# Add project root to path for imports (from pipeline/ folder)
sys.path.append(str(Path(__file__).parent.parent))

from data.database.nfl import NFLDatabase
from data.player_mapping import EnhancedPlayerMapper
from config.settings import Settings

# Handle imports with fallbacks for circular import issues
try:
    from data.features.nfl_features import NFLFeatureEngineer
    NFL_FEATURES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Could not import NFLFeatureEngineer: {e}")
    NFL_FEATURES_AVAILABLE = False
    NFLFeatureEngineer = None

try:
    from models.nfl.enhanced_hybrid_nfl_model import EnhancedHybridNFLModel
    NFL_MODELS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Could not import NFL models: {e}")
    NFL_MODELS_AVAILABLE = False
    EnhancedHybridNFLModel = None

try:
    from utils.betting_calculator import KellyCriterionCalculator
    BETTING_CALC_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Could not import KellyCriterionCalculator: {e}")
    BETTING_CALC_AVAILABLE = False
    KellyCriterionCalculator = None

# IMPROVEMENT 1: Centralize API Clients - Direct instantiation
from api_clients.sports_api import SportsAPIClient
from api_clients.unified_api_client import DataSourceClient


class NFLDataSchema:
    """NFL data schema definitions and validation."""
    
    REQUIRED_GAME_COLUMNS = [
        'game_id', 'date', 'home_team_id', 'away_team_id', 
        'home_team_name', 'away_team_name', 'week', 'season'
    ]
    
    OPTIONAL_GAME_COLUMNS = [
        'home_score', 'away_score', 'status', 'weather_conditions',
        'temperature', 'wind_speed', 'venue'
    ]
    
    REQUIRED_TEAM_COLUMNS = [
        'team_id', 'name', 'abbreviation', 'conference', 'division'
    ]
    
    REQUIRED_PLAYER_COLUMNS = [
        'player_id', 'name', 'team_id', 'position', 'season'
    ]
    
    @staticmethod
    def validate_games_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate games DataFrame schema."""
        missing_columns = []
        for col in NFLDataSchema.REQUIRED_GAME_COLUMNS:
            if col not in df.columns:
                missing_columns.append(col)
        
        is_valid = len(missing_columns) == 0
        return is_valid, missing_columns
    
    @staticmethod
    def validate_teams_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate teams DataFrame schema."""
        missing_columns = []
        for col in NFLDataSchema.REQUIRED_TEAM_COLUMNS:
            if col not in df.columns:
                missing_columns.append(col)
        
        is_valid = len(missing_columns) == 0
        return is_valid, missing_columns


class NFLDataPipeline:
    """
    Enhanced NFL data pipeline that handles the complete flow:
    Raw Data → Validation → Feature Engineering → Model Training → Predictions
    """
    
    def __init__(self, 
                 use_enhanced_features: bool = True,
                 cache_intermediate: bool = True):
        """
        Initialize NFL pipeline.
        
        Args:
            use_enhanced_features: Whether to use enhanced feature engineering
            cache_intermediate: Whether to cache intermediate results
        """
        self.use_enhanced_features = use_enhanced_features
        self.cache_intermediate = cache_intermediate
        
        # Initialize core components
        self.db = NFLDatabase()
        self.player_mapper = EnhancedPlayerMapper(sport='nfl', auto_build=True)
        
        # Initialize API clients with error handling
        try:
            self.api_client = DataSourceClient(sport='nfl')
            logger.info("✅ NFL DataSourceClient initialized")
        except Exception as e:
            logger.warning(f"DataSourceClient failed, using SportsAPIClient: {e}")
            self.api_client = SportsAPIClient(sport='nfl')
        
        # Initialize feature engineer
        if NFL_FEATURES_AVAILABLE:
            self.feature_engineer = NFLFeatureEngineer(use_current_data=True)
            logger.info("✅ NFL Feature Engineer initialized")
        else:
            self.feature_engineer = None
            logger.warning("⚠️ NFL Feature Engineer not available")
        
        # Initialize model
        if NFL_MODELS_AVAILABLE:
            self.model = EnhancedHybridNFLModel()
            logger.info("✅ Enhanced Hybrid NFL Model initialized")
        else:
            self.model = None
            logger.warning("⚠️ NFL Model not available")
        
        # Initialize betting calculator
        if BETTING_CALC_AVAILABLE:
            self.betting_calc = KellyCriterionCalculator()
        else:
            self.betting_calc = None
        
        # Pipeline state tracking
        self.pipeline_state = {
            'last_ingestion': None,
            'last_training': None,
            'last_prediction': None,
            'data_quality_score': None
        }
        
        logger.info("🏈 NFL Data Pipeline initialized")
    
    def run_full_training_pipeline(self, 
                                 seasons: Optional[List[int]] = None,
                                 force_retrain: bool = False) -> Dict[str, Any]:
        """
        Run the complete training pipeline from data ingestion to model training.
        
        Args:
            seasons: List of seasons to include in training
            force_retrain: Whether to force retraining even if recent model exists
            
        Returns:
            Dictionary with pipeline results and metrics
        """
        logger.info("🚀 Starting Full NFL Training Pipeline...")
        pipeline_start = datetime.now()
        
        results = {
            'pipeline_start': pipeline_start,
            'steps_completed': [],
            'errors': [],
            'data_summary': {},
            'model_metrics': {},
            'training_time': None
        }
        
        try:
            # STEP 1: Data Ingestion and Validation
            logger.info("📊 Step 1: Data Ingestion and Validation")
            ingestion_result = self._ingest_and_validate_data(seasons)
            results['steps_completed'].append('data_ingestion')
            results['data_summary'] = ingestion_result
            
            # STEP 2: Feature Engineering
            logger.info("🔧 Step 2: Feature Engineering")
            feature_result = self._engineer_training_features(ingestion_result['games'])
            results['steps_completed'].append('feature_engineering')
            results['feature_summary'] = feature_result
            
            # STEP 3: Model Training
            logger.info("🤖 Step 3: Model Training")
            if self.model and not self._should_skip_training(force_retrain):
                training_result = self._train_enhanced_model(feature_result['features'])
                results['steps_completed'].append('model_training')
                results['model_metrics'] = training_result
            else:
                logger.warning("⚠️ Model training skipped")
                results['errors'].append("Model not available or training skipped")
            
            # STEP 4: Model Validation
            logger.info("✅ Step 4: Model Validation")
            validation_result = self._validate_trained_model()
            results['steps_completed'].append('model_validation')
            results['validation_summary'] = validation_result
            
            # Update pipeline state
            self.pipeline_state['last_training'] = datetime.now()
            results['training_time'] = (datetime.now() - pipeline_start).total_seconds()
            
            logger.info(f"✅ NFL Training Pipeline Complete in {results['training_time']:.1f}s")
            
        except Exception as e:
            logger.error(f"❌ NFL Training Pipeline Failed: {e}")
            results['errors'].append(str(e))
            import traceback
            traceback.print_exc()
        
        return results
    
    def _ingest_and_validate_data(self, seasons: Optional[List[int]]) -> Dict[str, Any]:
        """Ingest and validate NFL data."""
        if not seasons:
            seasons = [datetime.now().year - i for i in range(5)]  # Last 5 seasons
        
        logger.info(f"📥 Ingesting NFL data for seasons: {seasons}")
        
        # Get games data
        games_df = self.db.get_historical_data(seasons)
        
        if games_df.empty:
            logger.warning("⚠️ No historical games found, attempting API fetch...")
            games_df = self._fetch_games_from_api(seasons)
        
        # Get teams data
        teams_df = self.player_mapper.team_map
        if teams_df.empty:
            logger.warning("⚠️ No teams data from player mapper")
            teams_df = self._get_fallback_teams_data()
        
        # Get player data
        players_df = self.player_mapper.player_map
        if players_df.empty:
            logger.warning("⚠️ No players data from player mapper")
            players_df = pd.DataFrame()  # Will use defaults in model
        
        # Validate data schemas
        games_valid, games_missing = NFLDataSchema.validate_games_data(games_df)
        teams_valid, teams_missing = NFLDataSchema.validate_teams_data(teams_df)
        
        # Data quality assessment
        quality_score = self._assess_data_quality(games_df, teams_df, players_df)
        self.pipeline_state['data_quality_score'] = quality_score
        
        return {
            'games': games_df,
            'teams': teams_df, 
            'players': players_df,
            'validation': {
                'games_valid': games_valid,
                'games_missing_columns': games_missing,
                'teams_valid': teams_valid,
                'teams_missing_columns': teams_missing
            },
            'data_quality_score': quality_score,
            'seasons': seasons
        }
    
    def _fetch_games_from_api(self, seasons: List[int]) -> pd.DataFrame:
        """Fetch games from API if not in database."""
        all_games = []
        
        for season in seasons:
            try:
                season_games = self.api_client.get_games(season=season)
                if not season_games.empty:
                    season_games['season'] = season
                    all_games.append(season_games)
                    logger.info(f"✅ Fetched {len(season_games)} games for {season}")
            except Exception as e:
                logger.warning(f"⚠️ Could not fetch {season} games: {e}")
        
        if all_games:
            combined_games = pd.concat(all_games, ignore_index=True)
            logger.info(f"✅ Total games fetched: {len(combined_games)}")
            return combined_games
        else:
            return pd.DataFrame()
    
    def _get_fallback_teams_data(self) -> pd.DataFrame:
        """Create fallback teams data."""
        teams_data = []
        team_info = [
            (1, 'Buffalo Bills', 'BUF', 'AFC', 'East'),
            (2, 'Miami Dolphins', 'MIA', 'AFC', 'East'),
            (3, 'New England Patriots', 'NE', 'AFC', 'East'),
            (4, 'New York Jets', 'NYJ', 'AFC', 'East'),
            (5, 'Baltimore Ravens', 'BAL', 'AFC', 'North'),
            (6, 'Cincinnati Bengals', 'CIN', 'AFC', 'North'),
            (7, 'Cleveland Browns', 'CLE', 'AFC', 'North'),
            (8, 'Pittsburgh Steelers', 'PIT', 'AFC', 'North'),
            (9, 'Houston Texans', 'HOU', 'AFC', 'South'),
            (10, 'Indianapolis Colts', 'IND', 'AFC', 'South'),
            (11, 'Jacksonville Jaguars', 'JAX', 'AFC', 'South'),
            (12, 'Tennessee Titans', 'TEN', 'AFC', 'South'),
            (13, 'Denver Broncos', 'DEN', 'AFC', 'West'),
            (14, 'Kansas City Chiefs', 'KC', 'AFC', 'West'),
            (15, 'Las Vegas Raiders', 'LV', 'AFC', 'West'),
            (16, 'Los Angeles Chargers', 'LAC', 'AFC', 'West'),
            (17, 'Dallas Cowboys', 'DAL', 'NFC', 'East'),
            (18, 'New York Giants', 'NYG', 'NFC', 'East'),
            (19, 'Philadelphia Eagles', 'PHI', 'NFC', 'East'),
            (20, 'Washington Commanders', 'WAS', 'NFC', 'East'),
            (21, 'Chicago Bears', 'CHI', 'NFC', 'North'),
            (22, 'Detroit Lions', 'DET', 'NFC', 'North'),
            (23, 'Green Bay Packers', 'GB', 'NFC', 'North'),
            (24, 'Minnesota Vikings', 'MIN', 'NFC', 'North'),
            (25, 'Atlanta Falcons', 'ATL', 'NFC', 'South'),
            (26, 'Carolina Panthers', 'CAR', 'NFC', 'South'),
            (27, 'New Orleans Saints', 'NO', 'NFC', 'South'),
            (28, 'Tampa Bay Buccaneers', 'TB', 'NFC', 'South'),
            (29, 'Arizona Cardinals', 'ARI', 'NFC', 'West'),
            (30, 'Los Angeles Rams', 'LAR', 'NFC', 'West'),
            (31, 'San Francisco 49ers', 'SF', 'NFC', 'West'),
            (32, 'Seattle Seahawks', 'SEA', 'NFC', 'West')
        ]
        
        for team_id, name, abbr, conference, division in team_info:
            teams_data.append({
                'team_id': team_id,
                'name': name,
                'abbreviation': abbr,
                'conference': conference,
                'division': division
            })
        
        return pd.DataFrame(teams_data)
    
    def _assess_data_quality(self, games_df: pd.DataFrame, 
                           teams_df: pd.DataFrame, 
                           players_df: pd.DataFrame) -> float:
        """Assess overall data quality."""
        quality_factors = []
        
        # Games data quality
        if not games_df.empty:
            games_completeness = 1 - (games_df.isnull().sum().sum() / (len(games_df) * len(games_df.columns)))
            quality_factors.append(games_completeness * 0.5)  # 50% weight
        
        # Teams data quality
        if not teams_df.empty:
            teams_completeness = 1 - (teams_df.isnull().sum().sum() / (len(teams_df) * len(teams_df.columns)))
            quality_factors.append(teams_completeness * 0.3)  # 30% weight
        
        # Players data quality
        if not players_df.empty:
            players_completeness = 1 - (players_df.isnull().sum().sum() / (len(players_df) * len(players_df.columns)))
            quality_factors.append(players_completeness * 0.2)  # 20% weight
        else:
            quality_factors.append(0.5 * 0.2)  # Default score when no player data
        
        overall_quality = sum(quality_factors) if quality_factors else 0.0
        
        logger.info(f"📊 NFL Data Quality Score: {overall_quality:.3f}")
        return overall_quality
    
    def _engineer_training_features(self, games_df: pd.DataFrame) -> Dict[str, Any]:
        """Engineer features for training."""
        if not self.feature_engineer:
            logger.warning("⚠️ Feature engineer not available, using basic features")
            return {'features': games_df, 'feature_count': len(games_df.columns)}
        
        try:
            if self.use_enhanced_features:
                features_df = self.feature_engineer.engineer_all_features(
                    games_df, 
                    include_advanced=True
                )
            else:
                features_df = self.feature_engineer.create_team_features(games_df)
            
            # Feature quality assessment
            feature_quality = self._assess_feature_quality(features_df)
            
            logger.info(f"✅ Engineered {len(features_df.columns)} NFL features")
            
            return {
                'features': features_df,
                'feature_count': len(features_df.columns),
                'feature_quality': feature_quality,
                'original_count': len(games_df.columns)
            }
            
        except Exception as e:
            logger.error(f"❌ Feature engineering failed: {e}")
            return {'features': games_df, 'feature_count': len(games_df.columns), 'error': str(e)}
    
    def _assess_feature_quality(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """Assess feature quality metrics."""
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        
        quality_metrics = {
            'completeness': 1 - (features_df[numeric_cols].isnull().sum().sum() / 
                                (len(features_df) * len(numeric_cols))),
            'variance_score': np.mean([features_df[col].var() for col in numeric_cols 
                                    if features_df[col].var() > 0]),
            'correlation_diversity': len(numeric_cols) - np.sum(
                np.corrcoef(features_df[numeric_cols].T, rowvar=True) > 0.95
            ) if len(numeric_cols) > 1 else 1.0
        }
        
        return quality_metrics
    
    def _train_enhanced_model(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Train the enhanced NFL model."""
        try:
            # Set training data in model
            self.model.current_season = datetime.now().year
            
            # Trigger training
            self.model.train_enhanced_nfl_model()
            
            # Get training metrics
            training_metrics = {
                'training_completed': True,
                'model_type': 'enhanced_hybrid_nfl',
                'feature_count': len(features_df.columns),
                'training_samples': len(features_df)
            }
            
            logger.info("✅ NFL Model training completed")
            return training_metrics
            
        except Exception as e:
            logger.error(f"❌ NFL Model training failed: {e}")
            return {'training_completed': False, 'error': str(e)}
    
    def _should_skip_training(self, force_retrain: bool) -> bool:
        """Determine if training should be skipped."""
        if force_retrain:
            return False
        
        # Check if model exists and is recent
        model_path = Path('models/nfl/enhanced_hybrid_nfl_model.pkl')
        if model_path.exists():
            model_age = datetime.now() - datetime.fromtimestamp(model_path.stat().st_mtime)
            if model_age.days < 7:  # Model is less than a week old
                logger.info("⏭️ Skipping training - recent model exists")
                return True
        
        return False
    
    def _validate_trained_model(self) -> Dict[str, Any]:
        """Validate the trained model."""
        if not self.model:
            return {'validation_passed': False, 'error': 'No model available'}
        
        try:
            # Check if model files exist
            model_path = Path('models/nfl/enhanced_hybrid_nfl_model.pkl')
            
            validation_results = {
                'model_file_exists': model_path.exists(),
                'model_loadable': False,
                'prediction_test_passed': False
            }
            
            if model_path.exists():
                # Test loading model
                try:
                    self.model._load_enhanced_nfl_model()
                    validation_results['model_loadable'] = True
                    
                    # Test prediction with dummy data
                    dummy_games = self._create_dummy_games()
                    predictions = self.model._make_enhanced_nfl_predictions(dummy_games)
                    
                    if not predictions.empty and 'home_win_prob' in predictions.columns:
                        validation_results['prediction_test_passed'] = True
                    
                except Exception as e:
                    validation_results['load_error'] = str(e)
            
            validation_results['validation_passed'] = all([
                validation_results['model_file_exists'],
                validation_results['model_loadable'], 
                validation_results['prediction_test_passed']
            ])
            
            return validation_results
            
        except Exception as e:
            return {'validation_passed': False, 'error': str(e)}
    
    def _create_dummy_games(self) -> pd.DataFrame:
        """Create dummy games for testing."""
        return pd.DataFrame({
            'game_id': ['test_1'],
            'home_team_name': ['Kansas City Chiefs'],
            'away_team_name': ['Denver Broncos'],
            'home_team_id': [14],
            'away_team_id': [13],
            'commence_time': [datetime.now()],
            'home_odds': [-150],
            'away_odds': [130]
        })
    
    def run_prediction_pipeline(self, 
                              date: Optional[str] = None,
                              include_betting_analysis: bool = True) -> Dict[str, Any]:
        """
        Run the prediction pipeline for a specific date.
        
        Args:
            date: Date to predict for (defaults to today)
            include_betting_analysis: Whether to include betting analysis
            
        Returns:
            Dictionary with predictions and analysis
        """
        logger.info("🔮 Starting NFL Prediction Pipeline...")
        
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
        
        results = {
            'prediction_date': date,
            'predictions': pd.DataFrame(),
            'betting_analysis': {},
            'model_confidence': 0.0,
            'errors': []
        }
        
        try:
            if not self.model:
                raise ValueError("No NFL model available for predictions")
            
            # Load trained model
            self.model._load_enhanced_nfl_model()
            
            # Get today's games
            todays_games = self.model._get_todays_nfl_games_from_odds_api()
            
            if todays_games.empty:
                logger.warning(f"⚪ No NFL games found for {date}")
                return results
            
            # Make predictions
            predictions = self.model._make_enhanced_nfl_predictions(todays_games)
            results['predictions'] = predictions
            
            # Calculate model confidence
            if 'home_win_prob' in predictions.columns:
                prob_variance = np.var(predictions['home_win_prob'])
                results['model_confidence'] = min(prob_variance * 10, 1.0)  # Scale to 0-1
            
            # Betting analysis
            if include_betting_analysis and self.betting_calc and 'home_odds' in predictions.columns:
                betting_analysis = self._analyze_betting_opportunities(predictions)
                results['betting_analysis'] = betting_analysis
            
            # Update pipeline state
            self.pipeline_state['last_prediction'] = datetime.now()
            
            logger.info(f"✅ NFL Prediction Pipeline Complete: {len(predictions)} games")
            
        except Exception as e:
            logger.error(f"❌ NFL Prediction Pipeline Failed: {e}")
            results['errors'].append(str(e))
        
        return results
    
    def _analyze_betting_opportunities(self, predictions: pd.DataFrame) -> Dict[str, Any]:
        """Analyze betting opportunities from predictions."""
        opportunities = []
        
        for _, game in predictions.iterrows():
            if 'home_ev' in game and 'away_ev' in game:
                # Home team analysis
                if game['home_ev'] > 0.05:  # 5% edge threshold
                    opportunities.append({
                        'game': f"{game['away_team_name']} @ {game['home_team_name']}",
                        'bet_type': 'Home ML',
                        'expected_value': game['home_ev'],
                        'confidence': game.get('confidence', 0.5),
                        'odds': game.get('home_odds', 0)
                    })
                
                # Away team analysis
                if game['away_ev'] > 0.05:
                    opportunities.append({
                        'game': f"{game['away_team_name']} @ {game['home_team_name']}",
                        'bet_type': 'Away ML',
                        'expected_value': game['away_ev'],
                        'confidence': game.get('confidence', 0.5),
                        'odds': game.get('away_odds', 0)
                    })
        
        # Sort by expected value
        opportunities.sort(key=lambda x: x['expected_value'], reverse=True)
        
        return {
            'total_opportunities': len(opportunities),
            'top_opportunities': opportunities[:5],  # Top 5
            'avg_expected_value': np.mean([opp['expected_value'] for opp in opportunities]) if opportunities else 0,
            'analysis_date': datetime.now().isoformat()
        }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and health."""
        status = {
            'pipeline_state': self.pipeline_state.copy(),
            'components_available': {
                'database': self.db is not None,
                'player_mapper': self.player_mapper is not None,
                'feature_engineer': self.feature_engineer is not None,
                'model': self.model is not None,
                'betting_calculator': self.betting_calc is not None
            },
            'model_status': self._check_model_status(),
            'data_freshness': self._check_data_freshness(),
            'system_health': 'unknown'
        }
        
        # Calculate system health
        component_health = sum(status['components_available'].values()) / len(status['components_available'])
        model_health = 1.0 if status['model_status']['model_available'] else 0.0
        data_health = min(status['data_freshness']['days_since_update'] / 7, 1.0) if status['data_freshness']['has_recent_data'] else 0.0
        
        overall_health = (component_health * 0.4 + model_health * 0.4 + data_health * 0.2)
        
        if overall_health > 0.8:
            status['system_health'] = 'excellent'
        elif overall_health > 0.6:
            status['system_health'] = 'good'
        elif overall_health > 0.4:
            status['system_health'] = 'fair'
        else:
            status['system_health'] = 'poor'
        
        return status
    
    def _check_model_status(self) -> Dict[str, Any]:
        """Check model availability and age."""
        model_path = Path('models/nfl/enhanced_hybrid_nfl_model.pkl')
        
        status = {
            'model_available': model_path.exists(),
            'model_age_days': None,
            'needs_retraining': False
        }
        
        if model_path.exists():
            model_age = datetime.now() - datetime.fromtimestamp(model_path.stat().st_mtime)
            status['model_age_days'] = model_age.days
            status['needs_retraining'] = model_age.days > 14  # Retrain every 2 weeks
        
        return status
    
    def _check_data_freshness(self) -> Dict[str, Any]:
        """Check data freshness."""
        # This would check the database for most recent data
        # For now, return placeholder
        return {
            'has_recent_data': True,
            'days_since_update': 1,
            'last_update': datetime.now().strftime('%Y-%m-%d')
        }


# Convenience functions
def run_nfl_training_pipeline(seasons: Optional[List[int]] = None,
                             force_retrain: bool = False,
                             use_enhanced_features: bool = True) -> Dict[str, Any]:
    """
    Convenience function to run NFL training pipeline.
    
    Args:
        seasons: Seasons to include in training
        force_retrain: Whether to force retraining
        use_enhanced_features: Whether to use enhanced features
        
    Returns:
        Pipeline results
    """
    pipeline = NFLDataPipeline(use_enhanced_features=use_enhanced_features)
    return pipeline.run_full_training_pipeline(seasons=seasons, force_retrain=force_retrain)


def run_nfl_prediction_pipeline(date: Optional[str] = None,
                               include_betting_analysis: bool = True) -> Dict[str, Any]:
    """
    Convenience function to run NFL prediction pipeline.
    
    Args:
        date: Date to predict for
        include_betting_analysis: Whether to include betting analysis
        
    Returns:
        Prediction results
    """
    pipeline = NFLDataPipeline()
    return pipeline.run_prediction_pipeline(date=date, include_betting_analysis=include_betting_analysis)


def get_nfl_pipeline_status() -> Dict[str, Any]:
    """Get NFL pipeline status."""
    pipeline = NFLDataPipeline()
    return pipeline.get_pipeline_status()


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='NFL Model Pipeline')
    parser.add_argument('--train', action='store_true', help='Run training pipeline')
    parser.add_argument('--predict', action='store_true', help='Run prediction pipeline') 
    parser.add_argument('--status', action='store_true', help='Show pipeline status')
    parser.add_argument('--force-retrain', action='store_true', help='Force model retraining')
    parser.add_argument('--date', type=str, help='Date for predictions (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    if args.train:
        print("🚀 Running NFL Training Pipeline...")
        results = run_nfl_training_pipeline(force_retrain=args.force_retrain)
        print(f"✅ Training complete. Steps: {results['steps_completed']}")
        
    elif args.predict:
        print("🔮 Running NFL Prediction Pipeline...")
        results = run_nfl_prediction_pipeline(date=args.date)
        print(f"✅ Predictions complete. {len(results['predictions'])} games analyzed")
        
    elif args.status:
        print("📊 NFL Pipeline Status:")
        status = get_nfl_pipeline_status()
        print(f"   System Health: {status['system_health']}")
        print(f"   Model Available: {status['model_status']['model_available']}")
        print(f"   Data Quality: {status['pipeline_state'].get('data_quality_score', 'Unknown')}")
        
    else:
        print("🏈 NFL Model Pipeline")
        print("Usage:")
        print("  --train              Run training pipeline")
        print("  --predict            Run prediction pipeline") 
        print("  --status             Show pipeline status")
        print("  --force-retrain      Force model retraining")
        print("  --date YYYY-MM-DD    Date for predictions")
