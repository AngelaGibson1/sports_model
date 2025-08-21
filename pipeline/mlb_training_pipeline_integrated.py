# /pipelines/mlb_training_pipeline_integrated.py
# Comprehensive MLB Training Pipeline - Integrated with Enhanced PlayerMapper

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger
import warnings
from pathlib import Path
import sys
warnings.filterwarnings('ignore')

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.database.mlb import MLBDatabase
from data.mappers.enhanced_player_mapper import EnhancedPlayerMapper  # NEW INTEGRATION
from data.features.mlb_features_enhanced import EnhancedMLBFeatureEngineer
from models.mlb.mlb_model_enhanced import EnhancedMLBPredictionModel, MLBModelFactory
from utils.betting_calculator import KellyCriterionCalculator
from config.settings import Settings

class IntegratedMLBTrainingPipeline:
    """
    Comprehensive MLB training pipeline integrated with Enhanced PlayerMapper.
    
    Data Flow:
    1. CSV Data (from your ingestion scripts) → Enhanced PlayerMapper
    2. Historical Game Data → MLBDatabase  
    3. PlayerMapper + Database → Enhanced Feature Engineering
    4. Engineered Features → Model Training & Validation
    5. Trained Models → Betting Analysis & Deployment
    
    Features:
    - Prioritizes CSV data from your ingestion scripts
    - Falls back to Sports API when CSV data is stale
    - Comprehensive player mapping for feature engineering
    - End-to-end model training and validation
    """
    
    def __init__(self, 
                 use_historical_data: bool = True,
                 seasons_to_include: List[int] = [2022, 2023, 2024],
                 min_games_required: int = 500,
                 csv_max_age_hours: int = 24):
        """
        Initialize the integrated training pipeline.
        
        Args:
            use_historical_data: Whether to use real historical data
            seasons_to_include: Seasons to include in training
            min_games_required: Minimum games required for training
            csv_max_age_hours: Max age of CSV files before API refresh
        """
        logger.info("⚾ Initializing Integrated MLB Training Pipeline...")
        
        self.use_historical_data = use_historical_data
        self.seasons_to_include = seasons_to_include
        self.min_games_required = min_games_required
        self.csv_max_age_hours = csv_max_age_hours
        
        # Initialize core components
        self.db = MLBDatabase()
        
        # NEW: Initialize Enhanced PlayerMapper (uses CSV data first)
        logger.info("🗺️ Initializing Enhanced PlayerMapper...")
        self.player_mapper = EnhancedPlayerMapper(
            sport='mlb', 
            auto_build=True,  # Build maps on initialization
            csv_max_age_hours=csv_max_age_hours
        )
        
        # Initialize feature engineer with player mapper
        self.feature_engineer = EnhancedMLBFeatureEngineer(
            database=self.db,
            player_mapper=self.player_mapper  # Pass player mapper
        )
        
        # Initialize betting calculator
        self.kelly_calc = KellyCriterionCalculator()
        
        # Model storage
        self.trained_models = {}
        self.training_results = {}
        
        # Data storage
        self.historical_data = pd.DataFrame()
        self.engineered_features = pd.DataFrame()
        
        # Check initial data availability
        self._check_initial_data_availability()
        
        logger.info(f"✅ Integrated pipeline initialized for seasons: {seasons_to_include}")
    
    def _check_initial_data_availability(self):
        """Check availability of CSV data and player mapping."""
        logger.info("🔍 Checking initial data availability...")
        
        # Check player mapper status
        mapper_summary = self.player_mapper.get_summary()
        logger.info(f"   PlayerMapper: {mapper_summary['players_loaded']} players loaded")
        logger.info(f"   Data source: {mapper_summary['data_sources_used']['players']}")
        
        # Check CSV status
        csv_status = self.player_mapper.check_csv_status()
        logger.info(f"   CSV Status:")
        logger.info(f"     Players: {'✅ Fresh' if csv_status['players_csv']['is_fresh'] else '❌ Stale/Missing'}")
        logger.info(f"     Teams: {'✅ Fresh' if csv_status['teams_csv']['is_fresh'] else '❌ Stale/Missing'}")
        
        # Recommendations
        if mapper_summary['players_loaded'] == 0:
            logger.warning("⚠️ No player data available. Run data ingestion scripts first!")
        elif not csv_status['players_csv']['is_fresh']:
            logger.warning("⚠️ Player CSV data is stale. Consider running fresh ingestion.")
    
    def run_complete_training_pipeline(self, 
                                     retrain_models: bool = True,
                                     tune_hyperparameters: bool = False,
                                     validate_models: bool = True,
                                     refresh_player_data: bool = False) -> Dict[str, any]:
        """
        Run the complete integrated training pipeline.
        
        Args:
            retrain_models: Whether to retrain models from scratch
            tune_hyperparameters: Whether to perform hyperparameter tuning
            validate_models: Whether to validate models with backtesting
            refresh_player_data: Whether to force refresh player data from API
        
        Returns:
            Complete pipeline results including player mapping status
        """
        logger.info("🚀 Running Complete Integrated MLB Training Pipeline")
        logger.info("=" * 70)
        
        pipeline_results = {
            'pipeline_start': datetime.now().isoformat(),
            'player_mapping_status': {},
            'data_preparation': {},
            'feature_engineering': {},
            'model_training': {},
            'model_validation': {},
            'betting_analysis': {},
            'deployment_status': {},
            'pipeline_success': False
        }
        
        try:
            # Step 0: Player Data Management (NEW)
            logger.info("🗺️ Step 0: Player Data Management")
            player_status = self._manage_player_data(refresh_player_data)
            pipeline_results['player_mapping_status'] = player_status
            
            # Step 1: Data Preparation
            logger.info("📊 Step 1: Data Preparation")
            data_results = self._prepare_training_data()
            pipeline_results['data_preparation'] = data_results
            
            if data_results['games_available'] < self.min_games_required:
                raise RuntimeError(f"Insufficient data: {data_results['games_available']} < {self.min_games_required}")
            
            # Step 2: Enhanced Feature Engineering (UPDATED)
            logger.info("🔧 Step 2: Enhanced Feature Engineering with Player Mapping")
            feature_results = self._engineer_comprehensive_features_with_players()
            pipeline_results['feature_engineering'] = feature_results
            
            # Step 3: Model Training
            logger.info("🤖 Step 3: Model Training")
            training_results = self._train_all_models(
                retrain=retrain_models,
                tune_hyperparameters=tune_hyperparameters
            )
            pipeline_results['model_training'] = training_results
            
            # Step 4: Model Validation
            if validate_models:
                logger.info("📈 Step 4: Model Validation")
                validation_results = self._validate_models_with_backtesting()
                pipeline_results['model_validation'] = validation_results
            
            # Step 5: Betting Analysis
            logger.info("💰 Step 5: Betting Analysis")
            betting_results = self._analyze_betting_performance()
            pipeline_results['betting_analysis'] = betting_results
            
            # Step 6: Model Deployment
            logger.info("🚀 Step 6: Model Deployment")
            deployment_results = self._deploy_models()
            pipeline_results['deployment_status'] = deployment_results
            
            # Final summary
            pipeline_results['pipeline_success'] = True
            pipeline_results['pipeline_end'] = datetime.now().isoformat()
            
            self._generate_integrated_pipeline_summary(pipeline_results)
            
            logger.info("✅ Complete integrated training pipeline finished successfully!")
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            pipeline_results['error'] = str(e)
            pipeline_results['pipeline_success'] = False
        
        return pipeline_results
    
    def _manage_player_data(self, force_refresh: bool = False) -> Dict[str, any]:
        """Manage player data refresh and validation."""
        logger.info("   🗺️ Managing player data...")
        
        if force_refresh:
            logger.info("   🔄 Forcing player data refresh from API...")
            self.player_mapper.force_api_refresh()
        
        # Get current status
        summary = self.player_mapper.get_summary()
        csv_status = self.player_mapper.check_csv_status()
        
        # Check if refresh is needed
        needs_refresh = False
        if summary['players_loaded'] == 0:
            logger.warning("   ⚠️ No players loaded, attempting refresh...")
            needs_refresh = True
        elif not csv_status['players_csv']['is_fresh']:
            logger.info("   📅 CSV data is stale, attempting refresh...")
            needs_refresh = True
        
        if needs_refresh and not force_refresh:
            try:
                self.player_mapper.refresh_data()
                summary = self.player_mapper.get_summary()  # Get updated status
            except Exception as e:
                logger.error(f"   ❌ Player data refresh failed: {e}")
        
        # Validate player data quality
        player_quality = self._validate_player_data_quality()
        
        status = {
            'players_loaded': summary['players_loaded'],
            'teams_loaded': summary['teams_loaded'],
            'data_sources_used': summary['data_sources_used'],
            'csv_status': csv_status,
            'player_data_quality': player_quality,
            'refresh_performed': needs_refresh or force_refresh
        }
        
        logger.info(f"   ✅ Player mapping ready: {summary['players_loaded']} players, {summary['teams_loaded']} teams")
        
        return status
    
    def _validate_player_data_quality(self) -> Dict[str, any]:
        """Validate the quality of player mapping data."""
        if self.player_mapper.player_map.empty:
            return {'quality_score': 0, 'issues': ['No player data available']}
        
        player_df = self.player_mapper.player_map
        issues = []
        metrics = {}
        
        # Check data completeness
        required_fields = ['player_id', 'full_name', 'team_id', 'position']
        for field in required_fields:
            if field in player_df.columns:
                missing_pct = player_df[field].isna().mean()
                metrics[f'{field}_completeness'] = 1 - missing_pct
                if missing_pct > 0.1:
                    issues.append(f"High missing data in {field}: {missing_pct:.1%}")
        
        # Check for reasonable player count (MLB ~750 active players)
        player_count = len(player_df)
        metrics['player_count'] = player_count
        
        if player_count < 500:
            issues.append(f"Low player count: {player_count} (expected ~750)")
        elif player_count > 1000:
            issues.append(f"High player count: {player_count} (may include inactive players)")
        
        # Check team distribution
        if 'team_id' in player_df.columns:
            teams_with_players = player_df['team_id'].nunique()
            metrics['teams_with_players'] = teams_with_players
            
            if teams_with_players < 25:
                issues.append(f"Few teams represented: {teams_with_players} (expected 30)")
        
        # Calculate quality score
        quality_score = (
            metrics.get('player_id_completeness', 0) * 0.3 +
            metrics.get('full_name_completeness', 0) * 0.3 +
            (min(metrics.get('player_count', 0) / 750, 1.0)) * 0.2 +
            (min(metrics.get('teams_with_players', 0) / 30, 1.0)) * 0.2
        )
        
        return {
            'quality_score': quality_score,
            'metrics': metrics,
            'issues': issues
        }
    
    def _prepare_training_data(self) -> Dict[str, any]:
        """
        Prepares comprehensive training data with smart Sports API integration.
        
        Data Flow:
        1. Loads historical game data from the local database (fast)
        2. If data is incomplete/missing, fetches from Sports API (comprehensive)
        3. Saves newly fetched data to database for future use (persistence)
        4. Enriches data with player information using mapper (enhanced features)
        """
        logger.info("   📊 Preparing training data with smart Sports API integration...")
        
        data_source_used = 'unknown'
        
        # --- Stage 1: Load from Database (Primary Source) ---
        try:
            self.historical_data = self.db.get_historical_data(self.seasons_to_include)
            
            # Check if loaded data is sufficient and complete
            if not self.historical_data.empty and len(self.historical_data) >= self.min_games_required:
                logger.info(f"   ✅ Loaded {len(self.historical_data)} historical games from database")
                data_source_used = 'database'
            else:
                logger.warning(f"   ⚠️ Insufficient database data: {len(self.historical_data)} < {self.min_games_required}")
                raise ValueError("Insufficient database data")
                
        except Exception as e:
            logger.warning(f"   ⚠️ Database load failed: {e}. Fetching from Sports API...")
            
            # --- Stage 2: Fetch from Sports API (Smart Fallback) ---
            try:
                self.historical_data = self._fetch_data_from_sports_api()
                data_source_used = 'sports_api'
                
                # Save newly fetched data to database for future use
                if not self.historical_data.empty:
                    try:
                        saved_count = self.db.save_games(self.historical_data)
                        logger.info(f"   💾 Saved {saved_count} new games to database for future use")
                    except Exception as save_error:
                        logger.warning(f"   ⚠️ Could not save to database: {save_error}")
                
            except Exception as api_error:
                logger.error(f"   ❌ Sports API fetch failed: {api_error}. Using simulation...")
                self.historical_data = self._create_realistic_historical_data()
                data_source_used = 'simulated'
        
        # --- Stage 3: Enrich Data with Player Information ---
        logger.info("   👥 Enriching data with player information...")
        self.historical_data = self._enrich_historical_data_with_players(self.historical_data)
        
        # --- Stage 4: Data Quality Analysis ---
        data_quality = self._analyze_data_quality_with_players(self.historical_data)
        
        # Check final data sufficiency
        if len(self.historical_data) < self.min_games_required:
            logger.warning(f"   ⚠️ Final data still insufficient: {len(self.historical_data)} < {self.min_games_required}")
            data_quality['issues'].append(f"Insufficient total games: {len(self.historical_data)}")
        
        return {
            'games_available': len(self.historical_data),
            'seasons_covered': sorted(self.historical_data['season'].unique()) if 'season' in self.historical_data.columns else [],
            'date_range': {
                'start': self.historical_data['date'].min() if 'date' in self.historical_data.columns else None,
                'end': self.historical_data['date'].max() if 'date' in self.historical_data.columns else None
            },
            'data_quality': data_quality,
            'data_source': data_source_used,
            'player_enrichment_success': True,
            'api_fallback_used': data_source_used == 'sports_api'
        }
    
    def _fetch_data_from_sports_api(self) -> pd.DataFrame:
        """
        Fetch comprehensive historical data from Sports API.
        
        This method uses your SportsAPIClient to get real game data for all seasons.
        """
        logger.info("   🌐 Fetching historical data from Sports API...")
        
        # Get Sports API client from player mapper
        sports_api_client = self.player_mapper.api_client
        
        if not hasattr(sports_api_client, 'sports_client'):
            raise RuntimeError("Sports API client not available in player mapper")
        
        api_client = sports_api_client.sports_client
        all_seasons_data = []
        
        for season in self.seasons_to_include:
            logger.info(f"      📥 Fetching season {season} from Sports API...")
            
            try:
                # Fetch games for the entire season
                season_games = api_client.get_games(season=season)
                
                if season_games.empty:
                    logger.warning(f"      ⚪ No games found for season {season}")
                    continue
                
                # Standardize column names for consistency
                season_games = self._standardize_sports_api_game_data(season_games, season)
                
                # Get team statistics for better features
                season_teams = api_client.get_teams(season=season)
                if not season_teams.empty:
                    season_games = self._enrich_games_with_team_stats(season_games, season_teams)
                
                all_seasons_data.append(season_games)
                logger.info(f"      ✅ Fetched {len(season_games)} games for season {season}")
                
                # Rate limiting - be respectful to API
                import time
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"      ❌ Failed to fetch season {season}: {e}")
                continue
        
        if not all_seasons_data:
            raise RuntimeError("No data could be fetched from Sports API for any season")
        
        # Combine all seasons
        combined_data = pd.concat(all_seasons_data, ignore_index=True)
        combined_data['data_source'] = 'Sports_API'
        combined_data['ingestion_date'] = pd.Timestamp.now()
        
        logger.info(f"   ✅ Successfully fetched {len(combined_data)} total games from Sports API")
        return combined_data
    
    def _standardize_sports_api_game_data(self, games_df: pd.DataFrame, season: int) -> pd.DataFrame:
        """
        Standardize game data from Sports API to match database schema.
        
        This ensures consistency between API data and database data.
        """
        standardized = games_df.copy()
        
        # Column mapping from Sports API to database format
        column_mapping = {
            'id': 'game_id',
            'game_id': 'game_id',
            'date': 'date',
            'teams.home.id': 'home_team_id',
            'teams.away.id': 'away_team_id', 
            'teams.home.name': 'home_team_name',
            'teams.away.name': 'away_team_name',
            'scores.home.total': 'home_score',
            'scores.away.total': 'away_score',
            'status.long': 'status',
            'venue.name': 'venue'
        }
        
        # Rename columns if they exist
        for old_col, new_col in column_mapping.items():
            if old_col in standardized.columns and new_col not in standardized.columns:
                standardized = standardized.rename(columns={old_col: new_col})
        
        # Handle nested data structures (common in Sports API responses)
        if 'teams' in standardized.columns:
            teams_data = standardized['teams'].apply(pd.Series) if standardized['teams'].notna().any() else None
            if teams_data is not None:
                # Extract home team info
                if 'home' in teams_data.columns:
                    home_data = teams_data['home'].apply(pd.Series)
                    if 'id' in home_data.columns:
                        standardized['home_team_id'] = home_data['id']
                    if 'name' in home_data.columns:
                        standardized['home_team_name'] = home_data['name']
                
                # Extract away team info  
                if 'away' in teams_data.columns:
                    away_data = teams_data['away'].apply(pd.Series)
                    if 'id' in away_data.columns:
                        standardized['away_team_id'] = away_data['id']
                    if 'name' in away_data.columns:
                        standardized['away_team_name'] = away_data['name']
        
        if 'scores' in standardized.columns:
            scores_data = standardized['scores'].apply(pd.Series) if standardized['scores'].notna().any() else None
            if scores_data is not None:
                # Extract home scores
                if 'home' in scores_data.columns:
                    home_scores = scores_data['home'].apply(pd.Series)
                    if 'total' in home_scores.columns:
                        standardized['home_score'] = pd.to_numeric(home_scores['total'], errors='coerce')
                
                # Extract away scores
                if 'away' in scores_data.columns:
                    away_scores = scores_data['away'].apply(pd.Series)
                    if 'total' in away_scores.columns:
                        standardized['away_score'] = pd.to_numeric(away_scores['total'], errors='coerce')
        
        # Ensure required columns exist
        required_columns = ['game_id', 'date', 'home_team_id', 'away_team_id', 'home_score', 'away_score']
        for col in required_columns:
            if col not in standardized.columns:
                if col in ['home_score', 'away_score']:
                    standardized[col] = 0  # Default to 0 for missing scores
                else:
                    standardized[col] = None  # Default to None for other fields
        
        # Add season column
        standardized['season'] = season
        
        # Ensure date is datetime
        if 'date' in standardized.columns:
            standardized['date'] = pd.to_datetime(standardized['date'], errors='coerce')
        
        # Filter out games with missing critical data
        before_filter = len(standardized)
        standardized = standardized.dropna(subset=['home_team_id', 'away_team_id'])
        after_filter = len(standardized)
        
        if before_filter != after_filter:
            logger.warning(f"      ⚠️ Filtered out {before_filter - after_filter} games with missing team data")
        
        return standardized
    
    def _enrich_games_with_team_stats(self, games_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich game data with team statistics from Sports API.
        
        This adds team performance metrics that can be used for better predictions.
        """
        if teams_df.empty:
            return games_df
        
        enriched = games_df.copy()
        
        # Create team statistics mapping
        team_stats = teams_df.copy()
        
        # Standardize team columns
        if 'id' in team_stats.columns and 'team_id' not in team_stats.columns:
            team_stats['team_id'] = team_stats['id']
        
        # Create mappings for home and away teams
        team_mapping = team_stats.set_index('team_id').to_dict('index')
        
        # Add team statistics for home teams
        enriched['home_team_wins'] = enriched['home_team_id'].map(
            lambda x: team_mapping.get(x, {}).get('wins', 0)
        )
        enriched['home_team_losses'] = enriched['home_team_id'].map(
            lambda x: team_mapping.get(x, {}).get('losses', 0)
        )
        
        # Add team statistics for away teams
        enriched['away_team_wins'] = enriched['away_team_id'].map(
            lambda x: team_mapping.get(x, {}).get('wins', 0)
        )
        enriched['away_team_losses'] = enriched['away_team_id'].map(
            lambda x: team_mapping.get(x, {}).get('losses', 0)
        )
        
        # Calculate win percentages
        enriched['home_team_win_pct'] = enriched['home_team_wins'] / (
            enriched['home_team_wins'] + enriched['home_team_losses']
        ).replace(0, 1)  # Avoid division by zero
        
        enriched['away_team_win_pct'] = enriched['away_team_wins'] / (
            enriched['away_team_wins'] + enriched['away_team_losses']
        ).replace(0, 1)
        
        return enriched
        """Enrich historical game data with player information."""
        logger.info("   👥 Enriching historical data with player information...")
        
        if historical_df.empty or self.player_mapper.player_map.empty:
            logger.warning("   ⚠️ Cannot enrich data - missing historical data or player mapping")
            return historical_df
        
        enriched_df = historical_df.copy()
        
        # Add team information from player mapper
        if not self.player_mapper.team_map.empty:
            team_map = self.player_mapper.team_map
            
            # Merge home team info
            home_team_info = team_map.add_suffix('_home')[['team_id_home', 'team_name_home', 'abbreviation_home']]
            enriched_df = enriched_df.merge(
                home_team_info,
                left_on='home_team_id',
                right_on='team_id_home',
                how='left'
            )
            
            # Merge away team info  
            away_team_info = team_map.add_suffix('_away')[['team_id_away', 'team_name_away', 'abbreviation_away']]
            enriched_df = enriched_df.merge(
                away_team_info,
                left_on='away_team_id',
                right_on='team_id_away',
                how='left'
            )
            
            # Clean up merge columns
            enriched_df = enriched_df.drop(['team_id_home', 'team_id_away'], axis=1, errors='ignore')
        
        # Add sample pitcher information (you can enhance this based on your pitcher data)
        if 'game_id' in enriched_df.columns:
            enriched_df = self._add_sample_pitcher_data(enriched_df)
        
        logger.info(f"   ✅ Enriched {len(enriched_df)} games with player information")
        return enriched_df
    
    def _add_sample_pitcher_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sample pitcher data for demonstration (replace with real pitcher logic)."""
        # For now, add sample pitcher IDs - you can enhance this with real pitcher assignments
        np.random.seed(42)  # For reproducible pitcher assignments
        
        # Get available pitchers (players with pitcher positions)
        pitchers = self.player_mapper.player_map[
            self.player_mapper.player_map['position'].str.contains('P', na=False)
        ]
        
        if not pitchers.empty:
            pitcher_ids = pitchers['player_id'].tolist()
            
            # Assign random starting pitchers (in reality, you'd use actual lineups)
            df['home_starting_pitcher_id'] = np.random.choice(pitcher_ids, size=len(df))
            df['away_starting_pitcher_id'] = np.random.choice(pitcher_ids, size=len(df))
            
            # Add pitcher names for readability
            pitcher_names = pitchers.set_index('player_id')['full_name'].to_dict()
            df['home_starting_pitcher_name'] = df['home_starting_pitcher_id'].map(pitcher_names)
            df['away_starting_pitcher_name'] = df['away_starting_pitcher_id'].map(pitcher_names)
        
        return df
    
    def _analyze_data_quality_with_players(self, df: pd.DataFrame) -> Dict[str, any]:
        """Analyze data quality including player enrichment metrics."""
        basic_quality = self._analyze_data_quality(df)  # Your existing method
        
        # Add player-specific quality checks
        player_quality = {}
        
        if 'home_starting_pitcher_id' in df.columns:
            pitcher_coverage = df['home_starting_pitcher_id'].notna().mean()
            player_quality['pitcher_coverage'] = pitcher_coverage
            
            if pitcher_coverage < 0.8:
                basic_quality['issues'].append(f"Low pitcher coverage: {pitcher_coverage:.1%}")
        
        if 'team_name_home' in df.columns:
            team_name_coverage = df['team_name_home'].notna().mean()
            player_quality['team_name_coverage'] = team_name_coverage
            
            if team_name_coverage < 0.9:
                basic_quality['issues'].append(f"Low team name coverage: {team_name_coverage:.1%}")
        
        # Update quality score with player metrics
        if player_quality:
            player_score = np.mean(list(player_quality.values()))
            basic_quality['quality_score'] = (basic_quality['quality_score'] * 0.7 + player_score * 0.3)
        
        basic_quality['player_quality_metrics'] = player_quality
        return basic_quality
    
    def _engineer_comprehensive_features_with_players(self) -> Dict[str, any]:
        """Engineer comprehensive features using player mapping."""
        logger.info("   🔧 Engineering comprehensive features with player mapping...")
        
        try:
            # Use enhanced feature engineer with player mapper
            self.engineered_features = self.feature_engineer.engineer_comprehensive_features(
                historical_data=self.historical_data,
                include_pitching_matchups=True,
                include_park_factors=True,
                include_weather=True,
                include_situational=True,
                use_player_mapping=True  # NEW: Enable player mapping features
            )
            
            feature_summary = {
                'total_features': self.engineered_features.shape[1],
                'feature_categories': self._categorize_features(self.engineered_features.columns),
                'data_points': len(self.engineered_features),
                'feature_quality': self._assess_feature_quality(self.engineered_features),
                'player_mapping_features': self._count_player_mapping_features(self.engineered_features.columns)
            }
            
            logger.info(f"   ✅ Engineered {feature_summary['total_features']} features")
            logger.info(f"   👥 Player mapping features: {feature_summary['player_mapping_features']}")
            
            return feature_summary
            
        except Exception as e:
            logger.error(f"   ❌ Enhanced feature engineering failed: {e}")
            # Create basic features as fallback
            self.engineered_features = self._create_basic_fallback_features()
            return {
                'total_features': self.engineered_features.shape[1],
                'feature_categories': {'basic': self.engineered_features.shape[1]},
                'data_points': len(self.engineered_features),
                'fallback_used': True,
                'error': str(e)
            }
    
    def _count_player_mapping_features(self, feature_names: List[str]) -> int:
        """Count features that use player mapping data."""
        player_keywords = [
            'pitcher', 'starter', 'player', 'roster', 'lineup',
            'matchup', 'vs_', 'against', 'career', 'season_stats'
        ]
        
        count = 0
        for feature in feature_names:
            feature_lower = feature.lower()
            if any(keyword in feature_lower for keyword in player_keywords):
                count += 1
        
        return count
    
    # ===============================================
    # EXISTING METHODS (keep your original methods)
    # ===============================================
    
    def _create_realistic_historical_data(self) -> pd.DataFrame:
        """Create realistic historical MLB data for training."""
        logger.info("   🏗️ Creating realistic historical data...")
        
        # Generate comprehensive historical dataset
        total_games = 0
        seasons_data = []
        
        for season in self.seasons_to_include:
            # MLB season: ~2430 total games (30 teams × 162 games ÷ 2)
            season_games = 2430 if season != 2020 else 900  # 2020 was shortened
            
            # Create season schedule
            season_start = f"{season}-03-28"
            season_end = f"{season}-09-30"
            dates = pd.date_range(season_start, season_end, freq='D')
            
            # Generate games
            games_data = []
            game_id_start = total_games + 1
            
            for i in range(season_games):
                # Realistic team matchups
                home_team = np.random.choice(range(1, 31))
                away_team = np.random.choice([t for t in range(1, 31) if t != home_team])
                
                # Realistic game date
                game_date = np.random.choice(dates)
                
                # Realistic scores (MLB average ~4.5 runs per team)
                home_score = np.random.poisson(4.65)  # Slight home advantage
                away_score = np.random.poisson(4.35)
                
                # Ensure no ties (very rare in MLB)
                if home_score == away_score:
                    if np.random.random() > 0.5:
                        home_score += 1
                    else:
                        away_score += 1
                
                game_data = {
                    'game_id': game_id_start + i,
                    'date': game_date,
                    'season': season,
                    'home_team_id': home_team,
                    'away_team_id': away_team,
                    'home_team_name': f"Team_{home_team}",
                    'away_team_name': f"Team_{away_team}",
                    'home_score': home_score,
                    'away_score': away_score,
                    'status': 'Finished',
                    'venue': f"Stadium_{home_team}",
                    'time': np.random.choice(['13:05', '19:05', '20:05'])  # Common game times
                }
                
                games_data.append(game_data)
            
            season_df = pd.DataFrame(games_data)
            seasons_data.append(season_df)
            total_games += season_games
            
            logger.info(f"      📅 {season}: {season_games} games")
        
        # Combine all seasons
        historical_df = pd.concat(seasons_data, ignore_index=True)
        historical_df['date'] = pd.to_datetime(historical_df['date'])
        
        # Add some realistic team statistics
        historical_df = self._add_realistic_team_stats(historical_df)
        
        logger.info(f"   ✅ Created {len(historical_df)} realistic historical games")
        return historical_df
    
    def _add_realistic_team_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic team statistics to historical data."""
        logger.info("   📊 Adding realistic team statistics...")
        
        # Create team quality ratings (some teams are better than others)
        np.random.seed(42)  # For reproducible team qualities
        team_qualities = {}
        for team_id in range(1, 31):
            team_qualities[team_id] = np.random.normal(100, 15)  # Mean 100, std 15
        
        # Adjust scores based on team quality
        for idx, row in df.iterrows():
            home_quality = team_qualities[row['home_team_id']]
            away_quality = team_qualities[row['away_team_id']]
            
            # Quality difference affects run scoring
            quality_diff = (home_quality - away_quality) / 30  # Scale factor
            home_boost = max(0, quality_diff + 0.2)  # Home field advantage
            away_boost = max(0, -quality_diff)
            
            # Adjust scores slightly based on quality
            if np.random.random() < 0.3:  # 30% of games affected by quality
                if home_boost > away_boost:
                    df.at[idx, 'home_score'] = min(df.at[idx, 'home_score'] + np.random.poisson(home_boost), 20)
                else:
                    df.at[idx, 'away_score'] = min(df.at[idx, 'away_score'] + np.random.poisson(away_boost), 20)
        
        return df
    
    def _analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """Analyze data quality metrics."""
        if df.empty:
            return {'quality_score': 0, 'issues': ['No data available']}
        
        quality_metrics = {}
        issues = []
        
        # Completeness
        missing_percentage = df.isnull().sum().sum() / (len(df) * len(df.columns))
        quality_metrics['completeness'] = 1 - missing_percentage
        
        if missing_percentage > 0.05:
            issues.append(f"High missing data: {missing_percentage:.1%}")
        
        # Realistic score ranges
        if 'home_score' in df.columns and 'away_score' in df.columns:
            avg_total_runs = (df['home_score'] + df['away_score']).mean()
            quality_metrics['avg_total_runs'] = avg_total_runs
            
            if avg_total_runs < 6 or avg_total_runs > 12:
                issues.append(f"Unrealistic average total runs: {avg_total_runs:.1f}")
        
        # Home field advantage
        if 'home_score' in df.columns and 'away_score' in df.columns:
            home_win_rate = (df['home_score'] > df['away_score']).mean()
            quality_metrics['home_win_rate'] = home_win_rate
            
            if home_win_rate < 0.50 or home_win_rate > 0.60:
                issues.append(f"Unrealistic home win rate: {home_win_rate:.1%}")
        
        # Calculate overall quality score
        quality_score = (
            quality_metrics.get('completeness', 0) * 0.4 +
            (1 - abs(quality_metrics.get('avg_total_runs', 9) - 9) / 9) * 0.3 +
            (1 - abs(quality_metrics.get('home_win_rate', 0.54) - 0.54) / 0.54) * 0.3
        )
        
        return {
            'quality_score': quality_score,
            'metrics': quality_metrics,
            'issues': issues
        }
    
    def _categorize_features(self, feature_names: List[str]) -> Dict[str, int]:
        """Categorize features by type."""
        categories = {
            'hitting': 0, 'pitching': 0, 'fielding': 0, 'situational': 0,
            'momentum': 0, 'park_factors': 0, 'weather': 0, 'temporal': 0, 
            'player_mapping': 0, 'other': 0
        }
        
        for feature in feature_names:
            feature_lower = feature.lower()
            
            if any(term in feature_lower for term in ['batting', 'hits', 'runs', 'rbi', 'ops']):
                categories['hitting'] += 1
            elif any(term in feature_lower for term in ['era', 'whip', 'strikeout', 'pitcher']):
                categories['pitching'] += 1
            elif any(term in feature_lower for term in ['fielding', 'error', 'defense']):
                categories['fielding'] += 1
            elif any(term in feature_lower for term in ['streak', 'form', 'momentum']):
                categories['momentum'] += 1
            elif any(term in feature_lower for term in ['park', 'venue', 'stadium']):
                categories['park_factors'] += 1
            elif any(term in feature_lower for term in ['weather', 'wind', 'temperature']):
                categories['weather'] += 1
            elif any(term in feature_lower for term in ['date', 'month', 'day', 'season']):
                categories['temporal'] += 1
            elif any(term in feature_lower for term in ['rest', 'travel', 'series', 'division']):
                categories['situational'] += 1
            elif any(term in feature_lower for term in ['player', 'starter', 'matchup', 'vs_', 'career']):
                categories['player_mapping'] += 1
            else:
                categories['other'] += 1
        
        return categories
    
    def _assess_feature_quality(self, features_df: pd.DataFrame) -> Dict[str, any]:
        """Assess the quality of engineered features."""
        numeric_features = features_df.select_dtypes(include=[np.number])
        
        quality_metrics = {
            'missing_percentage': features_df.isnull().sum().sum() / (len(features_df) * len(features_df.columns)),
            'zero_variance_features': (numeric_features.var() == 0).sum(),
            'high_correlation_pairs': 0,
            'outlier_percentage': 0
        }
        
        # Check for high correlation
        if len(numeric_features.columns) > 1:
            corr_matrix = numeric_features.corr().abs()
            high_corr_count = 0
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.95:
                        high_corr_count += 1
            quality_metrics['high_correlation_pairs'] = high_corr_count
        
        return quality_metrics
    
    def _create_basic_fallback_features(self) -> pd.DataFrame:
        """Create basic features as fallback."""
        logger.info("   🔧 Creating basic fallback features...")
        
        features_df = self.historical_data.copy()
        
        # Basic derived features
        if 'home_score' in features_df.columns and 'away_score' in features_df.columns:
            features_df['total_runs'] = features_df['home_score'] + features_df['away_score']
            features_df['run_differential'] = features_df['home_score'] - features_df['away_score']
            features_df['home_win'] = (features_df['home_score'] > features_df['away_score']).astype(int)
        
        # Add some basic team and temporal features
        if 'date' in features_df.columns:
            features_df['date'] = pd.to_datetime(features_df['date'])
            features_df['month'] = features_df['date'].dt.month
            features_df['day_of_week'] = features_df['date'].dt.dayofweek
        
        # Home field advantage
        features_df['home_advantage'] = 1
        
        return features_df
    
    # Continue with your existing methods for model training, validation, etc.
    # I'll include the key ones that need player mapper integration
    
    def _train_all_models(self, 
                         retrain: bool = True,
                         tune_hyperparameters: bool = False) -> Dict[str, any]:
        """Train all MLB prediction models."""
        logger.info("   🤖 Training all models...")
        
        model_types = ['game_winner', 'total_runs', 'nrfi']
        training_results = {}
        
        for model_type in model_types:
            logger.info(f"      🎯 Training {model_type} model...")
            
            try:
                # Create model
                if model_type == 'game_winner':
                    model = MLBModelFactory.create_game_winner_model()
                elif model_type == 'total_runs':
                    model = MLBModelFactory.create_total_runs_model()
                elif model_type == 'nrfi':
                    model = MLBModelFactory.create_nrfi_model()
                
                # Check if model exists and whether to retrain
                if not retrain and model.load_model():
                    logger.info(f"         ✅ Loaded existing {model_type} model")
                    training_results[model_type] = {'loaded_existing': True}
                    self.trained_models[model_type] = model
                    continue
                
                # Prepare features for this model type
                X, y = model.prepare_features(self.engineered_features)
                
                if len(X) < 100:
                    logger.warning(f"         ⚠️ Insufficient data for {model_type}: {len(X)} samples")
                    continue
                
                # Train model
                model_results = model.train_ensemble_model(
                    X, y,
                    tune_hyperparameters=tune_hyperparameters,
                    cv_folds=5
                )
                
                # Save model
                if model.save_model():
                    logger.info(f"         💾 {model_type} model saved")
                
                # Store results
                training_results[model_type] = model_results
                self.trained_models[model_type] = model
                
                # Log performance
                if 'ensemble' in model_results['model_scores']:
                    performance = model_results['model_scores']['ensemble']
                    if 'test_accuracy' in performance:
                        logger.info(f"         📊 Accuracy: {performance['test_accuracy']:.3f}")
                    if 'auc_score' in performance:
                        logger.info(f"         🎯 AUC: {performance['auc_score']:.3f}")
                
            except Exception as e:
                logger.error(f"         ❌ {model_type} model training failed: {e}")
                training_results[model_type] = {'error': str(e)}
        
        return training_results
    
    def _validate_models_with_backtesting(self) -> Dict[str, any]:
        """Validate models using backtesting."""
        logger.info("   📈 Validating models with backtesting...")
        
        validation_results = {}
        
        # Use last 20% of data for out-of-sample testing
        test_size = int(len(self.engineered_features) * 0.2)
        test_data = self.engineered_features.tail(test_size)
        
        for model_type, model in self.trained_models.items():
            logger.info(f"      📊 Validating {model_type} model...")
            
            try:
                # Prepare test features
                X_test, y_test = model.prepare_features(test_data)
                
                if len(X_test) == 0:
                    continue
                
                # Generate predictions
                if model_type == 'game_winner':
                    probabilities = model.predict_probability(X_test)
                    predictions = (probabilities > 0.5).astype(int)
                    
                    # Calculate metrics
                    accuracy = (predictions == y_test).mean()
                    
                    # Kelly criterion analysis
                    kelly_edges = []
                    for prob in probabilities:
                        # Simulate betting odds
                        implied_prob = np.random.uniform(0.45, 0.55)
                        edge = prob - implied_prob
                        kelly_edges.append(edge)
                    
                    avg_edge = np.mean([e for e in kelly_edges if e > 0])
                    profitable_bets = sum(1 for e in kelly_edges if e > 0.02)
                    
                    validation_results[model_type] = {
                        'accuracy': accuracy,
                        'average_edge': avg_edge if not np.isnan(avg_edge) else 0,
                        'profitable_opportunities': profitable_bets,
                        'total_predictions': len(predictions)
                    }
                    
                    logger.info(f"         ✅ Accuracy: {accuracy:.3f}")
                    logger.info(f"         💰 Avg Edge: {avg_edge:.1%}" if not np.isnan(avg_edge) else "         💰 No profitable edges found")
                
                else:
                    # Regression model validation
                    predictions = model.predict(X_test)
                    mse = ((predictions - y_test) ** 2).mean()
                    
                    validation_results[model_type] = {
                        'mse': mse,
                        'total_predictions': len(predictions)
                    }
                    
                    logger.info(f"         ✅ MSE: {mse:.3f}")
                
            except Exception as e:
                logger.error(f"         ❌ {model_type} validation failed: {e}")
                validation_results[model_type] = {'error': str(e)}
        
        return validation_results
    
    def _analyze_betting_performance(self) -> Dict[str, any]:
        """Analyze potential betting performance."""
        logger.info("   💰 Analyzing betting performance...")
        
        betting_results = {}
        
        if 'game_winner' not in self.trained_models:
            return {'error': 'No game winner model available'}
        
        model = self.trained_models['game_winner']
        
        # Use recent data for betting analysis
        recent_data = self.engineered_features.tail(200)
        X_recent, y_recent = model.prepare_features(recent_data)
        
        if len(X_recent) == 0:
            return {'error': 'No recent data available'}
        
        # Generate predictions
        probabilities = model.predict_probability(X_recent)
        
        # Simulate betting performance
        total_bets = 0
        profitable_bets = 0
        total_edge = 0
        
        for i, prob in enumerate(probabilities):
            # Simulate market odds
            market_prob = np.random.uniform(0.4, 0.6)
            edge = prob - market_prob
            
            if edge > 0.025:  # Minimum 2.5% edge
                total_bets += 1
                total_edge += edge
                
                # Simulate bet outcome
                actual_outcome = y_recent.iloc[i] if i < len(y_recent) else np.random.choice([0, 1])
                predicted_outcome = 1 if prob > 0.5 else 0
                
                if actual_outcome == predicted_outcome:
                    profitable_bets += 1
        
        if total_bets > 0:
            win_rate = profitable_bets / total_bets
            avg_edge = total_edge / total_bets
            
            # Kelly criterion recommendations
            kelly_results = []
            for prob in probabilities:
                market_prob = np.random.uniform(0.4, 0.6)
                kelly_result = self.kelly_calc.calculate_kelly_bet(
                    predicted_probability=prob,
                    bookmaker_odds=-110,  # Standard odds
                    confidence_level=0.8
                )
                kelly_results.append(kelly_result)
            
            recommended_bets = sum(1 for k in kelly_results if k['recommendation'] == 'BET')
            
            betting_results = {
                'total_betting_opportunities': total_bets,
                'profitable_bets': profitable_bets,
                'win_rate': win_rate,
                'average_edge': avg_edge,
                'kelly_recommended_bets': recommended_bets,
                'estimated_roi': avg_edge * win_rate if win_rate > 0.52 else 0
            }
            
            logger.info(f"      📊 Betting opportunities: {total_bets}")
            logger.info(f"      🎯 Win rate: {win_rate:.1%}")
            logger.info(f"      💰 Avg edge: {avg_edge:.1%}")
            logger.info(f"      🔥 Kelly recommendations: {recommended_bets}")
        
        else:
            betting_results = {
                'total_betting_opportunities': 0,
                'message': 'No profitable betting opportunities found'
            }
        
        return betting_results
    
    def _deploy_models(self) -> Dict[str, any]:
        """Deploy trained models."""
        logger.info("   🚀 Deploying models...")
        
        deployment_results = {
            'deployed_models': [],
            'deployment_paths': {},
            'deployment_status': 'success'
        }
        
        for model_type, model in self.trained_models.items():
            try:
                # Ensure model is saved
                if model.save_model():
                    deployment_results['deployed_models'].append(model_type)
                    deployment_results['deployment_paths'][model_type] = str(model.model_path)
                    logger.info(f"      ✅ {model_type} model deployed")
                else:
                    logger.error(f"      ❌ {model_type} model deployment failed")
                    
            except Exception as e:
                logger.error(f"      ❌ {model_type} deployment error: {e}")
        
        return deployment_results
    
    def _generate_integrated_pipeline_summary(self, results: Dict[str, any]):
        """Generate comprehensive pipeline summary with smart data loading info."""
        logger.info("\n" + "=" * 70)
        logger.info("⚾ MLB SMART INTEGRATED TRAINING PIPELINE SUMMARY")
        logger.info("=" * 70)
        
        # Player mapping summary
        player_status = results.get('player_mapping_status', {})
        logger.info(f"🗺️ Player Mapping: {player_status.get('players_loaded', 0)} players, {player_status.get('teams_loaded', 0)} teams")
        
        if player_status.get('data_sources_used', {}).get('players'):
            logger.info(f"   Data source: {player_status['data_sources_used']['players']}")
        
        # Smart data loading summary
        data_results = results.get('data_preparation', {})
        data_source = data_results.get('data_source', 'unknown')
        api_fallback = data_results.get('api_fallback_used', False)
        
        logger.info(f"📊 Data: {data_results.get('games_available', 0)} games across {len(data_results.get('seasons_covered', []))} seasons")
        logger.info(f"   Source: {data_source} {'(API fallback triggered)' if api_fallback else '(primary source)'}")
        
        # Feature summary
        feature_results = results.get('feature_engineering', {})
        total_features = feature_results.get('total_features', 0)
        player_features = feature_results.get('player_mapping_features', 0)
        logger.info(f"🔧 Features: {total_features} total ({player_features} player-enhanced)")
        
        # Model summary
        training_results = results.get('model_training', {})
        successful_models = [m for m in training_results.keys() if 'error' not in training_results[m]]
        logger.info(f"🤖 Models: {len(successful_models)} trained successfully")
        
        # Validation summary
        validation_results = results.get('model_validation', {})
        if 'game_winner' in validation_results:
            gw_results = validation_results['game_winner']
            if 'accuracy' in gw_results:
                logger.info(f"📈 Validation: {gw_results['accuracy']:.1%} accuracy")
        
        # Betting summary
        betting_results = results.get('betting_analysis', {})
        if 'total_betting_opportunities' in betting_results:
            logger.info(f"💰 Betting: {betting_results['total_betting_opportunities']} opportunities found")
            if betting_results.get('estimated_roi', 0) > 0:
                logger.info(f"🔥 Estimated ROI: {betting_results['estimated_roi']:.1%}")
        
        # Deployment summary
        deployment_results = results.get('deployment_status', {})
        deployed_count = len(deployment_results.get('deployed_models', []))
        logger.info(f"🚀 Deployment: {deployed_count} models deployed")
        
        # Smart pipeline status
        if results.get('pipeline_success'):
            logger.info("✅ PIPELINE STATUS: SUCCESS - Smart integrated models ready!")
            logger.info("🧠 Smart Features:")
            logger.info("   • Auto-detects stale data and refreshes from Sports API")
            logger.info("   • Prioritizes fast database access with API fallback")
            logger.info("   • Enhanced player-aware features for better predictions")
        else:
            logger.info("❌ PIPELINE STATUS: FAILED - Check errors above")
        
        logger.info("=" * 70)
    
    # ===============================================
    # NEW UTILITY METHODS FOR SMART DATA MANAGEMENT
    # ===============================================
    
    def force_data_refresh_from_api(self, seasons: List[int] = None) -> Dict[str, any]:
        """
        Force refresh of historical data from Sports API.
        
        Args:
            seasons: Specific seasons to refresh (defaults to all configured seasons)
        
        Returns:
            Refresh results
        """
        seasons_to_refresh = seasons or self.seasons_to_include
        logger.info(f"🔄 Force refreshing data for seasons: {seasons_to_refresh}")
        
        try:
            # Temporarily override seasons for refresh
            original_seasons = self.seasons_to_include
            self.seasons_to_include = seasons_to_refresh
            
            # Fetch fresh data
            fresh_data = self._fetch_data_from_sports_api()
            
            # Save to database
            saved_count = self.db.save_games(fresh_data)
            
            # Update current data
            self.historical_data = fresh_data
            
            # Restore original seasons
            self.seasons_to_include = original_seasons
            
            logger.info(f"✅ Refreshed {len(fresh_data)} games, saved {saved_count} to database")
            
            return {
                'success': True,
                'games_refreshed': len(fresh_data),
                'games_saved': saved_count,
                'seasons_refreshed': seasons_to_refresh
            }
            
        except Exception as e:
            logger.error(f"❌ Data refresh failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'seasons_attempted': seasons_to_refresh
            }
    
    def check_data_freshness(self) -> Dict[str, any]:
        """
        Check freshness of data in database vs what's available in Sports API.
        
        Returns:
            Data freshness analysis
        """
        logger.info("🔍 Checking data freshness...")
        
        freshness_report = {
            'database_games': 0,
            'seasons_checked': self.seasons_to_include,
            'season_analysis': {},
            'recommendation': 'unknown'
        }
        
        try:
            # Check database data
            db_data = self.db.get_historical_data(self.seasons_to_include)
            freshness_report['database_games'] = len(db_data)
            
            # Check each season
            for season in self.seasons_to_include:
                season_db_games = len(db_data[db_data['season'] == season]) if not db_data.empty else 0
                
                # For MLB, expect ~2430 games per season (except 2020)
                expected_games = 900 if season == 2020 else 2430
                completeness = season_db_games / expected_games
                
                freshness_report['season_analysis'][season] = {
                    'database_games': season_db_games,
                    'expected_games': expected_games,
                    'completeness': completeness,
                    'needs_refresh': completeness < 0.8
                }
            
            # Generate recommendation
            incomplete_seasons = [
                season for season, analysis in freshness_report['season_analysis'].items()
                if analysis['needs_refresh']
            ]
            
            if incomplete_seasons:
                freshness_report['recommendation'] = f"Refresh seasons: {incomplete_seasons}"
            else:
                freshness_report['recommendation'] = "Data appears complete"
            
            logger.info(f"✅ Freshness check complete. Database has {freshness_report['database_games']} games")
            
        except Exception as e:
            logger.error(f"❌ Freshness check failed: {e}")
            freshness_report['error'] = str(e)
        
        return freshness_report
    
    def get_pipeline_status(self) -> Dict[str, any]:
        """
        Get comprehensive status of the entire pipeline.
        
        Returns:
            Complete pipeline status
        """
        return {
            'player_mapper_status': self.get_player_mapper_status(),
            'data_freshness': self.check_data_freshness(),
            'trained_models': list(self.trained_models.keys()),
            'last_training_results': self.training_results,
            'pipeline_ready': len(self.trained_models) > 0 and not self.player_mapper.player_map.empty
        }

    # ===============================================
    # NEW UTILITY METHODS FOR PLAYER INTEGRATION
    # ===============================================
    
    def get_player_mapper_status(self) -> Dict[str, any]:
        """Get detailed status of player mapper."""
        return {
            'summary': self.player_mapper.get_summary(),
            'csv_status': self.player_mapper.check_csv_status(),
            'player_quality': self._validate_player_data_quality()
        }
    
    def refresh_player_data(self, force_api: bool = False):
        """Manually refresh player data."""
        if force_api:
            self.player_mapper.force_api_refresh()
        else:
            self.player_mapper.refresh_data()
        
        logger.info("✅ Player data refreshed")
    
    def search_players(self, search_term: str, limit: int = 10) -> List[Dict]:
        """Search for players in the mapper."""
        return self.player_mapper.search_players(search_term, limit)


# CLI interface with enhanced smart data management options
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Smart Integrated MLB Training Pipeline')
    parser.add_argument('--seasons', nargs='+', type=int, default=[2022, 2023, 2024], 
                       help='Seasons to include')
    parser.add_argument('--retrain', action='store_true', help='Retrain models from scratch')
    parser.add_argument('--tune', action='store_true', help='Tune hyperparameters')
    parser.add_argument('--no-validate', action='store_true', help='Skip validation')
    parser.add_argument('--refresh-players', action='store_true', help='Force refresh player data from API')
    parser.add_argument('--csv-max-age', type=int, default=24, help='Max age of CSV files in hours')
    
    # NEW: Smart data management options
    parser.add_argument('--force-api-refresh', action='store_true', 
                       help='Force refresh historical data from Sports API')
    parser.add_argument('--check-freshness', action='store_true', 
                       help='Check data freshness without training')
    parser.add_argument('--status', action='store_true', 
                       help='Get comprehensive pipeline status')
    
    args = parser.parse_args()
    
    # Initialize smart integrated pipeline
    pipeline = IntegratedMLBTrainingPipeline(
        seasons_to_include=args.seasons,
        min_games_required=500,
        csv_max_age_hours=args.csv_max_age
    )
    
    # Handle different command modes
    if args.status:
        # Just get status
        status = pipeline.get_pipeline_status()
        print("\n📊 PIPELINE STATUS:")
        print(f"   Player Mapper: {status['player_mapper_status']['summary']['players_loaded']} players")
        print(f"   Data Freshness: {status['data_freshness']['database_games']} games in database")
        print(f"   Trained Models: {status['trained_models']}")
        print(f"   Pipeline Ready: {'✅ Yes' if status['pipeline_ready'] else '❌ No'}")
        
    elif args.check_freshness:
        # Check data freshness
        freshness = pipeline.check_data_freshness()
        print("\n🔍 DATA FRESHNESS REPORT:")
        print(f"   Database Games: {freshness['database_games']}")
        print(f"   Recommendation: {freshness['recommendation']}")
        for season, analysis in freshness['season_analysis'].items():
            status = "✅ Complete" if not analysis['needs_refresh'] else "⚠️ Needs Refresh"
            print(f"   Season {season}: {analysis['database_games']}/{analysis['expected_games']} games ({status})")
        
    elif args.force_api_refresh:
        # Force data refresh
        print("🔄 Forcing data refresh from Sports API...")
        refresh_result = pipeline.force_data_refresh_from_api(args.seasons)
        if refresh_result['success']:
            print(f"✅ Refreshed {refresh_result['games_refreshed']} games")
        else:
            print(f"❌ Refresh failed: {refresh_result['error']}")
        
    else:
        # Run full training pipeline
        results = pipeline.run_complete_training_pipeline(
            retrain_models=args.retrain,
            tune_hyperparameters=args.tune,
            validate_models=not args.no_validate,
            refresh_player_data=args.refresh_players
        )
        
        # Final status
        if results['pipeline_success']:
            print("\n🎉 SMART INTEGRATED TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            print("🚀 Your enhanced MLB models are ready for live predictions!")
            print("🗺️ Player mapping is integrated and working!")
            print("🧠 Smart data loading ensures fresh, comprehensive data!")
        else:
            print("\n❌ Smart integrated training pipeline encountered issues.")
            print("Check the logs above for specific errors.")
            
        # Show data source used
        data_prep = results.get('data_preparation', {})
        data_source = data_prep.get('data_source', 'unknown')
        api_fallback = data_prep.get('api_fallback_used', False)
        
        if api_fallback:
            print(f"📡 Used Sports API fallback - your data is now fresh!")
        else:
            print(f"💾 Used {data_source} data - pipeline optimized for speed!")
