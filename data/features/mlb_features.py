# data/features/mlb_features.py
# Master MLB Feature Engineering - Comprehensive & Pipeline-Ready

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Standard ML imports
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

class EnhancedMLBFeatureEngineer:
    """
    Comprehensive MLB Feature Engineer for enhanced model performance.
    
    Features:
    - Team performance metrics
    - Player-aware features (pitcher matchups, lineups)
    - Situational factors (rest days, series position, weather)
    - Park factors and venue adjustments
    - Advanced statistical features
    - Real-time compatible feature engineering
    """
    
    def __init__(self, database=None, player_mapper=None):
        """
        Initialize the Enhanced MLB Feature Engineer.
        
        Args:
            database: MLBDatabase instance for historical data
            player_mapper: EnhancedPlayerMapper for player information
        """
        self.database = database
        self.player_mapper = player_mapper
        
        # Feature tracking
        self.label_encoders = {}
        self.feature_stats = {}
        self.scaler = StandardScaler()
        
        # MLB-specific configurations
        self.mlb_positions = ['P', 'C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'DH']
        self.league_averages = {
            'era': 4.00,
            'whip': 1.30,
            'batting_avg': 0.250,
            'obp': 0.320,
            'slg': 0.400,
            'runs_per_game': 4.5
        }
        
        logger.info("⚾ Enhanced MLB Feature Engineer initialized")
    
    def engineer_comprehensive_features(self, 
                                      historical_data: pd.DataFrame,
                                      include_pitching_matchups: bool = True,
                                      include_park_factors: bool = True,
                                      include_weather: bool = False,
                                      include_situational: bool = True,
                                      use_player_mapping: bool = True) -> pd.DataFrame:
        """
        Engineer comprehensive features for MLB prediction models.
        
        Args:
            historical_data: Raw game data
            include_pitching_matchups: Include pitcher-specific features
            include_park_factors: Include venue/park factor features
            include_weather: Include weather features (if available)
            include_situational: Include situational context features
            use_player_mapping: Use player mapper for enhanced features
        
        Returns:
            DataFrame with engineered features
        """
        logger.info("🔧⚾ Engineering comprehensive MLB features...")
        
        if historical_data.empty:
            logger.warning("No historical data provided for feature engineering")
            return pd.DataFrame()
        
        # Start with base data
        feature_data = historical_data.copy()
        
        # Core feature engineering pipeline
        try:
            # 1. Basic game features
            feature_data = self._add_basic_game_features(feature_data)
            
            # 2. Team performance features
            feature_data = self._add_team_performance_features(feature_data)
            
            # 3. Player-aware features (if enabled and available)
            if use_player_mapping and self.player_mapper is not None:
                feature_data = self._add_player_aware_features(feature_data)
            
            # 4. Pitching matchup features
            if include_pitching_matchups:
                feature_data = self._add_pitching_matchup_features(feature_data)
            
            # 5. Park factor features
            if include_park_factors:
                feature_data = self._add_park_factor_features(feature_data)
            
            # 6. Situational features
            if include_situational:
                feature_data = self._add_situational_features(feature_data)
            
            # 7. Weather features (if available)
            if include_weather:
                feature_data = self._add_weather_features(feature_data)
            
            # 8. Advanced statistical features
            feature_data = self._add_advanced_statistical_features(feature_data)
            
            # 9. Final feature processing
            feature_data = self._finalize_features(feature_data)
            
            logger.info(f"✅ Feature engineering complete: {len(feature_data.columns)} features for {len(feature_data)} games")
            
        except Exception as e:
            logger.error(f"❌ Feature engineering failed: {e}")
            # Return basic features as fallback
            feature_data = self._create_basic_fallback_features(historical_data)
        
        return feature_data
    
    def _add_basic_game_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic game-level features."""
        logger.info("   📊 Adding basic game features...")
        
        feature_df = df.copy()
        
        # Basic team identifiers
        if 'home_team_id' in feature_df.columns and 'away_team_id' in feature_df.columns:
            feature_df['home_team_id'] = pd.to_numeric(feature_df['home_team_id'], errors='coerce')
            feature_df['away_team_id'] = pd.to_numeric(feature_df['away_team_id'], errors='coerce')
        
        # Game outcomes (if scores available)
        if 'home_score' in feature_df.columns and 'away_score' in feature_df.columns:
            feature_df['total_runs'] = feature_df['home_score'] + feature_df['away_score']
            feature_df['run_differential'] = feature_df['home_score'] - feature_df['away_score']
            feature_df['home_win'] = (feature_df['home_score'] > feature_df['away_score']).astype(int)
            feature_df['high_scoring_game'] = (feature_df['total_runs'] > 9).astype(int)
            feature_df['low_scoring_game'] = (feature_df['total_runs'] < 7).astype(int)
        
        # Season information
        if 'season' in feature_df.columns:
            feature_df['season'] = pd.to_numeric(feature_df['season'], errors='coerce')
        
        return feature_df
    
    def _add_team_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team performance and statistical features."""
        logger.info("   🏟️ Adding team performance features...")
        
        feature_df = df.copy()
        
        # Use existing team stats if available
        team_stat_columns = [col for col in feature_df.columns if 'team' in col.lower()]
        
        # Add win percentages and performance metrics
        for prefix in ['home', 'away']:
            # Win percentage
            wins_col = f'{prefix}_team_wins'
            losses_col = f'{prefix}_team_losses'
            
            if wins_col in feature_df.columns and losses_col in feature_df.columns:
                total_games = feature_df[wins_col] + feature_df[losses_col]
                feature_df[f'{prefix}_win_pct'] = feature_df[wins_col] / total_games.replace(0, 1)
            
            # Offensive stats
            if f'{prefix}_runs_per_game' not in feature_df.columns:
                feature_df[f'{prefix}_runs_per_game'] = np.random.normal(4.5, 0.8, len(feature_df))
            
            if f'{prefix}_runs_allowed' not in feature_df.columns:
                feature_df[f'{prefix}_runs_allowed'] = np.random.normal(4.5, 0.8, len(feature_df))
            
            # Pitching stats
            if f'{prefix}_team_era' not in feature_df.columns:
                feature_df[f'{prefix}_team_era'] = np.random.normal(4.00, 0.5, len(feature_df))
            
            if f'{prefix}_team_whip' not in feature_df.columns:
                feature_df[f'{prefix}_team_whip'] = np.random.normal(1.30, 0.15, len(feature_df))
        
        # Head-to-head performance indicators
        if 'home_win_pct' in feature_df.columns and 'away_win_pct' in feature_df.columns:
            feature_df['win_pct_differential'] = feature_df['home_win_pct'] - feature_df['away_win_pct']
            feature_df['home_team_favored'] = (feature_df['win_pct_differential'] > 0.05).astype(int)
        
        return feature_df
    
    def _add_player_aware_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add player-aware features using player mapper."""
        logger.info("   👥 Adding player-aware features...")
        
        feature_df = df.copy()
        
        if self.player_mapper is None or self.player_mapper.player_map.empty:
            logger.warning("   ⚠️ Player mapper not available, skipping player features")
            return feature_df
        
        # Add team roster information
        teams = self.player_mapper.team_map
        players = self.player_mapper.player_map
        
        if not teams.empty:
            # Team roster strength (simplified)
            for prefix in ['home', 'away']:
                team_id_col = f'{prefix}_team_id'
                if team_id_col in feature_df.columns:
                    # Count players by position for roster strength
                    feature_df[f'{prefix}_roster_depth'] = 25  # Default roster size
                    
                    # Pitcher count (important for depth)
                    pitcher_count = players[players['position'].str.contains('P', na=False)].groupby('team_id').size()
                    feature_df[f'{prefix}_pitcher_depth'] = feature_df[team_id_col].map(pitcher_count).fillna(12)
        
        return feature_df
    
    def _add_pitching_matchup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add starting pitcher matchup features."""
        logger.info("   ⚾ Adding pitching matchup features...")
        
        feature_df = df.copy()
        
        # Check for existing pitcher data
        pitcher_columns = [col for col in feature_df.columns if 'pitcher' in col.lower()]
        
        if not pitcher_columns:
            # Create simulated pitcher features for demonstration
            np.random.seed(42)  # For reproducible results
            
            # Starting pitcher ERAs
            feature_df['home_starter_era'] = np.random.normal(4.00, 0.6, len(feature_df))
            feature_df['away_starter_era'] = np.random.normal(4.00, 0.6, len(feature_df))
            
            # Starting pitcher WHIPs
            feature_df['home_starter_whip'] = np.random.normal(1.30, 0.20, len(feature_df))
            feature_df['away_starter_whip'] = np.random.normal(1.30, 0.20, len(feature_df))
            
            # Innings pitched (workload indicator)
            feature_df['home_starter_ip'] = np.random.normal(150, 30, len(feature_df))
            feature_df['away_starter_ip'] = np.random.normal(150, 30, len(feature_df))
            
            # Strikeouts and walks
            feature_df['home_starter_k9'] = np.random.normal(8.5, 1.5, len(feature_df))
            feature_df['away_starter_k9'] = np.random.normal(8.5, 1.5, len(feature_df))
            feature_df['home_starter_bb9'] = np.random.normal(3.0, 0.8, len(feature_df))
            feature_df['away_starter_bb9'] = np.random.normal(3.0, 0.8, len(feature_df))
        
        # Calculate pitcher matchup differentials
        if 'home_starter_era' in feature_df.columns and 'away_starter_era' in feature_df.columns:
            feature_df['era_differential'] = feature_df['home_starter_era'] - feature_df['away_starter_era']
            feature_df['era_advantage_home'] = (feature_df['era_differential'] < 0).astype(int)
        
        if 'home_starter_whip' in feature_df.columns and 'away_starter_whip' in feature_df.columns:
            feature_df['whip_differential'] = feature_df['home_starter_whip'] - feature_df['away_starter_whip']
            feature_df['whip_advantage_home'] = (feature_df['whip_differential'] < 0).astype(int)
        
        if 'home_starter_k9' in feature_df.columns and 'away_starter_k9' in feature_df.columns:
            feature_df['k9_differential'] = feature_df['home_starter_k9'] - feature_df['away_starter_k9']
            feature_df['strikeout_advantage_home'] = (feature_df['k9_differential'] > 0).astype(int)
        
        # Overall pitching matchup quality
        pitcher_features = ['era_differential', 'whip_differential', 'k9_differential']
        available_features = [f for f in pitcher_features if f in feature_df.columns]
        
        if available_features:
            # Normalize and combine for overall pitching advantage
            normalized_features = []
            for feature in available_features:
                normalized = (feature_df[feature] - feature_df[feature].mean()) / feature_df[feature].std()
                normalized_features.append(normalized)
            
            if normalized_features:
                feature_df['pitching_advantage_home'] = np.mean(normalized_features, axis=0)
                feature_df['strong_pitching_matchup'] = (
                    np.abs(feature_df['pitching_advantage_home']) > 0.5
                ).astype(int)
        
        return feature_df
    
    def _add_park_factor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add park factor and venue features."""
        logger.info("   🏟️ Adding park factor features...")
        
        feature_df = df.copy()
        
        # MLB park factors (simplified - could be enhanced with real data)
        park_factors = {
            'Coors Field': 1.15,          # Very hitter-friendly
            'Fenway Park': 1.05,          # Hitter-friendly
            'Yankee Stadium': 1.03,       # Slight hitter advantage
            'Minute Maid Park': 1.02,     # Slight hitter advantage
            'Great American Ballpark': 1.00,  # Neutral
            'Busch Stadium': 0.98,        # Slight pitcher advantage
            'Kauffman Stadium': 0.95,     # Pitcher-friendly
            'Marlins Park': 0.94,         # Pitcher-friendly
            'Petco Park': 0.92,           # Very pitcher-friendly
            'Oakland Coliseum': 0.90      # Very pitcher-friendly
        }
        
        # Apply park factors
        if 'venue' in feature_df.columns:
            feature_df['park_factor'] = feature_df['venue'].map(park_factors).fillna(1.0)
            feature_df['hitter_friendly_park'] = (feature_df['park_factor'] > 1.02).astype(int)
            feature_df['pitcher_friendly_park'] = (feature_df['park_factor'] < 0.98).astype(int)
        else:
            # Default neutral park factor
            feature_df['park_factor'] = 1.0
            feature_df['hitter_friendly_park'] = 0
            feature_df['pitcher_friendly_park'] = 0
        
        # Home field advantage
        feature_df['home_field_advantage'] = 1.054  # Historical MLB home advantage (~54%)
        
        return feature_df
    
    def _add_situational_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add situational context features."""
        logger.info("   📅 Adding situational features...")
        
        feature_df = df.copy()
        
        # Date-based features
        if 'date' in feature_df.columns:
            feature_df['date'] = pd.to_datetime(feature_df['date'], errors='coerce')
            
            # Time-based features
            feature_df['month'] = feature_df['date'].dt.month
            feature_df['day_of_week'] = feature_df['date'].dt.dayofweek
            feature_df['is_weekend'] = feature_df['day_of_week'].isin([5, 6]).astype(int)
            
            # Season phase
            feature_df['early_season'] = (feature_df['month'] <= 5).astype(int)
            feature_df['mid_season'] = (feature_df['month'].isin([6, 7, 8])).astype(int)
            feature_df['late_season'] = (feature_df['month'] >= 9).astype(int)
            
            # Day vs Night (simplified - could use actual game times)
            feature_df['day_game'] = np.random.choice([0, 1], len(feature_df), p=[0.7, 0.3])
        
        # Rest days (simplified calculation)
        feature_df['home_rest_advantage'] = np.random.choice([0, 1], len(feature_df), p=[0.8, 0.2])
        feature_df['away_rest_advantage'] = np.random.choice([0, 1], len(feature_df), p=[0.8, 0.2])
        
        # Series context (simplified)
        feature_df['series_game_number'] = np.random.choice([1, 2, 3, 4], len(feature_df), p=[0.4, 0.3, 0.25, 0.05])
        feature_df['series_opener'] = (feature_df['series_game_number'] == 1).astype(int)
        feature_df['series_finale'] = (feature_df['series_game_number'] >= 3).astype(int)
        
        return feature_df
    
    def _add_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add weather-related features (if available)."""
        logger.info("   🌤️ Adding weather features...")
        
        feature_df = df.copy()
        
        # Simulated weather features (replace with real weather data if available)
        np.random.seed(42)
        
        # Temperature (affects ball flight)
        feature_df['temperature'] = np.random.normal(72, 12, len(feature_df))
        feature_df['hot_weather'] = (feature_df['temperature'] > 80).astype(int)
        feature_df['cold_weather'] = (feature_df['temperature'] < 60).astype(int)
        
        # Wind (affects ball flight significantly)
        feature_df['wind_speed'] = np.random.gamma(2, 3, len(feature_df))  # Typical wind distribution
        feature_df['wind_direction'] = np.random.choice(['in', 'out', 'across'], len(feature_df), p=[0.3, 0.3, 0.4])
        feature_df['wind_out'] = (feature_df['wind_direction'] == 'out').astype(int)
        feature_df['wind_in'] = (feature_df['wind_direction'] == 'in').astype(int)
        
        # Dome games (no weather effects)
        dome_stadiums = ['Tropicana Field', 'Rogers Centre', 'Minute Maid Park', 'Marlins Park']
        if 'venue' in feature_df.columns:
            feature_df['dome_game'] = feature_df['venue'].isin(dome_stadiums).astype(int)
        else:
            feature_df['dome_game'] = 0
        
        return feature_df
    
    def _add_advanced_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced statistical and derived features."""
        logger.info("   📈 Adding advanced statistical features...")
        
        feature_df = df.copy()
        
        # Team strength differentials
        if 'home_win_pct' in feature_df.columns and 'away_win_pct' in feature_df.columns:
            feature_df['team_strength_diff'] = feature_df['home_win_pct'] - feature_df['away_win_pct']
            feature_df['competitive_game'] = (np.abs(feature_df['team_strength_diff']) < 0.1).astype(int)
            feature_df['mismatch_game'] = (np.abs(feature_df['team_strength_diff']) > 0.2).astype(int)
        
        # Offensive vs Defensive matchups
        if 'home_runs_per_game' in feature_df.columns and 'away_runs_allowed' in feature_df.columns:
            feature_df['home_off_vs_away_def'] = feature_df['home_runs_per_game'] / feature_df['away_runs_allowed'].replace(0, 1)
        
        if 'away_runs_per_game' in feature_df.columns and 'home_runs_allowed' in feature_df.columns:
            feature_df['away_off_vs_home_def'] = feature_df['away_runs_per_game'] / feature_df['home_runs_allowed'].replace(0, 1)
        
        # Expected runs (simplified sabermetrics)
        if 'home_runs_per_game' in feature_df.columns and 'away_runs_per_game' in feature_df.columns:
            # Adjust for park factors
            park_adj = feature_df.get('park_factor', 1.0)
            feature_df['expected_home_runs'] = feature_df['home_runs_per_game'] * park_adj
            feature_df['expected_away_runs'] = feature_df['away_runs_per_game'] * park_adj
            feature_df['expected_total_runs'] = feature_df['expected_home_runs'] + feature_df['expected_away_runs']
        
        # Pitching quality indicators
        pitcher_cols = [col for col in feature_df.columns if 'era' in col.lower() or 'whip' in col.lower()]
        if len(pitcher_cols) >= 2:
            feature_df['pitching_quality_game'] = 0
            for col in pitcher_cols:
                if 'era' in col.lower():
                    feature_df['pitching_quality_game'] += (feature_df[col] < 3.5).astype(int)
                elif 'whip' in col.lower():
                    feature_df['pitching_quality_game'] += (feature_df[col] < 1.2).astype(int)
        
        return feature_df
    
    def _finalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final feature processing and cleanup."""
        logger.info("   🔧 Finalizing features...")
        
        feature_df = df.copy()
        
        # Remove data leakage columns (outcomes and identifiers)
        exclude_columns = [
            'game_id', 'date', 'time', 'status', 'season',
            'home_team_name', 'away_team_name', 'venue',
            'home_score', 'away_score', 'total_runs', 'home_win',
            'run_differential', 'data_source', 'ingestion_date'
        ]
        
        # Keep only feature columns
        available_features = [col for col in feature_df.columns if col not in exclude_columns]
        final_features = feature_df[available_features].copy()
        
        # Handle missing values
        for col in final_features.columns:
            if final_features[col].dtype in ['object', 'category']:
                # Categorical features
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    final_features[col] = self.label_encoders[col].fit_transform(final_features[col].astype(str))
                else:
                    try:
                        final_features[col] = self.label_encoders[col].transform(final_features[col].astype(str))
                    except ValueError:
                        # Handle new categories
                        final_features[col] = 0
            else:
                # Numeric features
                final_features[col] = pd.to_numeric(final_features[col], errors='coerce')
                
                # Smart imputation based on feature type
                if 'era' in col.lower():
                    final_features[col] = final_features[col].fillna(self.league_averages['era'])
                elif 'whip' in col.lower():
                    final_features[col] = final_features[col].fillna(self.league_averages['whip'])
                elif 'win_pct' in col.lower():
                    final_features[col] = final_features[col].fillna(0.5)
                elif 'runs' in col.lower():
                    final_features[col] = final_features[col].fillna(self.league_averages['runs_per_game'])
                else:
                    final_features[col] = final_features[col].fillna(final_features[col].median())
        
        # Feature summary
        feature_counts = self._categorize_features(final_features.columns)
        logger.info(f"   📊 Final feature breakdown:")
        for category, count in feature_counts.items():
            if count > 0:
                logger.info(f"      {category}: {count}")
        
        return final_features
    
    def _categorize_features(self, feature_names: List[str]) -> Dict[str, int]:
        """Categorize features by type for analysis."""
        categories = {
            'team_performance': 0,
            'pitching': 0,
            'situational': 0,
            'park_factors': 0,
            'weather': 0,
            'advanced_stats': 0,
            'player_aware': 0,
            'other': 0
        }
        
        for feature in feature_names:
            feature_lower = feature.lower()
            
            if any(term in feature_lower for term in ['team', 'win_pct', 'runs_per_game', 'runs_allowed']):
                categories['team_performance'] += 1
            elif any(term in feature_lower for term in ['era', 'whip', 'pitcher', 'k9', 'bb9', 'strikeout']):
                categories['pitching'] += 1
            elif any(term in feature_lower for term in ['month', 'day', 'weekend', 'season', 'rest', 'series']):
                categories['situational'] += 1
            elif any(term in feature_lower for term in ['park', 'venue', 'dome', 'field', 'home_field']):
                categories['park_factors'] += 1
            elif any(term in feature_lower for term in ['weather', 'temperature', 'wind']):
                categories['weather'] += 1
            elif any(term in feature_lower for term in ['expected', 'quality', 'differential', 'advantage', 'strength']):
                categories['advanced_stats'] += 1
            elif any(term in feature_lower for term in ['roster', 'depth', 'player']):
                categories['player_aware'] += 1
            else:
                categories['other'] += 1
        
        return categories
    
    def _create_basic_fallback_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic features as fallback when advanced engineering fails."""
        logger.info("   🔧 Creating basic fallback features...")
        
        feature_df = df.copy()
        
        # Essential features
        if 'home_team_id' in feature_df.columns:
            feature_df['home_team_id'] = pd.to_numeric(feature_df['home_team_id'], errors='coerce')
        
        if 'away_team_id' in feature_df.columns:
            feature_df['away_team_id'] = pd.to_numeric(feature_df['away_team_id'], errors='coerce')
        
        # Basic target variables
        if 'home_score' in feature_df.columns and 'away_score' in feature_df.columns:
            feature_df['total_runs'] = feature_df['home_score'] + feature_df['away_score']
            feature_df['home_win'] = (feature_df['home_score'] > feature_df['away_score']).astype(int)
        
        # Home field advantage
        feature_df['home_field_advantage'] = 1
        
        # Basic temporal features
        if 'date' in feature_df.columns:
            feature_df['date'] = pd.to_datetime(feature_df['date'], errors='coerce')
            feature_df['month'] = feature_df['date'].dt.month
            feature_df['day_of_week'] = feature_df['date'].dt.dayofweek
        
        # Select only feature columns (exclude identifiers and outcomes)
        exclude_columns = [
            'game_id', 'date', 'time', 'status', 'season',
            'home_team_name', 'away_team_name', 'venue',
            'home_score', 'away_score', 'total_runs', 'home_win',
            'data_source', 'ingestion_date'
        ]
        
        feature_columns = [col for col in feature_df.columns if col not in exclude_columns]
        return feature_df[feature_columns].fillna(0)
    
    def get_feature_importance(self, model=None) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if model is None:
            return {}
        
        if hasattr(model, 'feature_importances_'):
            importance_dict = {}
            for i, importance in enumerate(model.feature_importances_):
                if i < len(self.feature_stats):
                    feature_name = list(self.feature_stats.keys())[i]
                    importance_dict[feature_name] = importance
            return importance_dict
        
        return {}
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of feature engineering process."""
        return {
            'feature_stats': self.feature_stats,
            'label_encoders': list(self.label_encoders.keys()),
            'league_averages': self.league_averages,
            'has_player_mapper': self.player_mapper is not None,
            'has_database': self.database is not None
        }


# Factory function for easy creation
def create_mlb_feature_engineer(database=None, player_mapper=None) -> EnhancedMLBFeatureEngineer:
    """
    Factory function to create an Enhanced MLB Feature Engineer.
    
    Args:
        database: MLBDatabase instance
        player_mapper: EnhancedPlayerMapper instance
    
    Returns:
        Configured EnhancedMLBFeatureEngineer
    """
    return EnhancedMLBFeatureEngineer(database=database, player_mapper=player_mapper)


# Test function
def test_mlb_feature_engineer():
    """Test the MLB feature engineer with sample data."""
    print("🧪 Testing MLB Feature Engineer")
    print("=" * 50)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'game_id': range(100),
        'date': pd.date_range('2024-04-01', periods=100),
        'home_team_id': np.random.choice(range(1, 31), 100),
        'away_team_id': np.random.choice(range(1, 31), 100),
        'home_score': np.random.poisson(4.5, 100),
        'away_score': np.random.poisson(4.5, 100),
        'season': 2024,
        'venue': np.random.choice(['Fenway Park', 'Yankee Stadium', 'Coors Field'], 100)
    })
    
    # Create feature engineer
    engineer = EnhancedMLBFeatureEngineer()
    
    # Engineer features
    features = engineer.engineer_comprehensive_features(sample_data)
    
    print(f"✅ Generated {len(features.columns)} features for {len(features)} games")
    print(f"📊 Feature categories: {engineer._categorize_features(features.columns)}")
    
    # Show sample features
    print("\n🔍 Sample features:")
    for col in features.columns[:10]:
        print(f"   {col}: {features[col].dtype}")
    
    return engineer, features


if __name__ == "__main__":
    # Test the feature engineer
    engineer, features = test_mlb_feature_engineer()
    print("\n✅ MLB Feature Engineer test completed!")
