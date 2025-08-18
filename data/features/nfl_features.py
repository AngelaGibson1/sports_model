# data/features/nfl_features.py

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger
import warnings

from config.settings import Settings
from data.database.nfl import NFLDatabase
from utils.data_helpers import (
    calculate_rolling_averages,
    calculate_rolling_statistics,
    calculate_team_form,
    calculate_head_to_head_stats,
    create_feature_interactions,
    create_lag_features,
    handle_missing_values
)

warnings.filterwarnings('ignore')

class NFLFeatureEngineer:
    """
    Comprehensive NFL feature engineering for game prediction models.
    Creates advanced features from raw NFL data for machine learning models.
    """
    
    def __init__(self, db: Optional[NFLDatabase] = None):
        """
        Initialize NFL feature engineer.
        
        Args:
            db: Optional NFL database instance
        """
        self.db = db or NFLDatabase()
        self.nfl_config = Settings.SPORT_CONFIGS['nfl']
        self.rolling_windows = Settings.ROLLING_WINDOWS['nfl']
        
        # NFL-specific feature configuration
        self.key_stats = [
            'points_per_game', 'points_allowed_per_game', 'yards_per_game', 'yards_allowed_per_game',
            'passing_yards_per_game', 'rushing_yards_per_game', 'turnovers_per_game', 
            'turnover_differential', 'sacks_per_game', 'third_down_conversion_pct'
        ]
        
        self.logger = logger
        
        logger.info("🏈 NFL Feature Engineer initialized")
    
    def engineer_game_features(self, 
                              seasons: Optional[List[int]] = None,
                              include_advanced: bool = True,
                              include_situational: bool = True,
                              include_weather: bool = True) -> pd.DataFrame:
        """
        Create comprehensive features for NFL game prediction.
        
        Args:
            seasons: Seasons to include in feature engineering
            include_advanced: Whether to include advanced metrics
            include_situational: Whether to include situational features
            include_weather: Whether to include weather features
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("🔧 Starting NFL feature engineering...")
        
        # Get base historical data
        historical_df = self.db.get_historical_data(seasons)
        
        if historical_df.empty:
            logger.warning("No historical data available for feature engineering")
            return pd.DataFrame()
        
        # Sort by date for proper time series processing
        historical_df = historical_df.sort_values(['season', 'week', 'date'])
        
        # Create base features
        features_df = self._create_base_features(historical_df)
        
        # Add rolling performance features
        features_df = self._add_rolling_performance_features(features_df)
        
        # Add team form and momentum features
        features_df = self._add_team_form_features(features_df)
        
        # Add head-to-head features
        features_df = self._add_head_to_head_features(features_df)
        
        if include_situational:
            # Add situational features (rest, divisional games, etc.)
            features_df = self._add_situational_features(features_df)
        
        if include_weather:
            # Add weather impact features
            features_df = self._add_weather_features(features_df)
        
        if include_advanced:
            # Add advanced metrics
            features_df = self._add_advanced_metrics(features_df)
        
        # Add feature interactions
        features_df = self._add_feature_interactions(features_df)
        
        # Handle missing values
        features_df = handle_missing_values(features_df, strategy='smart')
        
        # Final cleanup
        features_df = self._cleanup_features(features_df)
        
        logger.info(f"✅ NFL feature engineering complete: {features_df.shape[0]} games, {features_df.shape[1]} features")
        
        return features_df
    
    def _create_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create base features from raw game data."""
        logger.info("📊 Creating base NFL features...")
        
        features_df = df.copy()
        
        # Basic game features
        features_df['total_points'] = features_df['home_score'] + features_df['away_score']
        features_df['point_differential'] = features_df['home_score'] - features_df['away_score']
        features_df['home_win'] = (features_df['home_score'] > features_df['away_score']).astype(int)
        
        # Date and time features
        features_df['date'] = pd.to_datetime(features_df['date'])
        features_df['month'] = features_df['date'].dt.month
        features_df['day_of_week'] = features_df['date'].dt.dayofweek
        features_df['is_sunday'] = (features_df['day_of_week'] == 6).astype(int)
        features_df['is_monday'] = (features_df['day_of_week'] == 0).astype(int)
        features_df['is_thursday'] = (features_df['day_of_week'] == 3).astype(int)
        features_df['is_primetime'] = (features_df['is_monday'] | features_df['is_thursday']).astype(int)
        
        # Season features
        features_df['season_progress'] = features_df['week'] / 17.0  # NFL regular season is 17 weeks
        features_df['early_season'] = (features_df['week'] <= 4).astype(int)
        features_df['late_season'] = (features_df['week'] >= 14).astype(int)
        
        # Game context features (already in data)
        if 'divisional_game' not in features_df.columns:
            features_df['divisional_game'] = 0
        if 'conference_game' not in features_df.columns:
            features_df['conference_game'] = 0
        
        # Team strength differentials (if stats available)
        stat_columns = [col for col in self.key_stats if f'home_{col}' in features_df.columns]
        
        for stat in stat_columns:
            home_col = f'home_{stat}'
            away_col = f'away_{stat}'
            
            if home_col in features_df.columns and away_col in features_df.columns:
                features_df[f'{stat}_diff'] = features_df[home_col] - features_df[away_col]
                if 'percentage' not in stat:  # Don't create ratios for percentages
                    features_df[f'{stat}_ratio'] = features_df[home_col] / (features_df[away_col] + 0.001)
        
        return features_df
    
    def _add_rolling_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling performance features for both teams."""
        logger.info("📈 Adding rolling NFL performance features...")
        
        features_df = df.copy()
        
        # Create team performance tracking
        home_performance = self._create_team_performance_tracking(features_df, 'home')
        away_performance = self._create_team_performance_tracking(features_df, 'away')
        
        # Add rolling statistics for multiple windows (smaller windows for NFL)
        for window_name, window_size in self.rolling_windows.items():
            # Home team rolling stats
            home_rolling = self._calculate_team_rolling_stats(
                home_performance, window_size, f'home_{window_name}'
            )
            
            # Away team rolling stats  
            away_rolling = self._calculate_team_rolling_stats(
                away_performance, window_size, f'away_{window_name}'
            )
            
            # Merge rolling stats back to features
            features_df = self._merge_rolling_stats(features_df, home_rolling, away_rolling, window_name)
        
        return features_df
    
    def _add_team_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team form and momentum features."""
        logger.info("🔥 Adding NFL team form features...")
        
        features_df = df.copy()
        
        # Create form tracking for home and away teams
        for team_type in ['home', 'away']:
            team_id_col = f'{team_type}_team_id'
            
            # Recent form (smaller windows for NFL)
            for window in [2, 4, 8]:
                form_col = f'{team_type}_form_{window}'
                features_df[form_col] = self._calculate_team_form_rolling(
                    features_df, team_id_col, window
                )
            
            # Win/loss streaks
            features_df[f'{team_type}_win_streak'] = self._calculate_win_streaks(
                features_df, team_id_col, streak_type='win'
            )
            features_df[f'{team_type}_loss_streak'] = self._calculate_win_streaks(
                features_df, team_id_col, streak_type='loss'
            )
            
            # Home/away specific form
            features_df[f'{team_type}_home_form_4'] = self._calculate_venue_form(
                features_df, team_id_col, venue='home', window=4
            )
            features_df[f'{team_type}_away_form_4'] = self._calculate_venue_form(
                features_df, team_id_col, venue='away', window=4
            )
        
        # Form differentials
        for window in [2, 4, 8]:
            features_df[f'form_diff_{window}'] = (
                features_df[f'home_form_{window}'] - features_df[f'away_form_{window}']
            )
        
        return features_df
    
    def _add_head_to_head_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add head-to-head historical features."""
        logger.info("⚔️ Adding NFL head-to-head features...")
        
        features_df = df.copy()
        
        # Calculate H2H statistics for each game
        h2h_features = []
        
        for idx, row in features_df.iterrows():
            home_team = row['home_team_id']
            away_team = row['away_team_id']
            game_date = row['date']
            season = row['season']
            
            # Get historical H2H data before this game
            h2h_stats = self._get_historical_h2h(
                home_team, away_team, game_date, season, features_df
            )
            
            h2h_features.append(h2h_stats)
        
        # Convert to DataFrame and merge
        h2h_df = pd.DataFrame(h2h_features)
        features_df = pd.concat([features_df.reset_index(drop=True), h2h_df], axis=1)
        
        return features_df
    
    def _add_situational_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add situational features specific to NFL."""
        logger.info("🎯 Adding NFL situational features...")
        
        features_df = df.copy()
        
        # Calculate rest days (more important in NFL)
        features_df['home_rest_days'] = self._calculate_rest_days(
            features_df, 'home_team_id'
        )
        features_df['away_rest_days'] = self._calculate_rest_days(
            features_df, 'away_team_id'
        )
        
        # Rest advantage
        features_df['rest_advantage'] = (
            features_df['home_rest_days'] - features_df['away_rest_days']
        )
        
        # Short week games (Thursday games)
        features_df['home_short_week'] = (features_df['home_rest_days'] < 6).astype(int)
        features_df['away_short_week'] = (features_df['away_rest_days'] < 6).astype(int)
        
        # Long rest (bye week)
        features_df['home_long_rest'] = (features_df['home_rest_days'] > 10).astype(int)
        features_df['away_long_rest'] = (features_df['away_rest_days'] > 10).astype(int)
        
        # Home field advantage (stronger in NFL)
        features_df['home_field_advantage'] = self._calculate_home_field_advantage(features_df)
        
        # Playoff implications (late season games matter more)
        features_df['playoff_implications'] = self._calculate_playoff_implications(features_df)
        
        # Travel distance approximation
        features_df['travel_burden'] = self._estimate_travel_burden(features_df)
        
        return features_df
    
    def _add_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add weather impact features."""
        logger.info("🌤️ Adding NFL weather features...")
        
        features_df = df.copy()
        
        # Temperature features (if available)
        if 'temperature' in features_df.columns:
            features_df['cold_weather'] = (features_df['temperature'] < 35).astype(int)
            features_df['very_cold'] = (features_df['temperature'] < 20).astype(int)
            features_df['hot_weather'] = (features_df['temperature'] > 85).astype(int)
        else:
            # Estimate based on month and dome status
            features_df['cold_weather'] = self._estimate_cold_weather(features_df)
            features_df['very_cold'] = 0
            features_df['hot_weather'] = 0
        
        # Wind features (if available)
        if 'wind_speed' in features_df.columns:
            features_df['windy'] = (features_df['wind_speed'] > 15).astype(int)
            features_df['very_windy'] = (features_df['wind_speed'] > 25).astype(int)
        else:
            features_df['windy'] = 0
            features_df['very_windy'] = 0
        
        # Dome/outdoor features
        if 'dome' in features_df.columns:
            features_df['dome_game'] = features_df['dome'].astype(int)
        else:
            features_df['dome_game'] = self._estimate_dome_games(features_df)
        
        # Weather advantage (outdoor teams vs dome teams in bad weather)
        features_df['weather_advantage'] = self._calculate_weather_advantage(features_df)
        
        return features_df
    
    def _add_advanced_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced NFL metrics."""
        logger.info("🧮 Adding advanced NFL metrics...")
        
        features_df = df.copy()
        
        # Pythagorean expectation (different exponent for NFL)
        features_df = self._add_pythagorean_expectation(features_df)
        
        # Strength of schedule
        features_df = self._add_strength_of_schedule(features_df)
        
        # Offensive/Defensive efficiency
        features_df = self._add_efficiency_metrics(features_df)
        
        # Turnover impact
        features_df = self._add_turnover_metrics(features_df)
        
        # Red zone efficiency (if available)
        if 'home_red_zone_efficiency' in features_df.columns:
            features_df['red_zone_diff'] = (
                features_df['home_red_zone_efficiency'] - features_df['away_red_zone_efficiency']
            )
        
        return features_df
    
    def _add_feature_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add feature interactions."""
        logger.info("🔗 Adding NFL feature interactions...")
        
        features_df = df.copy()
        
        # Define important feature pairs for interactions
        interaction_pairs = [
            ('home_form_4', 'away_form_4'),
            ('home_turnover_differential', 'away_turnover_differential'),
            ('rest_advantage', 'home_field_advantage'),
            ('divisional_game', 'home_form_4'),
            ('cold_weather', 'dome_game')
        ]
        
        # Filter pairs that exist in the DataFrame
        valid_pairs = [
            (feat1, feat2) for feat1, feat2 in interaction_pairs
            if feat1 in features_df.columns and feat2 in features_df.columns
        ]
        
        if valid_pairs:
            features_df = create_feature_interactions(
                features_df, valid_pairs, ['multiply', 'subtract']
            )
        
        return features_df
    
    # Helper methods specific to NFL
    def _calculate_rest_days(self, df: pd.DataFrame, team_id_col: str) -> pd.Series:
        """Calculate rest days since last game (NFL specific)."""
        rest_days = []
        
        for idx, row in df.iterrows():
            team_id = row[team_id_col]
            current_week = row['week']
            current_season = row['season']
            
            # Find last game for this team
            prev_mask = (
                (df['home_team_id'] == team_id) | (df['away_team_id'] == team_id)
            ) & (
                (df['season'] < current_season) | 
                ((df['season'] == current_season) & (df['week'] < current_week))
            )
            
            prev_games = df[prev_mask]
            
            if len(prev_games) == 0:
                rest_days.append(7)  # Default one week rest
            else:
                last_week = prev_games['week'].max()
                last_season = prev_games[prev_games['week'] == last_week]['season'].iloc[0]
                
                if last_season == current_season:
                    weeks_diff = current_week - last_week
                    rest_days.append(weeks_diff * 7)  # Approximate days
                else:
                    # Off-season rest (very long)
                    rest_days.append(180)  # About 6 months
        
        return pd.Series(rest_days, index=df.index)
    
    def _calculate_home_field_advantage(self, df: pd.DataFrame) -> pd.Series:
        """Calculate NFL home field advantage (typically stronger than other sports)."""
        return pd.Series([2.5] * len(df), index=df.index)  # NFL average HFA
    
    def _calculate_playoff_implications(self, df: pd.DataFrame) -> pd.Series:
        """Calculate playoff implications score."""
        implications = []
        
        for _, row in df.iterrows():
            week = row['week']
            
            if week >= 15:  # Final weeks have high playoff implications
                implications.append(1.0)
            elif week >= 12:  # Mid-season games have medium implications
                implications.append(0.6)
            elif week >= 8:  # Early games have some implications
                implications.append(0.3)
            else:  # Very early games have minimal implications
                implications.append(0.1)
        
        return pd.Series(implications, index=df.index)
    
    def _estimate_travel_burden(self, df: pd.DataFrame) -> pd.Series:
        """Estimate travel burden (simplified)."""
        # This could be enhanced with actual distance calculations
        return pd.Series([0.5] * len(df), index=df.index)  # Placeholder
    
    def _estimate_cold_weather(self, df: pd.DataFrame) -> pd.Series:
        """Estimate cold weather games based on month and location."""
        cold_weather = []
        
        for _, row in df.iterrows():
            month = row.get('month', 1)
            
            # Cold weather months
            if month in [11, 12, 1, 2]:  # Nov, Dec, Jan, Feb
                cold_weather.append(1)
            else:
                cold_weather.append(0)
        
        return pd.Series(cold_weather, index=df.index)
    
    def _estimate_dome_games(self, df: pd.DataFrame) -> pd.Series:
        """Estimate dome games (simplified)."""
        # This would need actual venue data
        return pd.Series([0.25] * len(df), index=df.index)  # About 25% of games in domes
    
    def _calculate_weather_advantage(self, df: pd.DataFrame) -> pd.Series:
        """Calculate weather advantage."""
        # Simplified - would need team-specific data
        return pd.Series([0.0] * len(df), index=df.index)  # Placeholder
    
    # ... (other helper methods similar to NBA but adapted for NFL)
    
    def _cleanup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final cleanup of features."""
        features_df = df.copy()
        
        # Remove obviously non-feature columns
        non_feature_cols = [
            'home_team_full_name', 'away_team_full_name', 'venue', 'city',
            'status', 'created_at', 'updated_at', 'season_type'
        ]
        
        features_df = features_df.drop(
            [col for col in non_feature_cols if col in features_df.columns], 
            axis=1
        )
        
        # Handle infinite values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        
        # Cap extreme values
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in features_df.columns:
                q99 = features_df[col].quantile(0.99)
                q01 = features_df[col].quantile(0.01)
                features_df[col] = features_df[col].clip(lower=q01, upper=q99)
        
        return features_df
    
    # Include other necessary helper methods adapted from NBA implementation...
