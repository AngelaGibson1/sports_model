# data/features/mlb_features.py 

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger
import warnings

from config.settings import Settings
from data.database.mlb import MLBDatabase
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

class MLBFeatureEngineer:
    """
    Comprehensive MLB feature engineering for game prediction models.
    Creates advanced features from raw MLB data for machine learning models.
    """
    
    def __init__(self, db: Optional[MLBDatabase] = None):
        """
        Initialize MLB feature engineer.
        
        Args:
            db: Optional MLB database instance
        """
        self.db = db or MLBDatabase()
        self.mlb_config = Settings.SPORT_CONFIGS['mlb']
        self.rolling_windows = Settings.ROLLING_WINDOWS['mlb']
        
        # MLB-specific feature configuration
        self.key_stats = [
            'runs_per_game', 'runs_allowed_per_game', 'batting_average', 'on_base_percentage',
            'slugging_percentage', 'earned_run_average', 'whip', 'strikeouts_per_nine',
            'walks_per_nine', 'fielding_percentage', 'home_runs_per_game'
        ]
        
        self.logger = logger
        
        logger.info("⚾ MLB Feature Engineer initialized")
    
    def engineer_game_features(self, 
                              seasons: Optional[List[int]] = None,
                              include_advanced: bool = True,
                              include_situational: bool = True,
                              include_weather: bool = True,
                              include_pitching_matchups: bool = True) -> pd.DataFrame:
        """
        Create comprehensive features for MLB game prediction.
        
        Args:
            seasons: Seasons to include in feature engineering
            include_advanced: Whether to include advanced metrics
            include_situational: Whether to include situational features
            include_weather: Whether to include weather features
            include_pitching_matchups: Whether to include pitcher vs batter matchups
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("🔧 Starting MLB feature engineering...")
        
        # Get base historical data
        historical_df = self.db.get_historical_data(seasons)
        
        if historical_df.empty:
            logger.warning("No historical data available for feature engineering")
            return pd.DataFrame()
        
        # Sort by date for proper time series processing
        historical_df = historical_df.sort_values(['date', 'game_id'])
        
        # Create base features
        features_df = self._create_base_features(historical_df)
        
        # Add rolling performance features
        features_df = self._add_rolling_performance_features(features_df)
        
        # Add team form and momentum features
        features_df = self._add_team_form_features(features_df)
        
        # Add head-to-head features
        features_df = self._add_head_to_head_features(features_df)
        
        if include_situational:
            # Add situational features (rest, series position, etc.)
            features_df = self._add_situational_features(features_df)
        
        if include_weather:
            # Add weather impact features (more important in baseball)
            features_df = self._add_weather_features(features_df)
        
        if include_pitching_matchups:
            # Add pitching matchup features
            features_df = self._add_pitching_features(features_df)
        
        if include_advanced:
            # Add advanced metrics
            features_df = self._add_advanced_metrics(features_df)
        
        # Add feature interactions
        features_df = self._add_feature_interactions(features_df)
        
        # Handle missing values
        features_df = handle_missing_values(features_df, strategy='smart')
        
        # Final cleanup
        features_df = self._cleanup_features(features_df)
        
        logger.info(f"✅ MLB feature engineering complete: {features_df.shape[0]} games, {features_df.shape[1]} features")
        
        return features_df
    
    def _create_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create base features from raw game data."""
        logger.info("📊 Creating base MLB features...")
        
        features_df = df.copy()
        
        # Basic game features
        features_df['total_runs'] = features_df['home_score'] + features_df['away_score']
        features_df['run_differential'] = features_df['home_score'] - features_df['away_score']
        features_df['home_win'] = (features_df['home_score'] > features_df['away_score']).astype(int)
        
        # NRFI (No Run First Inning) feature - will need to be calculated separately if inning data available
        features_df['nrfi'] = 0  # Placeholder - would need inning-by-inning data
        
        # Date and time features
        features_df['date'] = pd.to_datetime(features_df['date'])
        features_df['month'] = features_df['date'].dt.month
        features_df['day_of_week'] = features_df['date'].dt.dayofweek
        features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)
        features_df['is_day_game'] = 0  # Would need game time data
        features_df['is_night_game'] = 1  # Most MLB games are night games
        
        # Season features
        features_df['season_progress'] = self._calculate_season_progress(features_df)
        features_df['early_season'] = (features_df['month'] <= 5).astype(int)  # April-May
        features_df['late_season'] = (features_df['month'] >= 9).astype(int)  # September+
        features_df['playoff_race'] = (features_df['month'] >= 8).astype(int)  # August+
        
        # Team strength differentials (if stats available)
        stat_columns = [col for col in self.key_stats if f'home_{col}' in features_df.columns]
        
        for stat in stat_columns:
            home_col = f'home_{stat}'
            away_col = f'away_{stat}'
            
            if home_col in features_df.columns and away_col in features_df.columns:
                features_df[f'{stat}_diff'] = features_df[home_col] - features_df[away_col]
                if 'percentage' not in stat and 'average' not in stat:  # Don't create ratios for percentages/averages
                    features_df[f'{stat}_ratio'] = features_df[home_col] / (features_df[away_col] + 0.001)
        
        return features_df
    
    def _add_rolling_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling performance features for both teams."""
        logger.info("📈 Adding rolling MLB performance features...")
        
        features_df = df.copy()
        
        # Create team performance tracking
        home_performance = self._create_team_performance_tracking(features_df, 'home')
        away_performance = self._create_team_performance_tracking(features_df, 'away')
        
        # Add rolling statistics for multiple windows (smaller windows for baseball due to daily games)
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
        logger.info("🔥 Adding MLB team form features...")
        
        features_df = df.copy()
        
        # Create form tracking for home and away teams
        for team_type in ['home', 'away']:
            team_id_col = f'{team_type}_team_id'
            
            # Recent form (last 3, 7, 15 games - adapted for MLB's daily schedule)
            for window in [3, 7, 15]:
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
            features_df[f'{team_type}_home_form_7'] = self._calculate_venue_form(
                features_df, team_id_col, venue='home', window=7
            )
            features_df[f'{team_type}_away_form_7'] = self._calculate_venue_form(
                features_df, team_id_col, venue='away', window=7
            )
            
            # Run scoring trends
            features_df[f'{team_type}_runs_trend_7'] = self._calculate_runs_trend(
                features_df, team_id_col, team_type, window=7
            )
        
        # Form differentials
        for window in [3, 7, 15]:
            features_df[f'form_diff_{window}'] = (
                features_df[f'home_form_{window}'] - features_df[f'away_form_{window}']
            )
        
        return features_df
    
    def _add_head_to_head_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add head-to-head historical features."""
        logger.info("⚔️ Adding MLB head-to-head features...")
        
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
        """Add situational features specific to MLB."""
        logger.info("🎯 Adding MLB situational features...")
        
        features_df = df.copy()
        
        # Calculate rest days (less important in MLB due to daily games)
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
        
        # Series context (MLB plays series)
        features_df['series_game'] = self._calculate_series_position(features_df)
        
        # Travel burden (important in MLB)
        features_df['away_travel_burden'] = self._calculate_travel_burden(features_df)
        
        # Home field advantage (varies by park)
        features_df['home_field_advantage'] = self._calculate_home_field_advantage(features_df)
        
        # Playoff implications
        features_df['playoff_implications'] = self._calculate_playoff_implications(features_df)
        
        # Divisional games (important in MLB)
        features_df['divisional_game'] = self._calculate_divisional_games(features_df)
        
        # Interleague play
        features_df['interleague_game'] = self._calculate_interleague_games(features_df)
        
        return features_df
    
    def _add_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add weather impact features (very important in baseball)."""
        logger.info("🌤️ Adding MLB weather features...")
        
        features_df = df.copy()
        
        # Temperature features (if available)
        if 'temperature' in features_df.columns:
            features_df['cold_weather'] = (features_df['temperature'] < 50).astype(int)
            features_df['hot_weather'] = (features_df['temperature'] > 85).astype(int)
            features_df['optimal_temp'] = ((features_df['temperature'] >= 65) & 
                                          (features_df['temperature'] <= 78)).astype(int)
        else:
            # Estimate based on month and location
            features_df['cold_weather'] = self._estimate_cold_weather(features_df)
            features_df['hot_weather'] = self._estimate_hot_weather(features_df)
            features_df['optimal_temp'] = 0
        
        # Wind features (critical for home runs)
        if 'wind_speed' in features_df.columns and 'wind_direction' in features_df.columns:
            features_df['windy'] = (features_df['wind_speed'] > 10).astype(int)
            features_df['wind_out'] = self._calculate_wind_direction_impact(features_df)
            features_df['wind_in'] = (features_df['wind_out'] == 0).astype(int)
        else:
            features_df['windy'] = 0
            features_df['wind_out'] = 0
            features_df['wind_in'] = 0
        
        # Humidity features (affects ball flight)
        if 'humidity' in features_df.columns:
            features_df['high_humidity'] = (features_df['humidity'] > 70).astype(int)
        else:
            features_df['high_humidity'] = 0
        
        # Park factors (some parks favor hitters/pitchers)
        features_df['hitter_friendly_park'] = self._calculate_park_factors(features_df, 'hitter')
        features_df['pitcher_friendly_park'] = self._calculate_park_factors(features_df, 'pitcher')
        
        return features_df
    
    def _add_pitching_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pitching matchup features (crucial for baseball)."""
        logger.info("⚾ Adding MLB pitching features...")
        
        features_df = df.copy()
        
        # Starting pitcher features (would need pitcher data)
        # For now, create placeholders
        features_df['home_starter_era'] = 4.00  # League average
        features_df['away_starter_era'] = 4.00
        features_df['home_starter_whip'] = 1.30
        features_df['away_starter_whip'] = 1.30
        features_df['home_starter_k9'] = 8.0  # Strikeouts per 9 innings
        features_df['away_starter_k9'] = 8.0
        
        # Bullpen strength (would need bullpen data)
        features_df['home_bullpen_era'] = 4.20
        features_df['away_bullpen_era'] = 4.20
        
        # Pitcher vs batter handedness matchups
        features_df['favorable_matchups'] = self._calculate_handedness_advantage(features_df)
        
        # Starting pitcher rest
        features_df['home_starter_rest'] = 4  # Typical 5-man rotation
        features_df['away_starter_rest'] = 4
        
        # Pitcher differentials
        features_df['era_diff'] = features_df['home_starter_era'] - features_df['away_starter_era']
        features_df['whip_diff'] = features_df['home_starter_whip'] - features_df['away_starter_whip']
        features_df['k9_diff'] = features_df['home_starter_k9'] - features_df['away_starter_k9']
        
        return features_df
    
    def _add_advanced_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced MLB metrics."""
        logger.info("🧮 Adding advanced MLB metrics...")
        
        features_df = df.copy()
        
        # Pythagorean expectation (different exponent for MLB)
        features_df = self._add_pythagorean_expectation(features_df)
        
        # Strength of schedule
        features_df = self._add_strength_of_schedule(features_df)
        
        # OPS (On-base Plus Slugging) differentials
        if 'home_on_base_percentage' in features_df.columns and 'home_slugging_percentage' in features_df.columns:
            features_df['home_ops'] = features_df['home_on_base_percentage'] + features_df['home_slugging_percentage']
            features_df['away_ops'] = features_df['away_on_base_percentage'] + features_df['away_slugging_percentage']
            features_df['ops_diff'] = features_df['home_ops'] - features_df['away_ops']
        
        # Run differential vs expected (luck factor)
        features_df = self._add_run_differential_luck(features_df)
        
        # Clutch performance metrics
        features_df = self._add_clutch_metrics(features_df)
        
        return features_df
    
    def _add_feature_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add feature interactions."""
        logger.info("🔗 Adding MLB feature interactions...")
        
        features_df = df.copy()
        
        # Define important feature pairs for interactions
        interaction_pairs = [
            ('home_form_7', 'away_form_7'),
            ('home_starter_era', 'away_starter_era'),
            ('cold_weather', 'hitter_friendly_park'),
            ('wind_out', 'home_runs_per_game_diff'),
            ('divisional_game', 'home_form_7'),
            ('playoff_implications', 'home_starter_era')
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
    
    # Helper methods specific to MLB
    def _calculate_season_progress(self, df: pd.DataFrame) -> pd.Series:
        """Calculate season progress (0-1 scale) for MLB."""
        season_progress = []
        
        for _, row in df.iterrows():
            month = row['month']
            
            # MLB season: April (4) to October (10)
            if month < 4:
                progress = 0.0
            elif month > 10:
                progress = 1.0
            else:
                # Map April=0, October=1
                progress = (month - 4) / 6.0
            
            season_progress.append(progress)
        
        return pd.Series(season_progress, index=df.index)
    
    def _calculate_rest_days(self, df: pd.DataFrame, team_id_col: str) -> pd.Series:
        """Calculate rest days since last game (less important in MLB)."""
        rest_days = []
        
        for idx, row in df.iterrows():
            team_id = row[team_id_col]
            current_date = row['date']
            
            # Find last game for this team
            prev_mask = (
                (df['home_team_id'] == team_id) | (df['away_team_id'] == team_id)
            ) & (df['date'] < current_date)
            
            prev_games = df[prev_mask]
            
            if len(prev_games) == 0:
                rest_days.append(1)  # Default one day rest
            else:
                last_game_date = prev_games['date'].max()
                days_diff = (current_date - last_game_date).days
                rest_days.append(max(0, days_diff))
        
        return pd.Series(rest_days, index=df.index)
    
    def _calculate_series_position(self, df: pd.DataFrame) -> pd.Series:
        """Calculate position within a series (1st, 2nd, 3rd game, etc.)."""
        # This would need more sophisticated logic to track series
        # For now, return a placeholder
        return pd.Series([1] * len(df), index=df.index)
    
    def _calculate_travel_burden(self, df: pd.DataFrame) -> pd.Series:
        """Calculate away team travel burden."""
        # Simplified - would need actual geographic data
        return pd.Series([0.5] * len(df), index=df.index)
    
    def _calculate_home_field_advantage(self, df: pd.DataFrame) -> pd.Series:
        """Calculate park-specific home field advantage."""
        # MLB HFA varies by park - some parks favor pitchers, others hitters
        return pd.Series([0.54] * len(df), index=df.index)  # MLB average home win %
    
    def _calculate_playoff_implications(self, df: pd.DataFrame) -> pd.Series:
        """Calculate playoff implications score."""
        implications = []
        
        for _, row in df.iterrows():
            month = row['month']
            season_progress = row.get('season_progress', 0.5)
            
            if month >= 9:  # September+ - playoff race
                implications.append(1.0)
            elif month >= 8:  # August - getting important
                implications.append(0.7)
            elif season_progress > 0.6:  # Late season
                implications.append(0.5)
            else:  # Early season
                implications.append(0.2)
        
        return pd.Series(implications, index=df.index)
    
    def _calculate_divisional_games(self, df: pd.DataFrame) -> pd.Series:
        """Identify divisional games (more important in MLB)."""
        # Would need team division data
        return pd.Series([0.25] * len(df), index=df.index)  # ~25% of games are divisional
    
    def _calculate_interleague_games(self, df: pd.DataFrame) -> pd.Series:
        """Identify interleague games (AL vs NL)."""
        # Would need league data
        return pd.Series([0.1] * len(df), index=df.index)  # ~10% are interleague
    
    def _estimate_cold_weather(self, df: pd.DataFrame) -> pd.Series:
        """Estimate cold weather games."""
        cold_weather = []
        
        for _, row in df.iterrows():
            month = row.get('month', 6)
            
            if month in [4, 10]:  # April, October
                cold_weather.append(1)
            elif month in [5, 9]:  # May, September
                cold_weather.append(0.3)
            else:
                cold_weather.append(0)
        
        return pd.Series(cold_weather, index=df.index)
    
    def _estimate_hot_weather(self, df: pd.DataFrame) -> pd.Series:
        """Estimate hot weather games."""
        hot_weather = []
        
        for _, row in df.iterrows():
            month = row.get('month', 6)
            
            if month in [7, 8]:  # July, August
                hot_weather.append(1)
            elif month in [6, 9]:  # June, September
                hot_weather.append(0.5)
            else:
                hot_weather.append(0)
        
        return pd.Series(hot_weather, index=df.index)
    
    def _calculate_wind_direction_impact(self, df: pd.DataFrame) -> pd.Series:
        """Calculate if wind is helping (out) or hurting (in) offense."""
        # Would need park-specific wind direction data
        return pd.Series([0.5] * len(df), index=df.index)  # 50% favorable wind
    
    def _calculate_park_factors(self, df: pd.DataFrame, factor_type: str) -> pd.Series:
        """Calculate park factors."""
        # Would need park-specific data
        if factor_type == 'hitter':
            return pd.Series([0.5] * len(df), index=df.index)  # 50% hitter-friendly
        else:
            return pd.Series([0.5] * len(df), index=df.index)  # 50% pitcher-friendly
    
    def _calculate_handedness_advantage(self, df: pd.DataFrame) -> pd.Series:
        """Calculate pitcher vs batter handedness advantages."""
        # Would need pitcher/batter handedness data
        return pd.Series([0.5] * len(df), index=df.index)  # Neutral matchup
    
    def _add_pythagorean_expectation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Pythagorean expectation for wins (different exponent for MLB)."""
        features_df = df.copy()
        
        for team_type in ['home', 'away']:
            rpg_col = f'{team_type}_runs_per_game'
            rapg_col = f'{team_type}_runs_allowed_per_game'
            
            if rpg_col in features_df.columns and rapg_col in features_df.columns:
                pythag_col = f'{team_type}_pythagorean_wins'
                
                # MLB Pythagorean exponent is typically around 1.83
                exponent = 1.83
                features_df[pythag_col] = (
                    features_df[rpg_col] ** exponent /
                    (features_df[rpg_col] ** exponent + features_df[rapg_col] ** exponent)
                )
        
        # Pythagorean differential
        if 'home_pythagorean_wins' in features_df.columns and 'away_pythagorean_wins' in features_df.columns:
            features_df['pythagorean_diff'] = (
                features_df['home_pythagorean_wins'] - features_df['away_pythagorean_wins']
            )
        
        return features_df
    
    def _add_strength_of_schedule(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add strength of schedule metrics (simplified)."""
        features_df = df.copy()
        
        # Placeholder implementation
        features_df['home_sos'] = 0.5
        features_df['away_sos'] = 0.5
        features_df['sos_diff'] = 0.0
        
        return features_df
    
    def _add_run_differential_luck(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add run differential luck factors."""
        features_df = df.copy()
        
        # Placeholder - would need season-long run differential data
        features_df['home_run_diff_luck'] = 0.0
        features_df['away_run_diff_luck'] = 0.0
        
        return features_df
    
    def _add_clutch_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add clutch performance metrics."""
        features_df = df.copy()
        
        # Placeholder - would need clutch situation data
        features_df['home_clutch_performance'] = 0.5
        features_df['away_clutch_performance'] = 0.5
        
        return features_df
    
    # Additional helper methods similar to NBA implementation...
    def _create_team_performance_tracking(self, df: pd.DataFrame, team_type: str) -> pd.DataFrame:
        """Create team performance tracking DataFrame."""
        team_id_col = f'{team_type}_team_id'
        score_col = f'{team_type}_score'
        opp_score_col = 'away_score' if team_type == 'home' else 'home_score'
        
        performance_data = []
        
        for _, row in df.iterrows():
            perf_row = {
                'team_id': row[team_id_col],
                'date': row['date'],
                'game_id': row['game_id'],
                'runs_scored': row[score_col],
                'runs_allowed': row[opp_score_col],
                'win': 1 if row[score_col] > row[opp_score_col] else 0,
                'venue': team_type
            }
            
            # Add team stats if available
            for stat in self.key_stats:
                stat_col = f'{team_type}_{stat}'
                if stat_col in df.columns:
                    perf_row[stat] = row[stat_col]
            
            performance_data.append(perf_row)
        
        return pd.DataFrame(performance_data).sort_values(['team_id', 'date'])
    
    def _calculate_team_rolling_stats(self, team_df: pd.DataFrame, 
                                    window: int, prefix: str) -> pd.DataFrame:
        """Calculate rolling statistics for a team."""
        rolling_df = team_df.copy()
        
        # Rolling averages for key metrics
        rolling_cols = ['runs_scored', 'runs_allowed', 'win']
        rolling_cols.extend([col for col in self.key_stats if col in rolling_df.columns])
        
        rolling_df = calculate_rolling_averages(
            rolling_df, rolling_cols, [window], group_by='team_id'
        )
        
        # Rename columns with prefix
        rename_dict = {}
        for col in rolling_cols:
            old_name = f'{col}_roll_{window}'
            new_name = f'{prefix}_{col}_avg'
            if old_name in rolling_df.columns:
                rename_dict[old_name] = new_name
        
        rolling_df = rolling_df.rename(columns=rename_dict)
        
        return rolling_df[['team_id', 'game_id'] + list(rename_dict.values())]
    
    def _merge_rolling_stats(self, features_df: pd.DataFrame, 
                           home_rolling: pd.DataFrame, 
                           away_rolling: pd.DataFrame,
                           window_name: str) -> pd.DataFrame:
        """Merge rolling statistics back to features DataFrame."""
        # Similar to NBA implementation
        # Merge home team rolling stats
        features_df = features_df.merge(
            home_rolling,
            left_on=['home_team_id', 'game_id'],
            right_on=['team_id', 'game_id'],
            how='left',
            suffixes=('', '_home_roll')
        ).drop('team_id', axis=1, errors='ignore')
        
        # Merge away team rolling stats
        features_df = features_df.merge(
            away_rolling,
            left_on=['away_team_id', 'game_id'],
            right_on=['team_id', 'game_id'],
            how='left',
            suffixes=('', '_away_roll')
        ).drop('team_id', axis=1, errors='ignore')
        
        return features_df
    
    def _calculate_team_form_rolling(self, df: pd.DataFrame, 
                                   team_id_col: str, window: int) -> pd.Series:
        """Calculate rolling team form (win percentage)."""
        # Similar logic to NBA implementation, adapted for MLB
        team_wins = []
        
        for idx, row in df.iterrows():
            team_id = row[team_id_col]
            game_date = row['date']
            
            # Get recent games for this team before current game
            recent_mask = (
                (df[team_id_col] == team_id) |
                (df['home_team_id' if 'away' in team_id_col else 'away_team_id'] == team_id)
            ) & (df['date'] < game_date)
            
            recent_games = df[recent_mask].tail(window)
            
            if len(recent_games) == 0:
                team_wins.append(0.5)  # Neutral form for new teams
                continue
            
            # Calculate wins for this team
            wins = 0
            for _, game in recent_games.iterrows():
                if game['home_team_id'] == team_id:
                    wins += 1 if game['home_score'] > game['away_score'] else 0
                else:
                    wins += 1 if game['away_score'] > game['home_score'] else 0
            
            team_wins.append(wins / len(recent_games))
        
        return pd.Series(team_wins, index=df.index)
    
    def _calculate_win_streaks(self, df: pd.DataFrame, 
                             team_id_col: str, streak_type: str) -> pd.Series:
        """Calculate current win/loss streaks."""
        # Similar to NBA implementation
        streaks = []
        
        for idx, row in df.iterrows():
            team_id = row[team_id_col]
            game_date = row['date']
            
            # Get games before current game
            prev_mask = (
                (df[team_id_col] == team_id) |
                (df['home_team_id' if 'away' in team_id_col else 'away_team_id'] == team_id)
            ) & (df['date'] < game_date)
            
            prev_games = df[prev_mask].sort_values('date', ascending=False)
            
            streak = 0
            for _, game in prev_games.iterrows():
                # Determine if team won this game
                if game['home_team_id'] == team_id:
                    won = game['home_score'] > game['away_score']
                else:
                    won = game['away_score'] > game['home_score']
                
                target_result = (streak_type == 'win')
                
                if won == target_result:
                    streak += 1
                else:
                    break
            
            streaks.append(streak)
        
        return pd.Series(streaks, index=df.index)
    
    def _calculate_venue_form(self, df: pd.DataFrame, team_id_col: str, 
                            venue: str, window: int) -> pd.Series:
        """Calculate form at specific venue (home/away)."""
        # Similar to NBA implementation
        venue_forms = []
        
        for idx, row in df.iterrows():
            team_id = row[team_id_col]
            game_date = row['date']
            
            if venue == 'home':
                venue_mask = (df['home_team_id'] == team_id)
            else:
                venue_mask = (df['away_team_id'] == team_id)
            
            recent_mask = venue_mask & (df['date'] < game_date)
            recent_games = df[recent_mask].tail(window)
            
            if len(recent_games) == 0:
                venue_forms.append(0.5)
                continue
            
            wins = 0
            for _, game in recent_games.iterrows():
                if venue == 'home':
                    wins += 1 if game['home_score'] > game['away_score'] else 0
                else:
                    wins += 1 if game['away_score'] > game['home_score'] else 0
            
            venue_forms.append(wins / len(recent_games))
        
        return pd.Series(venue_forms, index=df.index)
    
    def _calculate_runs_trend(self, df: pd.DataFrame, team_id_col: str, 
                            team_type: str, window: int) -> pd.Series:
        """Calculate recent run scoring trend."""
        trends = []
        
        for idx, row in df.iterrows():
            team_id = row[team_id_col]
            game_date = row['date']
            
            # Get recent games
            recent_mask = (
                (df['home_team_id'] == team_id) | (df['away_team_id'] == team_id)
            ) & (df['date'] < game_date)
            
            recent_games = df[recent_mask].tail(window)
            
            if len(recent_games) == 0:
                trends.append(4.5)  # MLB average runs per game
                continue
            
            runs_scored = []
            for _, game in recent_games.iterrows():
                if game['home_team_id'] == team_id:
                    runs_scored.append(game['home_score'])
                else:
                    runs_scored.append(game['away_score'])
            
            trends.append(np.mean(runs_scored))
        
        return pd.Series(trends, index=df.index)
    
    def _get_historical_h2h(self, home_team: int, away_team: int, 
                          game_date: datetime, season: int, df: pd.DataFrame) -> Dict[str, float]:
        """Get historical head-to-head statistics."""
        # Similar to NBA implementation but with baseball-specific stats
        h2h_mask = (
            ((df['home_team_id'] == home_team) & (df['away_team_id'] == away_team)) |
            ((df['home_team_id'] == away_team) & (df['away_team_id'] == home_team))
        ) & (df['date'] < game_date)
        
        h2h_games = df[h2h_mask]
        
        if len(h2h_games) == 0:
            return {
                'h2h_games_played': 0,
                'h2h_home_wins': 0,
                'h2h_home_win_pct': 0.5,
                'h2h_avg_total_runs': 9.0,  # MLB average
                'h2h_avg_run_margin': 0
            }
        
        # Calculate H2H statistics
        home_wins = 0
        total_runs = []
        run_margins = []
        
        for _, game in h2h_games.iterrows():
            if game['home_team_id'] == home_team:
                # Current home team was home in this H2H game
                won = game['home_score'] > game['away_score']
                margin = game['home_score'] - game['away_score']
            else:
                # Current home team was away in this H2H game
                won = game['away_score'] > game['home_score']
                margin = game['away_score'] - game['home_score']
            
            if won:
                home_wins += 1
            
            total_runs.append(game['home_score'] + game['away_score'])
            run_margins.append(margin)
        
        return {
            'h2h_games_played': len(h2h_games),
            'h2h_home_wins': home_wins,
            'h2h_home_win_pct': home_wins / len(h2h_games),
            'h2h_avg_total_runs': np.mean(total_runs),
            'h2h_avg_run_margin': np.mean(run_margins)
        }
    
    def _cleanup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final cleanup of features."""
        features_df = df.copy()
        
        # Remove obviously non-feature columns
        non_feature_cols = [
            'home_team_full_name', 'away_team_full_name', 'venue', 'city',
            'status', 'created_at', 'updated_at', 'weather', 'inning', 'inning_half'
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
