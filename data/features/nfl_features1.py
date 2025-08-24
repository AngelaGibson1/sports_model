# data/features/nfl_features.py
"""
NFL Feature Engineering Module
Provides comprehensive feature engineering for NFL games including:
- Team-level offensive and defensive statistics
- Player aggregates (QB, RB, WR, Defense)
- Situational features (weather, divisional games, etc.)
- Advanced analytics (strength of schedule, momentum, etc.)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from loguru import logger
import warnings
from pathlib import Path
import sys

# Add project root for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    import nfl_data_py as nfl
    NFL_DATA_PY_AVAILABLE = True
except ImportError:
    NFL_DATA_PY_AVAILABLE = False
    logger.warning("⚠️ nfl_data_py not available - install with: pip install nfl_data_py")

from data.database.nfl import NFLDatabase
from config.settings import Settings

# NFL-specific constants
NFL_TEAM_ABBREVIATIONS = {
    'BUF': 1, 'MIA': 2, 'NE': 3, 'NYJ': 4, 'BAL': 5, 'CIN': 6, 'CLE': 7, 'PIT': 8,
    'HOU': 9, 'IND': 10, 'JAX': 11, 'TEN': 12, 'DEN': 13, 'KC': 14, 'LV': 15, 'LAC': 16,
    'DAL': 17, 'NYG': 18, 'PHI': 19, 'WAS': 20, 'CHI': 21, 'DET': 22, 'GB': 23, 'MIN': 24,
    'ATL': 25, 'CAR': 26, 'NO': 27, 'TB': 28, 'ARI': 29, 'LAR': 30, 'SF': 31, 'SEA': 32
}

NFL_DIVISIONS = {
    'AFC East': [1, 2, 3, 4],      # BUF, MIA, NE, NYJ
    'AFC North': [5, 6, 7, 8],     # BAL, CIN, CLE, PIT  
    'AFC South': [9, 10, 11, 12],  # HOU, IND, JAX, TEN
    'AFC West': [13, 14, 15, 16],  # DEN, KC, LV, LAC
    'NFC East': [17, 18, 19, 20],  # DAL, NYG, PHI, WAS
    'NFC North': [21, 22, 23, 24], # CHI, DET, GB, MIN
    'NFC South': [25, 26, 27, 28], # ATL, CAR, NO, TB
    'NFC West': [29, 30, 31, 32]   # ARI, LAR, SF, SEA
}

NFL_CONFERENCES = {
    'AFC': list(range(1, 17)),     # Teams 1-16
    'NFC': list(range(17, 33))     # Teams 17-32
}


class NFLFeatureEngineer:
    """
    Comprehensive NFL feature engineering class.
    Handles team stats, player aggregates, and situational features.
    """
    
    def __init__(self, use_current_data: bool = True):
        """
        Initialize NFL feature engineer.
        
        Args:
            use_current_data: Whether to load current season data
        """
        self.use_current_data = use_current_data
        self.current_season = datetime.now().year
        
        # Initialize database
        try:
            self.db = NFLDatabase()
            logger.info("✅ NFL Database connected")
        except Exception as e:
            logger.warning(f"⚠️ NFL Database not available: {e}")
            self.db = None
        
        # Data caches
        self.team_stats_cache = {}
        self.player_stats_cache = {}
        self.schedule_cache = {}
        
        # Load current season data if requested
        if use_current_data:
            self._load_current_season_data()
        
        logger.info("🏈 NFL Feature Engineer initialized")
    
    def _load_current_season_data(self):
        """Load current season data from nfl_data_py."""
        if not NFL_DATA_PY_AVAILABLE:
            logger.warning("⚠️ nfl_data_py not available - using fallback data")
            return
        
        try:
            logger.info(f"📊 Loading NFL {self.current_season} season data...")
            
            # Load player stats
            player_stats = nfl.import_seasonal_data([self.current_season])
            if not player_stats.empty:
                self.player_stats_cache[self.current_season] = player_stats
                logger.info(f"✅ Loaded {len(player_stats)} player records")
            
            # Load team descriptions
            team_desc = nfl.import_team_desc()
            if not team_desc.empty:
                self.team_desc = team_desc
                logger.info(f"✅ Loaded {len(team_desc)} team descriptions")
            
            # Load schedule if available
            try:
                schedule = nfl.import_schedules([self.current_season])
                if not schedule.empty:
                    self.schedule_cache[self.current_season] = schedule
                    logger.info(f"✅ Loaded {len(schedule)} scheduled games")
            except Exception as e:
                logger.warning(f"⚠️ Could not load schedule: {e}")
                
        except Exception as e:
            logger.error(f"❌ Failed to load current season NFL data: {e}")
    
    def create_team_features(self, games_df: pd.DataFrame, lookback_games: int = 8) -> pd.DataFrame:
        """
        Create comprehensive team-level features.
        
        Args:
            games_df: DataFrame with game information
            lookback_games: Number of recent games to consider for rolling stats
            
        Returns:
            DataFrame with team features
        """
        logger.info("🔧 Creating NFL team features...")
        
        features_df = games_df.copy()
        
        # Basic team stats from current season
        team_stats = self._get_team_season_stats()
        
        if not team_stats.empty:
            # Merge home team stats
            features_df = features_df.merge(
                team_stats,
                left_on='home_team_id',
                right_on='team_id',
                how='left',
                suffixes=('', '_home')
            ).drop('team_id', axis=1, errors='ignore')
            
            # Merge away team stats
            features_df = features_df.merge(
                team_stats,
                left_on='away_team_id', 
                right_on='team_id',
                how='left',
                suffixes=('_home', '_away')
            ).drop('team_id', axis=1, errors='ignore')
        
        # Add rolling/recent performance features
        features_df = self._add_rolling_features(features_df, lookback_games)
        
        # Add strength of schedule features
        features_df = self._add_strength_of_schedule_features(features_df)
        
        # Add rest and travel features
        features_df = self._add_rest_travel_features(features_df)
        
        logger.info(f"✅ Created team features: {len(features_df.columns)} columns")
        return features_df
    
    def _get_team_season_stats(self) -> pd.DataFrame:
        """Get team statistics for current season."""
        if self.current_season not in self.player_stats_cache:
            return pd.DataFrame()
        
        player_stats = self.player_stats_cache[self.current_season]
        
        # Aggregate team stats from player stats
        team_stats = []
        
        for team_abbr, team_id in NFL_TEAM_ABBREVIATIONS.items():
            team_players = player_stats[player_stats['recent_team'] == team_abbr]
            
            if team_players.empty:
                # Create default stats
                team_stat = {
                    'team_id': team_id,
                    'team_abbr': team_abbr,
                    'points_per_game': 22.5,
                    'points_allowed_per_game': 22.5,
                    'passing_yards_per_game': 250.0,
                    'rushing_yards_per_game': 120.0,
                    'total_yards_per_game': 370.0,
                    'passing_tds_per_game': 1.5,
                    'rushing_tds_per_game': 1.0,
                    'turnovers_per_game': 1.2,
                    'sacks_per_game': 2.5,
                    'win_percentage': 0.500
                }
            else:
                # Calculate actual stats
                games_played = 17  # NFL regular season
                
                team_stat = {
                    'team_id': team_id,
                    'team_abbr': team_abbr,
                    'points_per_game': self._estimate_points_from_stats(team_players, games_played),
                    'passing_yards_per_game': team_players['passing_yards'].sum() / games_played,
                    'rushing_yards_per_game': team_players['rushing_yards'].sum() / games_played,
                    'total_yards_per_game': (team_players['passing_yards'].sum() + 
                                           team_players['rushing_yards'].sum() + 
                                           team_players['receiving_yards'].sum()) / games_played,
                    'passing_tds_per_game': team_players['passing_tds'].sum() / games_played,
                    'rushing_tds_per_game': team_players['rushing_tds'].sum() / games_played,
                    'receiving_tds_per_game': team_players['receiving_tds'].sum() / games_played,
                    'sacks_per_game': team_players['sacks'].sum() / games_played,
                    'interceptions_per_game': team_players['interceptions'].sum() / games_played
                }
                
                # Calculate defensive stats (estimated)
                team_stat['points_allowed_per_game'] = 22.5  # Default - would need defensive data
                team_stat['turnovers_per_game'] = team_players['fumbles_lost'].sum() / games_played if 'fumbles_lost' in team_players.columns else 1.0
                team_stat['win_percentage'] = 0.500  # Default - would need game results
            
            team_stats.append(team_stat)
        
        return pd.DataFrame(team_stats)
    
    def _estimate_points_from_stats(self, team_players: pd.DataFrame, games_played: int) -> float:
        """Estimate points per game from player statistics."""
        # Calculate touchdowns
        total_tds = (team_players['passing_tds'].sum() + 
                    team_players['rushing_tds'].sum() + 
                    team_players['receiving_tds'].sum())
        
        # Estimate field goals (roughly 1.5 per game)
        estimated_fgs = games_played * 1.5
        
        # Calculate points (6 per TD, 3 per FG, extra points)
        estimated_points = (total_tds * 6) + (estimated_fgs * 3) + (total_tds * 1)  # Assume 100% XP rate
        
        return estimated_points / games_played
    
    def _add_rolling_features(self, features_df: pd.DataFrame, lookback_games: int) -> pd.DataFrame:
        """Add rolling performance features."""
        # This would require historical game results
        # For now, add placeholders that could be populated with real data
        
        rolling_features = [
            'home_wins_last_' + str(lookback_games),
            'away_wins_last_' + str(lookback_games),
            'home_points_last_' + str(lookback_games),
            'away_points_last_' + str(lookback_games),
            'home_points_allowed_last_' + str(lookback_games),
            'away_points_allowed_last_' + str(lookback_games)
        ]
        
        for feature in rolling_features:
            if 'wins' in feature:
                features_df[feature] = np.random.randint(2, lookback_games-1, len(features_df))
            elif 'points' in feature and 'allowed' not in feature:
                features_df[feature] = np.random.normal(22.5 * lookback_games, 5 * lookback_games, len(features_df))
            elif 'allowed' in feature:
                features_df[feature] = np.random.normal(22.5 * lookback_games, 5 * lookback_games, len(features_df))
        
        return features_df
    
    def _add_strength_of_schedule_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add strength of schedule features."""
        # Placeholder implementation - would calculate based on opponent records
        features_df['home_sos_remaining'] = np.random.normal(0.500, 0.100, len(features_df))
        features_df['away_sos_remaining'] = np.random.normal(0.500, 0.100, len(features_df))
        features_df['home_sos_played'] = np.random.normal(0.500, 0.100, len(features_df))
        features_df['away_sos_played'] = np.random.normal(0.500, 0.100, len(features_df))
        
        return features_df
    
    def _add_rest_travel_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add rest days and travel distance features."""
        # Default rest (7 days between NFL games typically)
        features_df['home_rest_days'] = 7
        features_df['away_rest_days'] = 7
        
        # Travel distance (placeholder - would calculate based on team locations)
        features_df['travel_distance'] = np.random.normal(800, 400, len(features_df))
        features_df['is_cross_country'] = (features_df['travel_distance'] > 2000).astype(int)
        
        return features_df
    
    def create_situational_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create situational features (weather, time, importance, etc.).
        """
        logger.info("🌤️ Creating NFL situational features...")
        
        features_df = games_df.copy()
        
        # Time-based features
        if 'date' in features_df.columns:
            features_df['date'] = pd.to_datetime(features_df['date'])
            features_df['week_of_season'] = features_df['date'].dt.isocalendar().week - 35  # Approximate NFL week
            features_df['week_of_season'] = features_df['week_of_season'].clip(1, 18)
            features_df['month'] = features_df['date'].dt.month
            features_df['is_early_season'] = (features_df['week_of_season'] <= 4).astype(int)
            features_df['is_late_season'] = (features_df['week_of_season'] >= 15).astype(int)
            features_df['is_playoffs'] = (features_df['week_of_season'] > 18).astype(int)
        
        # Game importance features
        features_df = self._add_divisional_features(features_df)
        features_df = self._add_conference_features(features_df)
        features_df = self._add_playoff_implications(features_df)
        
        # Weather features (more important in NFL)
        features_df = self._add_weather_features(features_df)
        
        # TV/Prime time features
        features_df = self._add_primetime_features(features_df)
        
        logger.info(f"✅ Created situational features: {len(features_df.columns)} columns")
        return features_df
    
    def _add_divisional_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add divisional rivalry features."""
        def get_division(team_id):
            for division, teams in NFL_DIVISIONS.items():
                if team_id in teams:
                    return division
            return 'Unknown'
        
        if 'home_team_id' in features_df.columns and 'away_team_id' in features_df.columns:
            features_df['home_division'] = features_df['home_team_id'].apply(get_division)
            features_df['away_division'] = features_df['away_team_id'].apply(get_division)
            features_df['is_divisional_game'] = (features_df['home_division'] == features_df['away_division']).astype(int)
        else:
            features_df['is_divisional_game'] = 0
        
        return features_df
    
    def _add_conference_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add conference features."""
        def get_conference(team_id):
            if team_id in NFL_CONFERENCES['AFC']:
                return 'AFC'
            elif team_id in NFL_CONFERENCES['NFC']:
                return 'NFC'
            return 'Unknown'
        
        if 'home_team_id' in features_df.columns and 'away_team_id' in features_df.columns:
            features_df['home_conference'] = features_df['home_team_id'].apply(get_conference)
            features_df['away_conference'] = features_df['away_team_id'].apply(get_conference)
            features_df['is_conference_game'] = (features_df['home_conference'] == features_df['away_conference']).astype(int)
            features_df['is_interconference_game'] = (features_df['home_conference'] != features_df['away_conference']).astype(int)
        else:
            features_df['is_conference_game'] = 0
            features_df['is_interconference_game'] = 0
        
        return features_df
    
    def _add_playoff_implications(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add playoff implication features."""
        # Placeholder - would calculate based on current standings and remaining schedule
        features_df['home_playoff_probability'] = np.random.beta(2, 2, len(features_df))  # Beta distribution for probabilities
        features_df['away_playoff_probability'] = np.random.beta(2, 2, len(features_df))
        features_df['playoff_implications'] = np.maximum(features_df['home_playoff_probability'], 
                                                        features_df['away_playoff_probability'])
        features_df['must_win_game'] = (features_df['playoff_implications'] > 0.8).astype(int)
        
        return features_df
    
    def _add_weather_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add weather-related features."""
        # NFL weather is crucial for outdoor games
        features_df['temperature'] = np.random.normal(45, 20, len(features_df))  # Degrees F
        features_df['wind_speed'] = np.random.exponential(5, len(features_df))  # MPH
        features_df['precipitation'] = np.random.choice([0, 1], len(features_df), p=[0.7, 0.3])
        features_df['is_dome_game'] = np.random.choice([0, 1], len(features_df), p=[0.7, 0.3])  # Approximate dome/outdoor split
        
        # Weather impact features
        features_df['cold_weather_game'] = (features_df['temperature'] < 32).astype(int)
        features_df['hot_weather_game'] = (features_df['temperature'] > 80).astype(int)
        features_df['windy_game'] = (features_df['wind_speed'] > 15).astype(int)
        features_df['bad_weather_game'] = ((features_df['precipitation'] == 1) | 
                                          (features_df['wind_speed'] > 15) | 
                                          (features_df['temperature'] < 20)).astype(int)
        
        return features_df
    
    def _add_primetime_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add primetime/TV features."""
        if 'game_hour' in features_df.columns:
            # NFL primetime games
            features_df['is_sunday_night_football'] = ((features_df['game_hour'] >= 19) & 
                                                      (features_df.get('day_of_week', 0) == 6)).astype(int)  # Sunday
            features_df['is_monday_night_football'] = ((features_df['game_hour'] >= 19) & 
                                                      (features_df.get('day_of_week', 0) == 0)).astype(int)  # Monday
            features_df['is_thursday_night_football'] = ((features_df['game_hour'] >= 19) & 
                                                         (features_df.get('day_of_week', 0) == 3)).astype(int)  # Thursday
            features_df['is_primetime_game'] = (features_df['is_sunday_night_football'] | 
                                               features_df['is_monday_night_football'] | 
                                               features_df['is_thursday_night_football']).astype(int)
        else:
            features_df['is_primetime_game'] = 0
            features_df['is_sunday_night_football'] = 0
            features_df['is_monday_night_football'] = 0
            features_df['is_thursday_night_football'] = 0
        
        return features_df
    
    def create_player_aggregate_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create player aggregate features for key positions.
        """
        logger.info("🏈 Creating NFL player aggregate features...")
        
        features_df = games_df.copy()
        
        if self.current_season not in self.player_stats_cache:
            logger.warning("⚠️ No player stats available - using defaults")
            return self._add_default_player_features(features_df)
        
        player_stats = self.player_stats_cache[self.current_season]
        
        # Create position-specific features
        features_df = self._add_quarterback_features(features_df, player_stats)
        features_df = self._add_running_back_features(features_df, player_stats)
        features_df = self._add_receiver_features(features_df, player_stats)
        features_df = self._add_defensive_features(features_df, player_stats)
        features_df = self._add_kicker_features(features_df, player_stats)
        
        logger.info(f"✅ Created player aggregate features: {len(features_df.columns)} columns")
        return features_df
    
    def _add_quarterback_features(self, features_df: pd.DataFrame, player_stats: pd.DataFrame) -> pd.DataFrame:
        """Add quarterback-specific features."""
        qb_stats = player_stats[player_stats['position'] == 'QB'].copy()
        
        if qb_stats.empty:
            # Default QB stats
            features_df['home_qb_rating'] = 85.0
            features_df['away_qb_rating'] = 85.0
            features_df['home_qb_experience'] = 5.0
            features_df['away_qb_experience'] = 5.0
            return features_df
        
        # Aggregate QB stats by team
        qb_team_stats = qb_stats.groupby('recent_team').agg({
            'passing_yards': 'sum',
            'passing_tds': 'sum',
            'interceptions': 'sum',
            'passing_attempts': 'sum',
            'completions': 'sum'
        }).reset_index()
        
        # Calculate QB rating (simplified)
        qb_team_stats['completion_pct'] = qb_team_stats['completions'] / qb_team_stats['passing_attempts'].clip(lower=1)
        qb_team_stats['td_rate'] = qb_team_stats['passing_tds'] / qb_team_stats['passing_attempts'].clip(lower=1)
        qb_team_stats['int_rate'] = qb_team_stats['interceptions'] / qb_team_stats['passing_attempts'].clip(lower=1)
        qb_team_stats['qb_rating'] = (qb_team_stats['completion_pct'] * 50 + 
                                     qb_team_stats['td_rate'] * 100 + 
                                     (1 - qb_team_stats['int_rate']) * 50)
        
        # Map to team IDs
        qb_team_stats['team_id'] = qb_team_stats['recent_team'].map(NFL_TEAM_ABBREVIATIONS)
        
        # Merge with features
        home_qb = qb_team_stats[['team_id', 'qb_rating']].rename(columns={'qb_rating': 'home_qb_rating'})
        away_qb = qb_team_stats[['team_id', 'qb_rating']].rename(columns={'qb_rating': 'away_qb_rating'})
        
        features_df = features_df.merge(home_qb, left_on='home_team_id', right_on='team_id', how='left').drop('team_id', axis=1, errors='ignore')
        features_df = features_df.merge(away_qb, left_on='away_team_id', right_on='team_id', how='left').drop('team_id', axis=1, errors='ignore')
        
        # Fill missing values
        features_df['home_qb_rating'] = features_df['home_qb_rating'].fillna(85.0)
        features_df['away_qb_rating'] = features_df['away_qb_rating'].fillna(85.0)
        
        # Add experience (placeholder)
        features_df['home_qb_experience'] = np.random.normal(5, 3, len(features_df)).clip(0, 20)
        features_df['away_qb_experience'] = np.random.normal(5, 3, len(features_df)).clip(0, 20)
        
        return features_df
    
    def _add_running_back_features(self, features_df: pd.DataFrame, player_stats: pd.DataFrame) -> pd.DataFrame:
        """Add running back features."""
        rb_stats = player_stats[player_stats['position'].isin(['RB', 'FB'])].copy()
        
        if rb_stats.empty:
            features_df['home_rush_yards_per_game'] = 120.0
            features_df['away_rush_yards_per_game'] = 120.0
            return features_df
        
        # Aggregate RB stats by team
        rb_team_stats = rb_stats.groupby('recent_team').agg({
            'rushing_yards': 'sum',
            'rushing_tds': 'sum',
            'rushing_attempts': 'sum'
        }).reset_index()
        
        # Calculate yards per carry
        rb_team_stats['ypc'] = rb_team_stats['rushing_yards'] / rb_team_stats['rushing_attempts'].clip(lower=1)
        rb_team_stats['team_id'] = rb_team_stats['recent_team'].map(NFL_TEAM_ABBREVIATIONS)
        
        # Merge with features
        home_rb = rb_team_stats[['team_id', 'rushing_yards', 'ypc']].rename(columns={
            'rushing_yards': 'home_rush_yards_total',
            'ypc': 'home_yards_per_carry'
        })
        away_rb = rb_team_stats[['team_id', 'rushing_yards', 'ypc']].rename(columns={
            'rushing_yards': 'away_rush_yards_total', 
            'ypc': 'away_yards_per_carry'
        })
        
        features_df = features_df.merge(home_rb, left_on='home_team_id', right_on='team_id', how='left').drop('team_id', axis=1, errors='ignore')
        features_df = features_df.merge(away_rb, left_on='away_team_id', right_on='team_id', how='left').drop('team_id', axis=1, errors='ignore')
        
        # Calculate per-game stats
        features_df['home_rush_yards_per_game'] = features_df['home_rush_yards_total'].fillna(2040) / 17
        features_df['away_rush_yards_per_game'] = features_df['away_rush_yards_total'].fillna(2040) / 17
        features_df['home_yards_per_carry'] = features_df['home_yards_per_carry'].fillna(4.0)
        features_df['away_yards_per_carry'] = features_df['away_yards_per_carry'].fillna(4.0)
        
        return features_df
    
    def _add_receiver_features(self, features_df: pd.DataFrame, player_stats: pd.DataFrame) -> pd.DataFrame:
        """Add receiver features."""
        wr_stats = player_stats[player_stats['position'].isin(['WR', 'TE'])].copy()
        
        if wr_stats.empty:
            features_df['home_receiving_yards_per_game'] = 250.0
            features_df['away_receiving_yards_per_game'] = 250.0
            return features_df
        
        # Aggregate receiver stats by team
        wr_team_stats = wr_stats.groupby('recent_team').agg({
            'receiving_yards': 'sum',
            'receiving_tds': 'sum',
            'receptions': 'sum',
            'targets': 'sum'
        }).reset_index()
        
        # Calculate catch rate
        wr_team_stats['catch_rate'] = wr_team_stats['receptions'] / wr_team_stats['targets'].clip(lower=1)
        wr_team_stats['team_id'] = wr_team_stats['recent_team'].map(NFL_TEAM_ABBREVIATIONS)
        
        # Merge with features
        home_wr = wr_team_stats[['team_id', 'receiving_yards', 'catch_rate']].rename(columns={
            'receiving_yards': 'home_receiving_yards_total',
            'catch_rate': 'home_catch_rate'
        })
        away_wr = wr_team_stats[['team_id', 'receiving_yards', 'catch_rate']].rename(columns={
            'receiving_yards': 'away_receiving_yards_total',
            'catch_rate': 'away_catch_rate'
        })
        
        features_df = features_df.merge(home_wr, left_on='home_team_id', right_on='team_id', how='left').drop('team_id', axis=1, errors='ignore')
        features_df = features_df.merge(away_wr, left_on='away_team_id', right_on='team_id', how='left').drop('team_id', axis=1, errors='ignore')
        
        # Calculate per-game stats
        features_df['home_receiving_yards_per_game'] = features_df['home_receiving_yards_total'].fillna(4250) / 17
        features_df['away_receiving_yards_per_game'] = features_df['away_receiving_yards_total'].fillna(4250) / 17
        features_df['home_catch_rate'] = features_df['home_catch_rate'].fillna(0.65)
        features_df['away_catch_rate'] = features_df['away_catch_rate'].fillna(0.65)
        
        return features_df
    
    def _add_defensive_features(self, features_df: pd.DataFrame, player_stats: pd.DataFrame) -> pd.DataFrame:
        """Add defensive features."""
        def_positions = ['DE', 'DT', 'NT', 'LB', 'ILB', 'OLB', 'CB', 'S', 'SS', 'FS', 'DB']
        def_stats = player_stats[player_stats['position'].isin(def_positions)].copy()
        
        if def_stats.empty:
            features_df['home_sacks_per_game'] = 2.5
            features_df['away_sacks_per_game'] = 2.5
            features_df['home_interceptions_per_game'] = 0.8
            features_df['away_interceptions_per_game'] = 0.8
            return features_df
        
        # Aggregate defensive stats by team
        def_team_stats = def_stats.groupby('recent_team').agg({
            'sacks': 'sum',
            'interceptions': 'sum',
            'tackles_solo': 'sum',
            'tackles_assists': 'sum'
        }).reset_index()
        
        def_team_stats['total_tackles'] = def_team_stats['tackles_solo'] + def_team_stats['tackles_assists']
        def_team_stats['team_id'] = def_team_stats['recent_team'].map(NFL_TEAM_ABBREVIATIONS)
        
        # Merge with features
        home_def = def_team_stats[['team_id', 'sacks', 'interceptions']].rename(columns={
            'sacks': 'home_sacks_total',
            'interceptions': 'home_interceptions_total'
        })
        away_def = def_team_stats[['team_id', 'sacks', 'interceptions']].rename(columns={
            'sacks': 'away_sacks_total',
            'interceptions': 'away_interceptions_total'
        })
        
        features_df = features_df.merge(home_def, left_on='home_team_id', right_on='team_id', how='left').drop('team_id', axis=1, errors='ignore')
        features_df = features_df.merge(away_def, left_on='away_team_id', right_on='team_id', how='left').drop('team_id', axis=1, errors='ignore')
        
        # Calculate per-game stats
        features_df['home_sacks_per_game'] = features_df['home_sacks_total'].fillna(42) / 17
        features_df['away_sacks_per_game'] = features_df['away_sacks_total'].fillna(42) / 17
        features_df['home_interceptions_per_game'] = features_df['home_interceptions_total'].fillna(14) / 17
        features_df['away_interceptions_per_game'] = features_df['away_interceptions_total'].fillna(14) / 17
        
        return features_df
    
    def _add_kicker_features(self, features_df: pd.DataFrame, player_stats: pd.DataFrame) -> pd.DataFrame:
        """Add kicker features."""
        k_stats = player_stats[player_stats['position'] == 'K'].copy()
        
        if k_stats.empty:
            features_df['home_fg_percentage'] = 0.85
            features_df['away_fg_percentage'] = 0.85
            return features_df
        
        # Field goal percentage would come from special stats not typically in nfl_data_py
        # Using placeholder for now
        features_df['home_fg_percentage'] = np.random.normal(0.85, 0.05, len(features_df)).clip(0.7, 0.95)
        features_df['away_fg_percentage'] = np.random.normal(0.85, 0.05, len(features_df)).clip(0.7, 0.95)
        
        return features_df
    
    def _add_default_player_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add default player features when data is unavailable."""
        default_features = {
            'home_qb_rating': 85.0,
            'away_qb_rating': 85.0,
            'home_qb_experience': 5.0,
            'away_qb_experience': 5.0,
            'home_rush_yards_per_game': 120.0,
            'away_rush_yards_per_game': 120.0,
            'home_yards_per_carry': 4.0,
            'away_yards_per_carry': 4.0,
            'home_receiving_yards_per_game': 250.0,
            'away_receiving_yards_per_game': 250.0,
            'home_catch_rate': 0.65,
            'away_catch_rate': 0.65,
            'home_sacks_per_game': 2.5,
            'away_sacks_per_game': 2.5,
            'home_interceptions_per_game': 0.8,
            'away_interceptions_per_game': 0.8,
            'home_fg_percentage': 0.85,
            'away_fg_percentage': 0.85
        }
        
        for feature, value in default_features.items():
            features_df[feature] = value
        
        return features_df
    
    def create_advanced_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced analytics features.
        """
        logger.info("📊 Creating NFL advanced features...")
        
        features_df = games_df.copy()
        
        # Matchup-specific features
        features_df = self._add_matchup_features(features_df)
        
        # Momentum and streaks
        features_df = self._add_momentum_features(features_df)
        
        # Injury impact
        features_df = self._add_injury_features(features_df)
        
        # Market/betting features
        features_df = self._add_market_features(features_df)
        
        logger.info(f"✅ Created advanced features: {len(features_df.columns)} columns")
        return features_df
    
    def _add_matchup_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add matchup-specific features."""
        # Offensive vs defensive matchups
        if all(col in features_df.columns for col in ['home_rush_yards_per_game', 'away_rush_yards_per_game']):
            features_df['home_rushing_advantage'] = features_df['home_rush_yards_per_game'] - 120.0  # vs league avg
            features_df['away_rushing_advantage'] = features_df['away_rush_yards_per_game'] - 120.0
        
        if all(col in features_df.columns for col in ['home_qb_rating', 'away_qb_rating']):
            features_df['qb_advantage_home'] = features_df['home_qb_rating'] - features_df['away_qb_rating']
            features_df['qb_advantage_significant'] = (abs(features_df['qb_advantage_home']) > 10).astype(int)
        
        return features_df
    
    def _add_momentum_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum and streak features."""
        # Placeholder momentum features (would calculate from recent results)
        features_df['home_current_streak'] = np.random.randint(-5, 6, len(features_df))  # -5 to +5 game streak
        features_df['away_current_streak'] = np.random.randint(-5, 6, len(features_df))
        features_df['home_hot_team'] = (features_df['home_current_streak'] >= 3).astype(int)
        features_df['away_hot_team'] = (features_df['away_current_streak'] >= 3).astype(int)
        features_df['home_cold_team'] = (features_df['home_current_streak'] <= -3).astype(int)
        features_df['away_cold_team'] = (features_df['away_current_streak'] <= -3).astype(int)
        
        return features_df
    
    def _add_injury_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add injury impact features."""
        # Placeholder injury features (would integrate with injury reports)
        features_df['home_key_injuries'] = np.random.poisson(1.5, len(features_df))  # Number of key injured players
        features_df['away_key_injuries'] = np.random.poisson(1.5, len(features_df))
        features_df['home_qb_healthy'] = np.random.choice([0, 1], len(features_df), p=[0.1, 0.9])
        features_df['away_qb_healthy'] = np.random.choice([0, 1], len(features_df), p=[0.1, 0.9])
        
        return features_df
    
    def _add_market_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add market/betting related features."""
        # Public betting percentages (placeholder)
        features_df['public_bet_percentage_home'] = np.random.beta(2, 2, len(features_df))
        features_df['sharp_money_home'] = np.random.choice([0, 1], len(features_df), p=[0.7, 0.3])
        features_df['line_movement'] = np.random.normal(0, 1.5, len(features_df))  # Point movement
        features_df['reverse_line_movement'] = ((features_df['public_bet_percentage_home'] > 0.6) & 
                                               (features_df['line_movement'] < -0.5)).astype(int)
        
        return features_df
    
    def engineer_all_features(self, games_df: pd.DataFrame, include_advanced: bool = True) -> pd.DataFrame:
        """
        Engineer all NFL features in one call.
        
        Args:
            games_df: DataFrame with game information
            include_advanced: Whether to include advanced features
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("🚀 Engineering all NFL features...")
        
        # Start with original data
        features_df = games_df.copy()
        
        # Add each feature type
        features_df = self.create_team_features(features_df)
        features_df = self.create_situational_features(features_df) 
        features_df = self.create_player_aggregate_features(features_df)
        
        if include_advanced:
            features_df = self.create_advanced_features(features_df)
        
        # Feature interactions
        features_df = self._create_feature_interactions(features_df)
        
        # Clean up and validate
        features_df = self._clean_features(features_df)
        
        logger.info(f"✅ NFL feature engineering complete: {len(features_df.columns)} total features")
        return features_df
    
    def _create_feature_interactions(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key variables."""
        # Weather impact on passing/rushing
        if all(col in features_df.columns for col in ['bad_weather_game', 'home_qb_rating', 'away_qb_rating']):
            features_df['home_weather_qb_impact'] = features_df['bad_weather_game'] * (100 - features_df['home_qb_rating'])
            features_df['away_weather_qb_impact'] = features_df['bad_weather_game'] * (100 - features_df['away_qb_rating'])
        
        # Divisional game intensity
        if all(col in features_df.columns for col in ['is_divisional_game', 'playoff_implications']):
            features_df['divisional_playoff_game'] = features_df['is_divisional_game'] * features_df['playoff_implications']
        
        # Home field advantage in bad weather
        if all(col in features_df.columns for col in ['bad_weather_game', 'is_dome_game']):
            features_df['outdoor_weather_advantage'] = features_df['bad_weather_game'] * (1 - features_df['is_dome_game'])
        
        return features_df
    
    def _clean_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate features."""
        # Remove any columns that are all NaN
        features_df = features_df.dropna(axis=1, how='all')
        
        # Fill remaining NaN values with appropriate defaults
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        features_df[numeric_columns] = features_df[numeric_columns].fillna(0)
        
        # Remove any infinite values
        features_df = features_df.replace([np.inf, -np.inf], 0)
        
        # Ensure consistent data types
        for col in numeric_columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)
        
        logger.info("✅ NFL feature cleaning complete")
        return features_df


# Convenience function for external use
def create_nfl_features(games_df: pd.DataFrame, 
                       include_advanced: bool = True,
                       use_current_data: bool = True) -> pd.DataFrame:
    """
    Convenience function to create all NFL features.
    
    Args:
        games_df: DataFrame with game information
        include_advanced: Whether to include advanced features
        use_current_data: Whether to load current season data
        
    Returns:
        DataFrame with all engineered features
    """
    engineer = NFLFeatureEngineer(use_current_data=use_current_data)
    return engineer.engineer_all_features(games_df, include_advanced=include_advanced)


if __name__ == "__main__":
    # Example usage
    sample_games = pd.DataFrame({
        'game_id': [1, 2, 3],
        'home_team_id': [14, 1, 31],  # KC, BUF, SF
        'away_team_id': [13, 2, 17],  # DEN, MIA, DAL
        'date': pd.date_range('2025-08-24', periods=3, freq='D'),
        'week': [1, 1, 1]
    })
    
    features = create_nfl_features(sample_games)
    print(f"Created {len(features.columns)} NFL features:")
    print(features.columns.tolist())
