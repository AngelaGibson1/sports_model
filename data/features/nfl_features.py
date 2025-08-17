# data/features/nba/nba_features.py

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger
import warnings

from config.settings import Settings
from data.database.nba import NBADatabase
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

class NBAFeatureEngineer:
    """
    Comprehensive NBA feature engineering for game prediction models.
    Creates advanced features from raw NBA data for machine learning models.
    """
    
    def __init__(self, db: Optional[NBADatabase] = None):
        """
        Initialize NBA feature engineer.
        
        Args:
            db: Optional NBA database instance
        """
        self.db = db or NBADatabase()
        self.nba_config = Settings.SPORT_CONFIGS['nba']
        self.rolling_windows = Settings.ROLLING_WINDOWS['nba']
        
        # NBA-specific feature configuration
        self.key_stats = [
            'points_per_game', 'points_allowed_per_game', 'field_goal_percentage',
            'three_point_percentage', 'free_throw_percentage', 'rebounds_per_game',
            'assists_per_game', 'steals_per_game', 'blocks_per_game', 
            'turnovers_per_game', 'offensive_rating', 'defensive_rating', 'pace'
        ]
        
        self.pace_adjustments = True
        self.logger = logger
        
        logger.info("🏀 NBA Feature Engineer initialized")
    
    def engineer_game_features(self, 
                              seasons: Optional[List[int]] = None,
                              include_advanced: bool = True,
                              include_situational: bool = True) -> pd.DataFrame:
        """
        Create comprehensive features for NBA game prediction.
        
        Args:
            seasons: Seasons to include in feature engineering
            include_advanced: Whether to include advanced metrics
            include_situational: Whether to include situational features
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("🔧 Starting NBA feature engineering...")
        
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
            # Add situational features
            features_df = self._add_situational_features(features_df)
        
        if include_advanced:
            # Add advanced metrics
            features_df = self._add_advanced_metrics(features_df)
        
        # Add feature interactions
        features_df = self._add_feature_interactions(features_df)
        
        # Handle missing values
        features_df = handle_missing_values(features_df, strategy='smart')
        
        # Final cleanup
        features_df = self._cleanup_features(features_df)
        
        logger.info(f"✅ Feature engineering complete: {features_df.shape[0]} games, {features_df.shape[1]} features")
        
        return features_df
    
    def _create_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create base features from raw game data."""
        logger.info("📊 Creating base features...")
        
        features_df = df.copy()
        
        # Basic game features
        features_df['total_points'] = features_df['home_score'] + features_df['away_score']
        features_df['point_differential'] = features_df['home_score'] - features_df['away_score']
        features_df['home_win'] = (features_df['home_score'] > features_df['away_score']).astype(int)
        
        # Date features
        features_df['date'] = pd.to_datetime(features_df['date'])
        features_df['month'] = features_df['date'].dt.month
        features_df['day_of_week'] = features_df['date'].dt.dayofweek
        features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)
        
        # Season features
        features_df['season_progress'] = self._calculate_season_progress(features_df)
        
        # Team strength differentials (if stats available)
        stat_columns = [col for col in self.key_stats if f'home_{col}' in features_df.columns]
        
        for stat in stat_columns:
            home_col = f'home_{stat}'
            away_col = f'away_{stat}'
            
            if home_col in features_df.columns and away_col in features_df.columns:
                features_df[f'{stat}_diff'] = features_df[home_col] - features_df[away_col]
                features_df[f'{stat}_ratio'] = features_df[home_col] / (features_df[away_col] + 0.001)
        
        return features_df
    
    def _add_rolling_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling performance features for both teams."""
        logger.info("📈 Adding rolling performance features...")
        
        features_df = df.copy()
        
        # Create team performance tracking
        home_performance = self._create_team_performance_tracking(features_df, 'home')
        away_performance = self._create_team_performance_tracking(features_df, 'away')
        
        # Add rolling statistics for multiple windows
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
        logger.info("🔥 Adding team form features...")
        
        features_df = df.copy()
        
        # Create form tracking for home and away teams
        for team_type in ['home', 'away']:
            team_id_col = f'{team_type}_team_id'
            
            # Recent form (last 5, 10, 20 games)
            for window in [5, 10, 20]:
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
            features_df[f'{team_type}_home_form_10'] = self._calculate_venue_form(
                features_df, team_id_col, venue='home', window=10
            )
            features_df[f'{team_type}_away_form_10'] = self._calculate_venue_form(
                features_df, team_id_col, venue='away', window=10
            )
        
        # Form differentials
        for window in [5, 10, 20]:
            features_df[f'form_diff_{window}'] = (
                features_df[f'home_form_{window}'] - features_df[f'away_form_{window}']
            )
        
        return features_df
    
    def _add_head_to_head_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add head-to-head historical features."""
        logger.info("⚔️ Adding head-to-head features...")
        
        features_df = df.copy()
        
        # Calculate H2H statistics for each game
        h2h_features = []
        
        for idx, row in features_df.iterrows():
            home_team = row['home_team_id']
            away_team = row['away_team_id']
            game_date = row['date']
            
            # Get historical H2H data before this game
            h2h_stats = self._get_historical_h2h(
                home_team, away_team, game_date, features_df
            )
            
            h2h_features.append(h2h_stats)
        
        # Convert to DataFrame and merge
        h2h_df = pd.DataFrame(h2h_features)
        features_df = pd.concat([features_df.reset_index(drop=True), h2h_df], axis=1)
        
        return features_df
    
    def _add_situational_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add situational features (rest, travel, etc.)."""
        logger.info("🎯 Adding situational features...")
        
        features_df = df.copy()
        
        # Calculate rest days for both teams
        features_df['home_rest_days'] = self._calculate_rest_days(
            features_df, 'home_team_id', 'home'
        )
        features_df['away_rest_days'] = self._calculate_rest_days(
            features_df, 'away_team_id', 'away'
        )
        
        # Rest advantage
        features_df['rest_advantage'] = (
            features_df['home_rest_days'] - features_df['away_rest_days']
        )
        
        # Back-to-back games
        features_df['home_back_to_back'] = (features_df['home_rest_days'] == 0).astype(int)
        features_df['away_back_to_back'] = (features_df['away_rest_days'] == 0).astype(int)
        
        # Both teams on back-to-back
        features_df['both_back_to_back'] = (
            features_df['home_back_to_back'] & features_df['away_back_to_back']
        ).astype(int)
        
        # Home court advantage features
        features_df['home_court_advantage'] = self._calculate_home_court_advantage(features_df)
        
        # Schedule difficulty (games in last X days)
        for days in [7, 14]:
            features_df[f'home_games_last_{days}d'] = self._count_recent_games(
                features_df, 'home_team_id', days
            )
            features_df[f'away_games_last_{days}d'] = self._count_recent_games(
                features_df, 'away_team_id', days
            )
        
        return features_df
    
    def _add_advanced_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced NBA metrics."""
        logger.info("🧮 Adding advanced metrics...")
        
        features_df = df.copy()
        
        # Pace adjustments for relevant stats
        if self.pace_adjustments:
            features_df = self._add_pace_adjusted_stats(features_df)
        
        # Pythagorean expectation
        features_df = self._add_pythagorean_expectation(features_df)
        
        # Strength of schedule
        features_df = self._add_strength_of_schedule(features_df)
        
        # Net rating differentials
        if 'home_offensive_rating' in features_df.columns:
            features_df['home_net_rating'] = (
                features_df['home_offensive_rating'] - features_df['home_defensive_rating']
            )
            features_df['away_net_rating'] = (
                features_df['away_offensive_rating'] - features_df['away_defensive_rating']
            )
            features_df['net_rating_diff'] = (
                features_df['home_net_rating'] - features_df['away_net_rating']
            )
        
        # Efficiency metrics
        features_df = self._add_efficiency_metrics(features_df)
        
        return features_df
    
    def _add_feature_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add feature interactions."""
        logger.info("🔗 Adding feature interactions...")
        
        features_df = df.copy()
        
        # Define important feature pairs for interactions
        interaction_pairs = [
            ('home_form_10', 'away_form_10'),
            ('home_offensive_rating', 'away_defensive_rating'),
            ('home_defensive_rating', 'away_offensive_rating'),
            ('home_pace', 'away_pace'),
            ('rest_advantage', 'home_court_advantage')
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
                'points_scored': row[score_col],
                'points_allowed': row[opp_score_col],
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
        rolling_cols = ['points_scored', 'points_allowed', 'win']
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
    
    def _get_historical_h2h(self, home_team: int, away_team: int, 
                          game_date: datetime, df: pd.DataFrame) -> Dict[str, float]:
        """Get historical head-to-head statistics."""
        # Find all previous games between these teams
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
                'h2h_avg_total': 200,  # NBA average
                'h2h_avg_margin': 0
            }
        
        # Calculate H2H statistics
        home_wins = 0
        total_points = []
        margins = []
        
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
            
            total_points.append(game['home_score'] + game['away_score'])
            margins.append(margin)
        
        return {
            'h2h_games_played': len(h2h_games),
            'h2h_home_wins': home_wins,
            'h2h_home_win_pct': home_wins / len(h2h_games),
            'h2h_avg_total': np.mean(total_points),
            'h2h_avg_margin': np.mean(margins)
        }
    
    def _calculate_rest_days(self, df: pd.DataFrame, team_id_col: str, 
                           team_type: str) -> pd.Series:
        """Calculate rest days since last game."""
        rest_days = []
        
        for idx, row in df.iterrows():
            team_id = row[team_id_col]
            game_date = row['date']
            
            # Find last game for this team
            prev_mask = (
                (df['home_team_id'] == team_id) | (df['away_team_id'] == team_id)
            ) & (df['date'] < game_date)
            
            prev_games = df[prev_mask]
            
            if len(prev_games) == 0:
                rest_days.append(3)  # Default rest for first game
            else:
                last_game_date = prev_games['date'].max()
                days_diff = (game_date - last_game_date).days
                rest_days.append(max(0, days_diff - 1))  # Subtract 1 for game day
        
        return pd.Series(rest_days, index=df.index)
    
    def _count_recent_games(self, df: pd.DataFrame, team_id_col: str, days: int) -> pd.Series:
        """Count games played in last N days."""
        game_counts = []
        
        for idx, row in df.iterrows():
            team_id = row[team_id_col]
            game_date = row['date']
            
            # Count games in last N days
            recent_mask = (
                (df['home_team_id'] == team_id) | (df['away_team_id'] == team_id)
            ) & (df['date'] >= game_date - timedelta(days=days)) & (df['date'] < game_date)
            
            game_counts.append(len(df[recent_mask]))
        
        return pd.Series(game_counts, index=df.index)
    
    def _calculate_home_court_advantage(self, df: pd.DataFrame) -> pd.Series:
        """Calculate dynamic home court advantage."""
        # This could be made more sophisticated with venue-specific data
        return pd.Series([3.0] * len(df), index=df.index)  # NBA average HCA
    
    def _add_pace_adjusted_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pace-adjusted statistics."""
        features_df = df.copy()
        
        # League average pace (could be calculated dynamically)
        league_avg_pace = 100.0
        
        pace_stats = ['points_per_game', 'assists_per_game', 'rebounds_per_game']
        
        for team_type in ['home', 'away']:
            pace_col = f'{team_type}_pace'
            
            if pace_col in features_df.columns:
                for stat in pace_stats:
                    stat_col = f'{team_type}_{stat}'
                    if stat_col in features_df.columns:
                        adj_col = f'{team_type}_{stat}_pace_adj'
                        features_df[adj_col] = (
                            features_df[stat_col] * league_avg_pace / 
                            (features_df[pace_col] + 0.001)
                        )
        
        return features_df
    
    def _add_pythagorean_expectation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Pythagorean expectation for wins."""
        features_df = df.copy()
        
        for team_type in ['home', 'away']:
            ppg_col = f'{team_type}_points_per_game'
            papg_col = f'{team_type}_points_allowed_per_game'
            
            if ppg_col in features_df.columns and papg_col in features_df.columns:
                pythag_col = f'{team_type}_pythagorean_wins'
                
                # NBA Pythagorean exponent is typically around 14
                exponent = 14.0
                features_df[pythag_col] = (
                    features_df[ppg_col] ** exponent /
                    (features_df[ppg_col] ** exponent + features_df[papg_col] ** exponent)
                )
        
        # Pythagorean differential
        if 'home_pythagorean_wins' in features_df.columns and 'away_pythagorean_wins' in features_df.columns:
            features_df['pythagorean_diff'] = (
                features_df['home_pythagorean_wins'] - features_df['away_pythagorean_wins']
            )
        
        return features_df
    
    def _add_strength_of_schedule(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add strength of schedule metrics."""
        features_df = df.copy()
        
        # This is a simplified SOS - could be made more sophisticated
        for team_type in ['home', 'away']:
            sos_values = []
            team_id_col = f'{team_type}_team_id'
            
            for idx, row in features_df.iterrows():
                team_id = row[team_id_col]
                game_date = row['date']
                
                # Get recent opponents
                recent_mask = (
                    (df['home_team_id'] == team_id) | (df['away_team_id'] == team_id)
                ) & (df['date'] < game_date)
                
                recent_games = df[recent_mask].tail(10)  # Last 10 games
                
                if len(recent_games) == 0:
                    sos_values.append(0.5)  # Neutral SOS
                    continue
                
                opp_win_pcts = []
                for _, game in recent_games.iterrows():
                    # Get opponent ID
                    if game['home_team_id'] == team_id:
                        opp_id = game['away_team_id']
                    else:
                        opp_id = game['home_team_id']
                    
                    # Calculate opponent's win percentage
                    opp_mask = (
                        (df['home_team_id'] == opp_id) | (df['away_team_id'] == opp_id)
                    ) & (df['date'] < game_date)
                    
                    opp_games = df[opp_mask]
                    if len(opp_games) > 0:
                        opp_wins = 0
                        for _, opp_game in opp_games.iterrows():
                            if opp_game['home_team_id'] == opp_id:
                                opp_wins += 1 if opp_game['home_score'] > opp_game['away_score'] else 0
                            else:
                                opp_wins += 1 if opp_game['away_score'] > opp_game['home_score'] else 0
                        
                        opp_win_pcts.append(opp_wins / len(opp_games))
                
                sos_values.append(np.mean(opp_win_pcts) if opp_win_pcts else 0.5)
            
            features_df[f'{team_type}_sos'] = sos_values
        
        # SOS differential
        if 'home_sos' in features_df.columns and 'away_sos' in features_df.columns:
            features_df['sos_diff'] = features_df['home_sos'] - features_df['away_sos']
        
        return features_df
    
    def _add_efficiency_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add efficiency metrics."""
        features_df = df.copy()
        
        for team_type in ['home', 'away']:
            # True shooting percentage
            ppg_col = f'{team_type}_points_per_game'
            fga_col = f'{team_type}_field_goals_attempted'  # If available
            fta_col = f'{team_type}_free_throws_attempted'  # If available
            
            # Effective field goal percentage (if 3PT data available)
            fg_pct_col = f'{team_type}_field_goal_percentage'
            three_pct_col = f'{team_type}_three_point_percentage'
            
            if fg_pct_col in features_df.columns and three_pct_col in features_df.columns:
                # Simplified eFG% calculation
                efg_col = f'{team_type}_efg_pct'
                features_df[efg_col] = (
                    features_df[fg_pct_col] + 0.5 * features_df[three_pct_col]
                )
        
        return features_df
    
    def _calculate_season_progress(self, df: pd.DataFrame) -> pd.Series:
        """Calculate season progress (0-1 scale)."""
        season_progress = []
        
        for _, row in df.iterrows():
            season = row['season']
            game_date = row['date']
            
            # NBA season typically runs October to April
            if season:
                season_start = datetime(season, 10, 1)  # October 1st
                season_end = datetime(season + 1, 4, 30)  # April 30th next year
                
                total_days = (season_end - season_start).days
                days_elapsed = (game_date - season_start).days
                
                progress = max(0, min(1, days_elapsed / total_days))
            else:
                progress = 0.5  # Default for missing season
            
            season_progress.append(progress)
        
        return pd.Series(season_progress, index=df.index)
    
    def _cleanup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final cleanup of features."""
        features_df = df.copy()
        
        # Remove obviously non-feature columns
        non_feature_cols = [
            'home_team_full_name', 'away_team_full_name', 'venue', 'city',
            'status', 'created_at', 'updated_at'
        ]
        
        features_df = features_df.drop(
            [col for col in non_feature_cols if col in features_df.columns], 
            axis=1
        )
        
        # Handle infinite values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        
        # Cap extreme values (outlier handling)
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in features_df.columns:
                q99 = features_df[col].quantile(0.99)
                q01 = features_df[col].quantile(0.01)
                features_df[col] = features_df[col].clip(lower=q01, upper=q99)
        
        return features_df
    
    def get_feature_importance_mapping(self) -> Dict[str, str]:
        """Get human-readable descriptions for features."""
        return {
            'home_win': 'Target: Home team won',
            'total_points': 'Total points scored in game',
            'point_differential': 'Home score minus away score',
            'home_form_10': 'Home team win rate last 10 games',
            'away_form_10': 'Away team win rate last 10 games',
            'form_diff_10': 'Difference in 10-game form',
            'h2h_home_win_pct': 'Home team win rate in head-to-head',
            'rest_advantage': 'Rest days advantage (home - away)',
            'home_court_advantage': 'Estimated home court advantage',
            'net_rating_diff': 'Net rating differential',
            'pythagorean_diff': 'Pythagorean win expectation diff',
            'sos_diff': 'Strength of schedule differential'
        }
    
    def create_prediction_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for upcoming games (prediction mode).
        
        Args:
            games_df: DataFrame with upcoming games
            
        Returns:
            DataFrame with features for prediction
        """
        logger.info("🔮 Creating prediction features for upcoming games...")
        
        if games_df.empty:
            return pd.DataFrame()
        
        # Get historical context for feature creation
        historical_df = self.db.get_historical_data()
        
        if historical_df.empty:
            logger.warning("No historical data available for prediction features")
            return games_df
        
        # Combine with historical data for context
        combined_df = pd.concat([historical_df, games_df], ignore_index=True)
        combined_df = combined_df.sort_values(['date', 'game_id'])
        
        # Engineer features on combined data
        features_df = self.engineer_game_features(include_advanced=True, include_situational=True)
        
        # Return only the prediction games
        prediction_mask = features_df['game_id'].isin(games_df['game_id'])
        prediction_features = features_df[prediction_mask].copy()
        
        logger.info(f"✅ Created prediction features: {len(prediction_features)} games")
        
        return prediction_features
