# utils/nfl_data_helpers.py
"""
NFL-specific data helper functions and utilities.
Provides specialized functions for NFL data processing, validation, and analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from loguru import logger
import re

# NFL-specific constants
NFL_POSITIONS = {
    'offense': ['QB', 'RB', 'FB', 'WR', 'TE', 'C', 'G', 'T', 'OL'],
    'defense': ['DE', 'DT', 'NT', 'LB', 'ILB', 'OLB', 'CB', 'S', 'SS', 'FS', 'DB'],
    'special_teams': ['K', 'P', 'LS']
}

NFL_TEAM_MAPPINGS = {
    'arizona cardinals': {'id': 29, 'abbr': 'ARI', 'conference': 'NFC', 'division': 'West'},
    'atlanta falcons': {'id': 25, 'abbr': 'ATL', 'conference': 'NFC', 'division': 'South'},
    'baltimore ravens': {'id': 5, 'abbr': 'BAL', 'conference': 'AFC', 'division': 'North'},
    'buffalo bills': {'id': 1, 'abbr': 'BUF', 'conference': 'AFC', 'division': 'East'},
    'carolina panthers': {'id': 26, 'abbr': 'CAR', 'conference': 'NFC', 'division': 'South'},
    'chicago bears': {'id': 21, 'abbr': 'CHI', 'conference': 'NFC', 'division': 'North'},
    'cincinnati bengals': {'id': 6, 'abbr': 'CIN', 'conference': 'AFC', 'division': 'North'},
    'cleveland browns': {'id': 7, 'abbr': 'CLE', 'conference': 'AFC', 'division': 'North'},
    'dallas cowboys': {'id': 17, 'abbr': 'DAL', 'conference': 'NFC', 'division': 'East'},
    'denver broncos': {'id': 13, 'abbr': 'DEN', 'conference': 'AFC', 'division': 'West'},
    'detroit lions': {'id': 22, 'abbr': 'DET', 'conference': 'NFC', 'division': 'North'},
    'green bay packers': {'id': 23, 'abbr': 'GB', 'conference': 'NFC', 'division': 'North'},
    'houston texans': {'id': 9, 'abbr': 'HOU', 'conference': 'AFC', 'division': 'South'},
    'indianapolis colts': {'id': 10, 'abbr': 'IND', 'conference': 'AFC', 'division': 'South'},
    'jacksonville jaguars': {'id': 11, 'abbr': 'JAX', 'conference': 'AFC', 'division': 'South'},
    'kansas city chiefs': {'id': 14, 'abbr': 'KC', 'conference': 'AFC', 'division': 'West'},
    'las vegas raiders': {'id': 15, 'abbr': 'LV', 'conference': 'AFC', 'division': 'West'},
    'los angeles chargers': {'id': 16, 'abbr': 'LAC', 'conference': 'AFC', 'division': 'West'},
    'los angeles rams': {'id': 30, 'abbr': 'LAR', 'conference': 'NFC', 'division': 'West'},
    'miami dolphins': {'id': 2, 'abbr': 'MIA', 'conference': 'AFC', 'division': 'East'},
    'minnesota vikings': {'id': 24, 'abbr': 'MIN', 'conference': 'NFC', 'division': 'North'},
    'new england patriots': {'id': 3, 'abbr': 'NE', 'conference': 'AFC', 'division': 'East'},
    'new orleans saints': {'id': 27, 'abbr': 'NO', 'conference': 'NFC', 'division': 'South'},
    'new york giants': {'id': 18, 'abbr': 'NYG', 'conference': 'NFC', 'division': 'East'},
    'new york jets': {'id': 4, 'abbr': 'NYJ', 'conference': 'AFC', 'division': 'East'},
    'philadelphia eagles': {'id': 19, 'abbr': 'PHI', 'conference': 'NFC', 'division': 'East'},
    'pittsburgh steelers': {'id': 8, 'abbr': 'PIT', 'conference': 'AFC', 'division': 'North'},
    'san francisco 49ers': {'id': 31, 'abbr': 'SF', 'conference': 'NFC', 'division': 'West'},
    'seattle seahawks': {'id': 32, 'abbr': 'SEA', 'conference': 'NFC', 'division': 'West'},
    'tampa bay buccaneers': {'id': 28, 'abbr': 'TB', 'conference': 'NFC', 'division': 'South'},
    'tennessee titans': {'id': 12, 'abbr': 'TEN', 'conference': 'AFC', 'division': 'South'},
    'washington commanders': {'id': 20, 'abbr': 'WAS', 'conference': 'NFC', 'division': 'East'}
}

NFL_SEASON_WEEKS = {
    'preseason': list(range(0, 4)),      # Weeks 0-3
    'regular_season': list(range(1, 19)), # Weeks 1-18
    'playoffs': list(range(19, 23)),      # Weeks 19-22 (Wild Card, Divisional, Conference, Super Bowl)
}


def normalize_nfl_team_name(team_name: str, return_format: str = 'id') -> Optional[Union[int, str, Dict]]:
    """
    Normalize NFL team names to consistent format.
    
    Args:
        team_name: Team name to normalize
        return_format: Format to return ('id', 'abbr', 'full_info')
        
    Returns:
        Normalized team identifier or None if not found
    """
    if not team_name or pd.isna(team_name):
        return None
    
    # Clean the input
    clean_name = str(team_name).lower().strip()
    
    # Remove common prefixes/suffixes
    clean_name = re.sub(r'\b(the|football|team|fc|nfl)\b', '', clean_name).strip()
    
    # Handle common abbreviations and variations
    name_variations = {
        'arizona': 'arizona cardinals',
        'cards': 'arizona cardinals',
        'atlanta': 'atlanta falcons',
        'baltimore': 'baltimore ravens',
        'buffalo': 'buffalo bills',
        'carolina': 'carolina panthers',
        'chicago': 'chicago bears',
        'bears': 'chicago bears',
        'cincinnati': 'cincinnati bengals',
        'bengals': 'cincinnati bengals',
        'cleveland': 'cleveland browns',
        'browns': 'cleveland browns',
        'dallas': 'dallas cowboys',
        'cowboys': 'dallas cowboys',
        'denver': 'denver broncos',
        'broncos': 'denver broncos',
        'detroit': 'detroit lions',
        'lions': 'detroit lions',
        'green bay': 'green bay packers',
        'packers': 'green bay packers',
        'houston': 'houston texans',
        'texans': 'houston texans',
        'indianapolis': 'indianapolis colts',
        'colts': 'indianapolis colts',
        'jacksonville': 'jacksonville jaguars',
        'jaguars': 'jacksonville jaguars',
        'kansas city': 'kansas city chiefs',
        'chiefs': 'kansas city chiefs',
        'las vegas': 'las vegas raiders',
        'raiders': 'las vegas raiders',
        'oakland raiders': 'las vegas raiders',
        'los angeles chargers': 'los angeles chargers',
        'chargers': 'los angeles chargers',
        'san diego chargers': 'los angeles chargers',
        'los angeles rams': 'los angeles rams',
        'rams': 'los angeles rams',
        'st louis rams': 'los angeles rams',
        'miami': 'miami dolphins',
        'dolphins': 'miami dolphins',
        'minnesota': 'minnesota vikings',
        'vikings': 'minnesota vikings',
        'new england': 'new england patriots',
        'patriots': 'new england patriots',
        'new orleans': 'new orleans saints',
        'saints': 'new orleans saints',
        'new york giants': 'new york giants',
        'giants': 'new york giants',
        'new york jets': 'new york jets',
        'jets': 'new york jets',
        'philadelphia': 'philadelphia eagles',
        'eagles': 'philadelphia eagles',
        'pittsburgh': 'pittsburgh steelers',
        'steelers': 'pittsburgh steelers',
        'san francisco': 'san francisco 49ers',
        '49ers': 'san francisco 49ers',
        'niners': 'san francisco 49ers',
        'seattle': 'seattle seahawks',
        'seahawks': 'seattle seahawks',
        'tampa bay': 'tampa bay buccaneers',
        'buccaneers': 'tampa bay buccaneers',
        'bucs': 'tampa bay buccaneers',
        'tennessee': 'tennessee titans',
        'titans': 'tennessee titans',
        'washington': 'washington commanders',
        'commanders': 'washington commanders',
        'washington redskins': 'washington commanders',
        'washington football team': 'washington commanders'
    }
    
    # Check variations first
    if clean_name in name_variations:
        clean_name = name_variations[clean_name]
    
    # Check against official names
    if clean_name in NFL_TEAM_MAPPINGS:
        team_info = NFL_TEAM_MAPPINGS[clean_name]
        
        if return_format == 'id':
            return team_info['id']
        elif return_format == 'abbr':
            return team_info['abbr']
        elif return_format == 'full_info':
            return team_info
        else:
            return team_info['id']
    
    # Check abbreviations
    for team, info in NFL_TEAM_MAPPINGS.items():
        if clean_name.upper() == info['abbr']:
            if return_format == 'id':
                return info['id']
            elif return_format == 'abbr':
                return info['abbr']
            elif return_format == 'full_info':
                return info
            else:
                return info['id']
    
    logger.warning(f"Could not normalize NFL team name: {team_name}")
    return None


def calculate_nfl_rolling_averages(df: pd.DataFrame, 
                                  stat_columns: List[str], 
                                  window: int = 4,
                                  team_id_col: str = 'team_id',
                                  date_col: str = 'date') -> pd.DataFrame:
    """
    Calculate rolling averages for NFL team statistics.
    
    Args:
        df: DataFrame with team stats
        stat_columns: Columns to calculate rolling averages for
        window: Rolling window size (default 4 games)
        team_id_col: Column name for team ID
        date_col: Column name for date
        
    Returns:
        DataFrame with rolling average columns added
    """
    df_with_rolling = df.copy()
    df_with_rolling[date_col] = pd.to_datetime(df_with_rolling[date_col])
    df_with_rolling = df_with_rolling.sort_values([team_id_col, date_col])
    
    for stat in stat_columns:
        if stat in df_with_rolling.columns:
            rolling_col = f"{stat}_rolling_{window}"
            df_with_rolling[rolling_col] = (
                df_with_rolling.groupby(team_id_col)[stat]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
    
    return df_with_rolling


def calculate_nfl_strength_of_schedule(games_df: pd.DataFrame,
                                     opponent_records: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Calculate strength of schedule for NFL teams.
    
    Args:
        games_df: DataFrame with game information
        opponent_records: DataFrame with team win percentages (optional)
        
    Returns:
        DataFrame with strength of schedule metrics
    """
    if opponent_records is None:
        # Create dummy opponent records if not provided
        opponent_records = pd.DataFrame({
            'team_id': range(1, 33),
            'win_percentage': np.random.uniform(0.3, 0.7, 32)
        })
    
    sos_results = []
    
    for team_id in range(1, 33):
        # Get games for this team
        home_games = games_df[games_df['home_team_id'] == team_id]
        away_games = games_df[games_df['away_team_id'] == team_id]
        
        # Combine opponents
        home_opponents = home_games['away_team_id'].tolist()
        away_opponents = away_games['home_team_id'].tolist()
        all_opponents = home_opponents + away_opponents
        
        if all_opponents:
            # Calculate average opponent win percentage
            opponent_win_pcts = opponent_records[
                opponent_records['team_id'].isin(all_opponents)
            ]['win_percentage']
            
            avg_opponent_win_pct = opponent_win_pcts.mean() if not opponent_win_pcts.empty else 0.5
        else:
            avg_opponent_win_pct = 0.5
        
        sos_results.append({
            'team_id': team_id,
            'strength_of_schedule': avg_opponent_win_pct,
            'num_opponents': len(set(all_opponents))
        })
    
    return pd.DataFrame(sos_results)


def validate_nfl_player_positions(positions: pd.Series) -> pd.Series:
    """
    Validate and standardize NFL player positions.
    
    Args:
        positions: Series of position strings
        
    Returns:
        Series with standardized positions
    """
    def standardize_position(pos):
        if pd.isna(pos):
            return 'UNKNOWN'
        
        pos = str(pos).upper().strip()
        
        # Handle common variations
        position_map = {
            'QUARTERBACK': 'QB',
            'RUNNINGBACK': 'RB', 'RUNNING BACK': 'RB', 'HALFBACK': 'RB', 'HB': 'RB',
            'FULLBACK': 'FB', 'FULL BACK': 'FB',
            'WIDE RECEIVER': 'WR', 'WIDERECEIVER': 'WR', 'RECEIVER': 'WR',
            'TIGHT END': 'TE', 'TIGHTEND': 'TE',
            'CENTER': 'C',
            'GUARD': 'G', 'LEFT GUARD': 'G', 'RIGHT GUARD': 'G', 'LG': 'G', 'RG': 'G',
            'TACKLE': 'T', 'LEFT TACKLE': 'T', 'RIGHT TACKLE': 'T', 'LT': 'T', 'RT': 'T',
            'OFFENSIVE LINE': 'OL', 'OFFENSIVE LINEMAN': 'OL',
            'DEFENSIVE END': 'DE', 'DEFENSIVEEND': 'DE',
            'DEFENSIVE TACKLE': 'DT', 'DEFENSIVETACKLE': 'DT',
            'NOSE TACKLE': 'NT', 'NOSETACKLE': 'NT', 'NOSE GUARD': 'NT',
            'LINEBACKER': 'LB', 'INSIDE LINEBACKER': 'ILB', 'OUTSIDE LINEBACKER': 'OLB',
            'MIDDLE LINEBACKER': 'LB', 'MLB': 'LB',
            'CORNERBACK': 'CB', 'CORNER BACK': 'CB', 'CORNER': 'CB',
            'SAFETY': 'S', 'FREE SAFETY': 'FS', 'STRONG SAFETY': 'SS',
            'DEFENSIVE BACK': 'DB', 'DEFENSIVEBACK': 'DB',
            'KICKER': 'K', 'PLACE KICKER': 'K', 'PLACEKICKER': 'K',
            'PUNTER': 'P',
            'LONG SNAPPER': 'LS', 'LONGSNAPPER': 'LS'
        }
        
        return position_map.get(pos, pos)
    
    return positions.apply(standardize_position)


def calculate_nfl_weather_impact_score(temperature: Optional[float] = None,
                                     wind_speed: Optional[float] = None,
                                     precipitation: Optional[bool] = None,
                                     is_dome: Optional[bool] = None) -> float:
    """
    Calculate weather impact score for NFL games (0 = no impact, 1 = severe impact).
    
    Args:
        temperature: Temperature in Fahrenheit
        wind_speed: Wind speed in MPH
        precipitation: Whether there's precipitation
        is_dome: Whether game is in a dome
        
    Returns:
        Weather impact score (0-1)
    """
    if is_dome:
        return 0.0  # No weather impact in domes
    
    impact_score = 0.0
    
    # Temperature impact
    if temperature is not None:
        if temperature < 20:  # Extremely cold
            impact_score += 0.4
        elif temperature < 32:  # Below freezing
            impact_score += 0.3
        elif temperature < 40:  # Cold
            impact_score += 0.1
        elif temperature > 90:  # Very hot
            impact_score += 0.2
        elif temperature > 80:  # Hot
            impact_score += 0.1
    
    # Wind impact
    if wind_speed is not None:
        if wind_speed > 25:  # Severe winds
            impact_score += 0.3
        elif wind_speed > 15:  # Strong winds
            impact_score += 0.2
        elif wind_speed > 10:  # Moderate winds
            impact_score += 0.1
    
    # Precipitation impact
    if precipitation:
        impact_score += 0.2
    
    return min(impact_score, 1.0)  # Cap at 1.0


def determine_nfl_game_importance(week: int, 
                                home_record: Optional[Tuple[int, int]] = None,
                                away_record: Optional[Tuple[int, int]] = None,
                                is_divisional: bool = False,
                                playoff_implications: bool = False) -> Dict[str, Any]:
    """
    Determine NFL game importance score and factors.
    
    Args:
        week: Week of the season (1-22)
        home_record: Home team record as (wins, losses)
        away_record: Away team record as (wins, losses) 
        is_divisional: Whether it's a divisional game
        playoff_implications: Whether game has playoff implications
        
    Returns:
        Dictionary with importance score and factors
    """
    importance_score = 0.0
    factors = []
    
    # Week-based importance
    if week <= 4:
        week_factor = 0.1  # Early season
        factors.append("early_season")
    elif week <= 12:
        week_factor = 0.3  # Mid season
        factors.append("mid_season")
    elif week <= 18:
        week_factor = 0.5  # Late regular season
        factors.append("late_season")
    else:
        week_factor = 1.0  # Playoffs
        factors.append("playoffs")
    
    importance_score += week_factor
    
    # Record-based importance
    if home_record and away_record:
        home_wins, home_losses = home_record
        away_wins, away_losses = away_record
        
        home_win_pct = home_wins / (home_wins + home_losses) if (home_wins + home_losses) > 0 else 0.5
        away_win_pct = away_wins / (away_wins + away_losses) if (away_wins + away_losses) > 0 else 0.5
        
        # Close records make game more important
        record_diff = abs(home_win_pct - away_win_pct)
        if record_diff < 0.2:
            importance_score += 0.2
            factors.append("close_records")
        
        # Both teams being good makes it more important
        avg_win_pct = (home_win_pct + away_win_pct) / 2
        if avg_win_pct > 0.6:
            importance_score += 0.2
            factors.append("good_teams")
    
    # Divisional games are more important
    if is_divisional:
        importance_score += 0.3
        factors.append("divisional_rivalry")
    
    # Playoff implications
    if playoff_implications:
        importance_score += 0.4
        factors.append("playoff_implications")
    
    # Cap at 1.0
    importance_score = min(importance_score, 1.0)
    
    # Determine importance level
    if importance_score >= 0.8:
        importance_level = "critical"
    elif importance_score >= 0.6:
        importance_level = "high"
    elif importance_score >= 0.4:
        importance_level = "moderate"
    else:
        importance_level = "low"
    
    return {
        'importance_score': importance_score,
        'importance_level': importance_level,
        'factors': factors,
        'week_factor': week_factor
    }


def calculate_nfl_home_field_advantage(venue: Optional[str] = None,
                                     temperature: Optional[float] = None,
                                     is_dome: bool = False,
                                     crowd_noise_level: Optional[str] = None) -> float:
    """
    Calculate home field advantage factor for NFL teams.
    
    Args:
        venue: Stadium name
        temperature: Temperature (affects outdoor games more)
        is_dome: Whether stadium is a dome
        crowd_noise_level: Estimated crowd noise ('low', 'medium', 'high')
        
    Returns:
        Home field advantage factor (typically 0.5-1.5)
    """
    base_hfa = 1.0  # Baseline home field advantage
    
    # Venue-specific adjustments (some stadiums are known for strong HFA)
    venue_adjustments = {
        'arrowhead stadium': 0.3,  # Kansas City
        'centurylink field': 0.25,  # Seattle (loud)
        'lambeau field': 0.2,      # Green Bay (cold weather)
        'gillette stadium': 0.15,   # New England
        'heinz field': 0.15,       # Pittsburgh
        'soldier field': 0.1,      # Chicago (cold weather)
    }
    
    if venue and venue.lower() in venue_adjustments:
        base_hfa += venue_adjustments[venue.lower()]
    
    # Weather advantage for outdoor stadiums
    if not is_dome and temperature is not None:
        if temperature < 32:  # Cold weather advantage
            base_hfa += 0.1
        elif temperature < 20:  # Extreme cold
            base_hfa += 0.2
    
    # Crowd noise advantage
    noise_adjustments = {
        'high': 0.15,
        'medium': 0.05,
        'low': -0.05
    }
    
    if crowd_noise_level and crowd_noise_level.lower() in noise_adjustments:
        base_hfa += noise_adjustments[crowd_noise_level.lower()]
    
    # Dome advantage (controlled conditions)
    if is_dome:
        base_hfa += 0.05
    
    return max(base_hfa, 0.3)  # Minimum some advantage


def handle_nfl_missing_values(df: pd.DataFrame, strategy: str = 'nfl_defaults') -> pd.DataFrame:
    """
    Handle missing values in NFL datasets using sport-specific strategies.
    
    Args:
        df: DataFrame with potential missing values
        strategy: Strategy to use ('nfl_defaults', 'forward_fill', 'interpolate')
        
    Returns:
        DataFrame with missing values handled
    """
    df_cleaned = df.copy()
    
    if strategy == 'nfl_defaults':
        # NFL-specific default values
        nfl_defaults = {
            'points_per_game': 22.5,
            'yards_per_game': 350.0,
            'passing_yards': 250.0,
            'rushing_yards': 120.0,
            'turnovers': 1.2,
            'sacks': 2.5,
            'field_goal_percentage': 0.85,
            'win_percentage': 0.5,
            'temperature': 65.0,
            'wind_speed': 5.0
        }
        
        for col in df_cleaned.columns:
            if col in nfl_defaults and df_cleaned[col].isna().any():
                df_cleaned[col].fillna(nfl_defaults[col], inplace=True)
                logger.info(f"Filled {col} missing values with NFL default: {nfl_defaults[col]}")
        
        # Fill remaining numeric columns with 0
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(0)
        
        # Fill text columns with 'UNKNOWN'
        text_cols = df_cleaned.select_dtypes(include=['object']).columns
        df_cleaned[text_cols] = df_cleaned[text_cols].fillna('UNKNOWN')
    
    elif strategy == 'forward_fill':
        df_cleaned = df_cleaned.fillna(method='ffill')
    
    elif strategy == 'interpolate':
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        df_cleaned[numeric_cols] = df_cleaned[numeric_cols].interpolate()
    
    return df_cleaned


def validate_nfl_data_quality(df: pd.DataFrame, 
                            data_type: str = 'games') -> Dict[str, Any]:
    """
    Validate NFL data quality and return quality metrics.
    
    Args:
        df: DataFrame to validate
        data_type: Type of data ('games', 'players', 'teams')
        
    Returns:
        Dictionary with quality metrics and recommendations
    """
    if df.empty:
        return {
            'quality_score': 0.0,
            'issues': ['Dataset is empty'],
            'recommendations': ['Acquire data before proceeding']
        }
    
    quality_metrics = {
        'completeness': 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns))),
        'uniqueness': len(df.drop_duplicates()) / len(df),
        'consistency': 1.0,  # Placeholder
        'validity': 1.0      # Placeholder
    }
    
    issues = []
    recommendations = []
    
    # Check completeness
    if quality_metrics['completeness'] < 0.9:
        issues.append(f"High missing data rate: {(1-quality_metrics['completeness'])*100:.1f}%")
        recommendations.append("Review data ingestion process")
    
    # Check for duplicates
    if quality_metrics['uniqueness'] < 0.95:
        issues.append(f"Duplicate records detected: {(1-quality_metrics['uniqueness'])*100:.1f}%")
        recommendations.append("Implement deduplication logic")
    
    # Data type specific validations
    if data_type == 'games':
        required_cols = ['home_team_id', 'away_team_id', 'date']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
            recommendations.append("Ensure all required game columns are present")
        
        # Check for reasonable score values
        if 'home_score' in df.columns:
            invalid_scores = df[(df['home_score'] < 0) | (df['home_score'] > 60)]
            if not invalid_scores.empty:
                issues.append(f"{len(invalid_scores)} games with unrealistic scores")
                recommendations.append("Review score data for outliers")
    
    elif data_type == 'players':
        if 'position' in df.columns:
            unknown_positions = df[~df['position'].isin(
                NFL_POSITIONS['offense'] + NFL_POSITIONS['defense'] + NFL_POSITIONS['special_teams']
            )]
            if not unknown_positions.empty:
                issues.append(f"{len(unknown_positions)} players with unknown positions")
                recommendations.append("Standardize position names")
    
    # Calculate overall quality score
    overall_quality = np.mean(list(quality_metrics.values()))
    
    return {
        'quality_score': overall_quality,
        'metrics': quality_metrics,
        'issues': issues,
        'recommendations': recommendations,
        'record_count': len(df),
        'column_count': len(df.columns)
    }


# Convenience function for common NFL data processing
def process_nfl_dataset(df: pd.DataFrame, 
                       data_type: str = 'games',
                       normalize_teams: bool = True,
                       handle_missing: bool = True,
                       validate_quality: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Process NFL dataset with common transformations.
    
    Args:
        df: Input DataFrame
        data_type: Type of data being processed
        normalize_teams: Whether to normalize team names
        handle_missing: Whether to handle missing values
        validate_quality: Whether to validate data quality
        
    Returns:
        Tuple of (processed_df, quality_report)
    """
    processed_df = df.copy()
    quality_report = {}
    
    # Normalize team names if requested
    if normalize_teams:
        team_cols = [col for col in processed_df.columns if 'team' in col.lower() and 'name' in col.lower()]
        for col in team_cols:
            id_col = col.replace('name', 'id')
            if id_col not in processed_df.columns:
                processed_df[id_col] = processed_df[col].apply(
                    lambda x: normalize_nfl_team_name(x, return_format='id')
                )
    
    # Handle missing values if requested
    if handle_missing:
        processed_df = handle_nfl_missing_values(processed_df, strategy='nfl_defaults')
    
    # Validate quality if requested
    if validate_quality:
        quality_report = validate_nfl_data_quality(processed_df, data_type=data_type)
    
    return processed_df, quality_report


if __name__ == "__main__":
    # Example usage and testing
    sample_data = pd.DataFrame({
        'home_team_name': ['Kansas City Chiefs', 'Green Bay', 'TB'],
        'away_team_name': ['Denver Broncos', 'Chicago Bears', 'Saints'],
        'home_score': [28, 21, np.nan],
        'away_score': [17, 14, 24],
        'week': [1, 2, 3],
        'temperature': [75, 32, np.nan],
        'wind_speed': [5, 15, 8]
    })
    
    print("Testing NFL data helpers...")
    processed_data, quality = process_nfl_dataset(sample_data, data_type='games')
    print(f"Quality score: {quality['quality_score']:.3f}")
    print(f"Processed {len(processed_data)} games")
