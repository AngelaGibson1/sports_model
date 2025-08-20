import pandas as pd 
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import warnings
from loguru import logger
import json
from pathlib import Path

warnings.filterwarnings('ignore')

def calculate_rolling_averages(df: pd.DataFrame, 
                             columns: List[str], 
                             window_sizes: List[int],
                             group_by: Optional[str] = None,
                             min_periods: Optional[int] = None) -> pd.DataFrame:
    """
    Calculate rolling averages for specified columns.
    
    Args:
        df: Input DataFrame
        columns: Columns to calculate rolling averages for
        window_sizes: List of window sizes to use
        group_by: Column to group by (e.g., 'team_id', 'player_id')
        min_periods: Minimum periods required for rolling calculation
        
    Returns:
        DataFrame with rolling average columns added
    """
    df_copy = df.copy()
    
    # Ensure DataFrame is sorted by date if available
    if 'date' in df_copy.columns:
        df_copy = df_copy.sort_values('date')
    elif 'game_date' in df_copy.columns:
        df_copy = df_copy.sort_values('game_date')
    
    for window in window_sizes:
        min_periods_val = min_periods or max(1, window // 3)
        
        for col in columns:
            if col not in df_copy.columns:
                logger.warning(f"Column {col} not found in DataFrame")
                continue
            
            new_col_name = f"{col}_roll_{window}"
            
            if group_by and group_by in df_copy.columns:
                # Calculate rolling average within groups
                df_copy[new_col_name] = (df_copy.groupby(group_by)[col]
                                       .rolling(window=window, min_periods=min_periods_val)
                                       .mean()
                                       .reset_index(level=0, drop=True))
            else:
                # Calculate rolling average for entire DataFrame
                df_copy[new_col_name] = (df_copy[col]
                                       .rolling(window=window, min_periods=min_periods_val)
                                       .mean())
    
    return df_copy

def calculate_rolling_statistics(df: pd.DataFrame,
                               columns: List[str],
                               window: int,
                               stats: List[str] = ['mean', 'std', 'min', 'max'],
                               group_by: Optional[str] = None) -> pd.DataFrame:
    """
    Calculate multiple rolling statistics for specified columns.
    
    Args:
        df: Input DataFrame
        columns: Columns to calculate statistics for
        window: Rolling window size
        stats: List of statistics to calculate ('mean', 'std', 'min', 'max', 'median')
        group_by: Column to group by
        
    Returns:
        DataFrame with rolling statistics columns added
    """
    df_copy = df.copy()
    
    for col in columns:
        if col not in df_copy.columns:
            continue
        
        for stat in stats:
            new_col_name = f"{col}_roll_{window}_{stat}"
            
            if group_by and group_by in df_copy.columns:
                rolling_obj = df_copy.groupby(group_by)[col].rolling(window=window, min_periods=1)
            else:
                rolling_obj = df_copy[col].rolling(window=window, min_periods=1)
            
            if stat == 'mean':
                df_copy[new_col_name] = rolling_obj.mean()
            elif stat == 'std':
                df_copy[new_col_name] = rolling_obj.std()
            elif stat == 'min':
                df_copy[new_col_name] = rolling_obj.min()
            elif stat == 'max':
                df_copy[new_col_name] = rolling_obj.max()
            elif stat == 'median':
                df_copy[new_col_name] = rolling_obj.median()
            
            # Reset index if grouped
            if group_by and group_by in df_copy.columns:
                df_copy[new_col_name] = df_copy[new_col_name].reset_index(level=0, drop=True)
    
    return df_copy

def calculate_team_form(df: pd.DataFrame,
                       team_col: str = 'team',
                       result_col: str = 'win',
                       window: int = 10) -> pd.DataFrame:
    """
    Calculate team form (win percentage over recent games).
    
    Args:
        df: DataFrame with game results
        team_col: Column containing team identifier
        result_col: Column containing win/loss (1/0)
        window: Number of recent games to consider
        
    Returns:
        DataFrame with form columns added
    """
    df_copy = df.copy()
    
    # Calculate rolling win percentage
    df_copy = df_copy.sort_values(['date'] if 'date' in df_copy.columns else df_copy.columns[0])
    
    df_copy[f'form_{window}'] = (df_copy.groupby(team_col)[result_col]
                                .rolling(window=window, min_periods=1)
                                .mean()
                                .reset_index(level=0, drop=True))
    
    # Calculate form trend (recent form vs longer term)
    if window >= 6:
        short_window = max(3, window // 3)
        df_copy[f'form_{short_window}'] = (df_copy.groupby(team_col)[result_col]
                                         .rolling(window=short_window, min_periods=1)
                                         .mean()
                                         .reset_index(level=0, drop=True))
        
        df_copy[f'form_trend_{window}'] = df_copy[f'form_{short_window}'] - df_copy[f'form_{window}']
    
    return df_copy

def calculate_head_to_head_stats(df: pd.DataFrame,
                                team1_col: str = 'home_team',
                                team2_col: str = 'away_team',
                                result_col: str = 'home_win') -> pd.DataFrame:
    """
    Calculate head-to-head statistics between teams.
    
    Args:
        df: DataFrame with game results
        team1_col: Home team column
        team2_col: Away team column  
        result_col: Result column (1 if home wins, 0 if away wins)
        
    Returns:
        DataFrame with H2H statistics
    """
    h2h_stats = []
    
    # Get unique team pairs
    team_pairs = df[[team1_col, team2_col]].drop_duplicates()
    
    for _, row in team_pairs.iterrows():
        team1, team2 = row[team1_col], row[team2_col]
        
        # Get all games between these teams (both home and away)
        mask1 = (df[team1_col] == team1) & (df[team2_col] == team2)
        mask2 = (df[team1_col] == team2) & (df[team2_col] == team1)
        
        games = df[mask1 | mask2].copy()
        
        if len(games) > 0:
            # Calculate stats for team1 vs team2
            team1_wins = 0
            team2_wins = 0
            
            for _, game in games.iterrows():
                if game[team1_col] == team1:
                    # team1 is home
                    if game[result_col] == 1:
                        team1_wins += 1
                    else:
                        team2_wins += 1
                else:
                    # team1 is away
                    if game[result_col] == 0:
                        team1_wins += 1
                    else:
                        team2_wins += 1
            
            total_games = len(games)
            h2h_stats.append({
                'team1': team1,
                'team2': team2,
                'games_played': total_games,
                'team1_wins': team1_wins,
                'team2_wins': team2_wins,
                'team1_win_pct': team1_wins / total_games if total_games > 0 else 0
            })
    
    return pd.DataFrame(h2h_stats)

def normalize_team_names(df: pd.DataFrame, 
                        team_columns: List[str],
                        sport: str) -> pd.DataFrame:
    """
    Normalize team names to standard abbreviations.
    
    Args:
        df: Input DataFrame
        team_columns: Columns containing team names
        sport: Sport type ('nba', 'mlb', 'nfl')
        
    Returns:
        DataFrame with normalized team names
    """
    df_copy = df.copy()
    
    # Define team name mappings for each sport
    team_mappings = {
        'nba': {
            'Los Angeles Lakers': 'LAL',
            'Los Angeles Clippers': 'LAC', 
            'Golden State Warriors': 'GSW',
            'Boston Celtics': 'BOS',
            'Miami Heat': 'MIA',
            # Add more NBA mappings as needed
        },
        'mlb': {
            'New York Yankees': 'NYY',
            'Los Angeles Dodgers': 'LAD',
            'Boston Red Sox': 'BOS',
            'Chicago Cubs': 'CHC',
            'Tampa Bay Rays': 'TBR',
            # Add more MLB mappings as needed  
        },
        'nfl': {
            'New England Patriots': 'NE',
            'Dallas Cowboys': 'DAL',
            'Green Bay Packers': 'GB',
            'Pittsburgh Steelers': 'PIT',
            'San Francisco 49ers': 'SF',
            # Add more NFL mappings as needed
        }
    }
    
    mapping = team_mappings.get(sport.lower(), {})
    
    for col in team_columns:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].replace(mapping)
    
    return df_copy

def create_feature_interactions(df: pd.DataFrame,
                              feature_pairs: List[Tuple[str, str]],
                              interaction_types: List[str] = ['multiply', 'divide', 'subtract']) -> pd.DataFrame:
    """
    Create interaction features between specified feature pairs.
    
    Args:
        df: Input DataFrame
        feature_pairs: List of tuples containing feature column names
        interaction_types: Types of interactions to create
        
    Returns:
        DataFrame with interaction features added
    """
    df_copy = df.copy()
    
    for feat1, feat2 in feature_pairs:
        if feat1 not in df_copy.columns or feat2 not in df_copy.columns:
            continue
        
        # Skip if either feature has non-numeric data
        if not pd.api.types.is_numeric_dtype(df_copy[feat1]) or not pd.api.types.is_numeric_dtype(df_copy[feat2]):
            continue
        
        for interaction in interaction_types:
            if interaction == 'multiply':
                df_copy[f"{feat1}_x_{feat2}"] = df_copy[feat1] * df_copy[feat2]
            elif interaction == 'divide':
                # Avoid division by zero
                df_copy[f"{feat1}_div_{feat2}"] = df_copy[feat1] / (df_copy[feat2] + 1e-8)
            elif interaction == 'subtract':
                df_copy[f"{feat1}_minus_{feat2}"] = df_copy[feat1] - df_copy[feat2]
            elif interaction == 'add':
                df_copy[f"{feat1}_plus_{feat2}"] = df_copy[feat1] + df_copy[feat2]
    
    return df_copy

def handle_missing_values(df: pd.DataFrame,
                         strategy: str = 'smart',
                         numeric_fill: Optional[Union[str, float]] = None) -> pd.DataFrame:
    """
    Handle missing values in DataFrame using various strategies.
    
    Args:
        df: Input DataFrame
        strategy: Strategy to use ('smart', 'drop', 'fill', 'interpolate')
        numeric_fill: Value or method to use for numeric columns
        
    Returns:
        DataFrame with missing values handled
    """
    df_copy = df.copy()
    
    if strategy == 'smart':
        # Use different strategies for different column types
        for col in df_copy.columns:
            if df_copy[col].isnull().sum() == 0:
                continue
            
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                # For numeric columns, use median
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
            elif pd.api.types.is_categorical_dtype(df_copy[col]) or df_copy[col].dtype == 'object':
                # For categorical columns, use mode
                mode_val = df_copy[col].mode()
                if len(mode_val) > 0:
                    df_copy[col] = df_copy[col].fillna(mode_val[0])
                else:
                    df_copy[col] = df_copy[col].fillna('Unknown')
            else:
                # For datetime and other types, forward fill
                df_copy[col] = df_copy[col].fillna(method='ffill')
    
    elif strategy == 'drop':
        df_copy = df_copy.dropna()
    
    elif strategy == 'fill':
        if numeric_fill is not None:
            df_copy = df_copy.fillna(numeric_fill)
        else:
            df_copy = df_copy.fillna(0)
    
    elif strategy == 'interpolate':
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        df_copy[numeric_cols] = df_copy[numeric_cols].interpolate()
        
        # Fill remaining non-numeric columns
        for col in df_copy.columns:
            if col not in numeric_cols:
                df_copy[col] = df_copy[col].fillna(method='ffill')
    
    return df_copy

def detect_outliers(df: pd.DataFrame,
                   columns: List[str],
                   method: str = 'iqr',
                   threshold: float = 1.5) -> pd.DataFrame:
    """
    Detect outliers in specified columns.
    
    Args:
        df: Input DataFrame
        columns: Columns to check for outliers
        method: Method to use ('iqr', 'zscore', 'isolation')
        threshold: Threshold for outlier detection
        
    Returns:
        DataFrame with outlier flags added
    """
    df_copy = df.copy()
    
    for col in columns:
        if col not in df_copy.columns or not pd.api.types.is_numeric_dtype(df_copy[col]):
            continue
        
        outlier_col = f"{col}_outlier"
        
        if method == 'iqr':
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df_copy[outlier_col] = (df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)
        
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(df_copy[col].dropna()))
            df_copy[outlier_col] = False
            df_copy.loc[df_copy[col].notna(), outlier_col] = z_scores > threshold
        
        elif method == 'isolation':
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso_forest.fit_predict(df_copy[[col]].dropna())
            df_copy[outlier_col] = False
            df_copy.loc[df_copy[col].notna(), outlier_col] = outliers == -1
    
    return df_copy

def encode_categorical_features(df: pd.DataFrame,
                              categorical_columns: List[str],
                              encoding_type: str = 'onehot') -> pd.DataFrame:
    """
    Encode categorical features for machine learning.
    
    Args:
        df: Input DataFrame
        categorical_columns: Columns to encode
        encoding_type: Type of encoding ('onehot', 'label', 'target')
        
    Returns:
        DataFrame with encoded features
    """
    df_copy = df.copy()
    
    for col in categorical_columns:
        if col not in df_copy.columns:
            continue
        
        if encoding_type == 'onehot':
            # One-hot encoding
            dummies = pd.get_dummies(df_copy[col], prefix=col, dummy_na=False)
            df_copy = pd.concat([df_copy, dummies], axis=1)
            df_copy = df_copy.drop(col, axis=1)
        
        elif encoding_type == 'label':
            # Label encoding
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df_copy[f"{col}_encoded"] = le.fit_transform(df_copy[col].astype(str))
    
    return df_copy

def create_lag_features(df: pd.DataFrame,
                       columns: List[str],
                       lags: List[int],
                       group_by: Optional[str] = None) -> pd.DataFrame:
    """
    Create lag features for time series data.
    
    Args:
        df: Input DataFrame
        columns: Columns to create lags for
        lags: List of lag periods
        group_by: Column to group by for lag calculation
        
    Returns:
        DataFrame with lag features added
    """
    df_copy = df.copy()
    
    for col in columns:
        if col not in df_copy.columns:
            continue
        
        for lag in lags:
            lag_col = f"{col}_lag_{lag}"
            
            if group_by and group_by in df_copy.columns:
                df_copy[lag_col] = df_copy.groupby(group_by)[col].shift(lag)
            else:
                df_copy[lag_col] = df_copy[col].shift(lag)
    
    return df_copy

def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate data quality and return summary statistics.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dictionary with data quality metrics
    """
    quality_report = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'column_types': df.dtypes.to_dict()
    }
    
    # Check for columns with high missing percentage
    total_rows = len(df)
    high_missing_cols = []
    for col, missing_count in quality_report['missing_values'].items():
        if missing_count / total_rows > 0.5:  # More than 50% missing
            high_missing_cols.append(col)
    
    quality_report['high_missing_columns'] = high_missing_cols
    
    # Check for constant columns
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
    
    quality_report['constant_columns'] = constant_cols
    
    # Check numeric column ranges
    numeric_ranges = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        numeric_ranges[col] = {
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean(),
            'std': df[col].std()
        }
    
    quality_report['numeric_ranges'] = numeric_ranges
    
    return quality_report

def save_processed_data(df: pd.DataFrame, 
                       filepath: Union[str, Path],
                       format: str = 'parquet') -> None:
    """
    Save processed DataFrame to file.
    
    Args:
        df: DataFrame to save
        filepath: Output file path
        format: File format ('parquet', 'csv', 'pickle')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'parquet':
        df.to_parquet(filepath, compression='snappy')
    elif format == 'csv':
        df.to_csv(filepath, index=False)
    elif format == 'pickle':
        df.to_pickle(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Saved DataFrame with shape {df.shape} to {filepath}")

def load_processed_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load processed DataFrame from file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded DataFrame
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.parquet':
        return pd.read_parquet(filepath)
    elif filepath.suffix == '.csv':
        return pd.read_csv(filepath)
    elif filepath.suffix in ['.pkl', '.pickle']:
        return pd.read_pickle(filepath)
    else:
        raise ValueError(f"Unsupported file type: {filepath.suffix}")
