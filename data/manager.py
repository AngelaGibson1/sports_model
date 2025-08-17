import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import asyncio
import concurrent.futures
from loguru import logger
import redis
import json
from pathlib import Path

from api_clients.odds_api import OddsAPIClient
from api_clients.sports_api import SportsAPIClient
from data.database.nba import NBADatabase
from data.database.mlb import MLBDatabase
from data.database.nfl import NFLDatabase
from config.settings import Settings
from utils.data_helpers import (
    calculate_rolling_averages, 
    normalize_team_names,
    handle_missing_values,
    validate_data_quality
)

class UniversalSportsDataManager:
    """
    Centralized data manager for all sports.
    Handles data ingestion, storage, and retrieval across NBA, MLB, and NFL.
    """
    
    def __init__(self, use_redis: bool = True):
        """Initialize the universal data manager."""
        logger.info("Initializing Universal Sports Data Manager...")
        
        # Initialize API clients
        self.nba_api = SportsAPIClient(Settings.NBA_HOST)
        self.mlb_api = SportsAPIClient(Settings.MLB_HOST)
        self.nfl_api = SportsAPIClient(Settings.NFL_HOST)
        self.odds_api = OddsAPIClient()
        
        # Initialize databases
        self.databases = {
            'nba': NBADatabase(),
            'mlb': MLBDatabase(),
            'nfl': NFLDatabase()
        }
        
        # Initialize Redis cache if available
        self.use_redis = use_redis
        self.redis_client = None
        if use_redis:
            try:
                self.redis_client = redis.Redis(
                    host=Settings.REDIS_HOST,
                    port=Settings.REDIS_PORT,
                    db=Settings.REDIS_DB,
                    password=Settings.REDIS_PASSWORD,
                    decode_responses=True
                )
                self.redis_client.ping()
                logger.info("✅ Redis cache connected")
            except Exception as e:
                logger.warning(f"Redis not available: {e}")
                self.use_redis = False
        
        # API client mapping
        self.api_clients = {
            'nba': self.nba_api,
            'mlb': self.mlb_api,
            'nfl': self.nfl_api
        }
        
        logger.info("✅ Universal Sports Data Manager initialized")
    
    # ============= MAIN DATA INGESTION METHODS =============
    
    def ingest_daily_data(self, 
                         sport: str, 
                         date: Optional[str] = None,
                         include_odds: bool = True,
                         include_player_stats: bool = False) -> Dict[str, Any]:
        """
        Ingest all daily data for a sport.
        
        Args:
            sport: Sport key ('nba', 'mlb', 'nfl')
            date: Date to ingest (YYYY-MM-DD), defaults to today
            include_odds: Whether to fetch odds data
            include_player_stats: Whether to fetch player statistics
            
        Returns:
            Dictionary with ingestion results
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"🔄 Starting daily data ingestion for {sport.upper()} - {date}")
        
        results = {
            'sport': sport,
            'date': date,
            'games': 0,
            'odds': 0,
            'player_stats': 0,
            'errors': []
        }
        
        try:
            # 1. Fetch games/schedule
            logger.info(f"📅 Fetching games for {date}")
            games_df = self.get_games(sport, date=date)
            
            if not games_df.empty:
                # Save to database
                self.databases[sport].save_games(games_df)
                results['games'] = len(games_df)
                logger.info(f"✅ Saved {len(games_df)} games to database")
                
                # 2. Fetch odds if requested
                if include_odds and not games_df.empty:
                    logger.info(f"💰 Fetching odds data")
                    odds_df = self.get_odds(sport, date=date)
                    
                    if not odds_df.empty:
                        # Merge with games and save
                        enriched_games = self._merge_games_with_odds(games_df, odds_df)
                        self.databases[sport].save_games_with_odds(enriched_games)
                        results['odds'] = len(odds_df)
                        logger.info(f"✅ Saved odds for {len(odds_df)} games")
                
                # 3. Fetch player stats if requested
                if include_player_stats:
                    logger.info(f"⚾ Fetching player statistics")
                    player_stats = self._fetch_daily_player_stats(sport, games_df)
                    
                    if player_stats:
                        self.databases[sport].save_player_stats(player_stats)
                        results['player_stats'] = len(player_stats)
                        logger.info(f"✅ Saved stats for {len(player_stats)} players")
            else:
                logger.warning(f"No games found for {sport} on {date}")
        
        except Exception as e:
            error_msg = f"Error in daily ingestion for {sport}: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
        
        logger.info(f"✅ Daily ingestion complete for {sport.upper()}: {results}")
        return results
    
    def ingest_historical_data(self,
                              sport: str,
                              start_date: str,
                              end_date: str,
                              batch_size: int = 7) -> Dict[str, Any]:
        """
        Ingest historical data for a date range.
        
        Args:
            sport: Sport key
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            batch_size: Number of days to process in each batch
            
        Returns:
            Dictionary with ingestion results
        """
        logger.info(f"🔄 Starting historical ingestion for {sport.upper()}: {start_date} to {end_date}")
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        total_days = (end_dt - start_dt).days + 1
        results = {
            'sport': sport,
            'total_days': total_days,
            'processed_days': 0,
            'total_games': 0,
            'errors': []
        }
        
        current_date = start_dt
        
        while current_date <= end_dt:
            batch_end = min(current_date + timedelta(days=batch_size-1), end_dt)
            
            logger.info(f"📅 Processing batch: {current_date.strftime('%Y-%m-%d')} to {batch_end.strftime('%Y-%m-%d')}")
            
            # Process each day in the batch
            batch_results = []
            batch_date = current_date
            
            while batch_date <= batch_end:
                date_str = batch_date.strftime('%Y-%m-%d')
                
                try:
                    day_result = self.ingest_daily_data(sport, date_str, include_odds=False)
                    batch_results.append(day_result)
                    results['total_games'] += day_result['games']
                    results['processed_days'] += 1
                    
                except Exception as e:
                    error_msg = f"Error processing {date_str}: {e}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
                
                batch_date += timedelta(days=1)
            
            # Log batch progress
            progress = (results['processed_days'] / total_days) * 100
            logger.info(f"📊 Progress: {results['processed_days']}/{total_days} days ({progress:.1f}%)")
            
            current_date = batch_end + timedelta(days=1)
        
        logger.info(f"✅ Historical ingestion complete: {results}")
        return results
    
    # ============= DATA RETRIEVAL METHODS =============
    
    def get_games(self, 
                  sport: str,
                  date: Optional[str] = None,
                  season: Optional[int] = None,
                  team_id: Optional[int] = None,
                  live_only: bool = False) -> pd.DataFrame:
        """
        Get games for a sport with caching.
        
        Args:
            sport: Sport key
            date: Optional date filter
            season: Optional season filter
            team_id: Optional team filter
            live_only: Get only live games
            
        Returns:
            DataFrame with game data
        """
        cache_key = f"games_{sport}_{date}_{season}_{team_id}_{live_only}"
        
        # Try cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        try:
            api_client = self.api_clients[sport]
            
            # Determine league ID based on sport
            league_id = self._get_default_league_id(sport)
            
            games_df = api_client.get_games(
                date=date,
                season=season,
                league_id=league_id,
                team_id=team_id,
                live=live_only
            )
            
            if not games_df.empty:
                # Normalize team names
                games_df = normalize_team_names(games_df, ['home_team_name', 'away_team_name'], sport)
                
                # Cache the result
                self._save_to_cache(cache_key, games_df, Settings.CACHE_DURATIONS['game_schedule'])
            
            return games_df
            
        except Exception as e:
            logger.error(f"Error fetching games for {sport}: {e}")
            return pd.DataFrame()
    
    def get_odds(self,
                sport: str,
                date: Optional[str] = None,
                markets: List[str] = ['h2h', 'spreads', 'totals']) -> pd.DataFrame:
        """
        Get odds data for a sport.
        
        Args:
            sport: Sport key
            date: Optional date filter
            markets: Markets to fetch
            
        Returns:
            DataFrame with odds data
        """
        cache_key = f"odds_{sport}_{date}_{'_'.join(markets)}"
        
        # Try cache first (shorter cache for odds)
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        try:
            odds_df = self.odds_api.get_odds(
                sport=sport,
                markets=markets,
                commence_time_from=f"{date}T00:00:00Z" if date else None,
                commence_time_to=f"{date}T23:59:59Z" if date else None
            )
            
            if not odds_df.empty:
                # Normalize team names in odds
                odds_df = normalize_team_names(odds_df, ['home_team', 'away_team'], sport)
                
                # Cache with shorter duration for odds
                self._save_to_cache(cache_key, odds_df, Settings.CACHE_DURATIONS['odds'])
            
            return odds_df
            
        except Exception as e:
            logger.error(f"Error fetching odds for {sport}: {e}")
            return pd.DataFrame()
    
    def get_team_statistics(self,
                           sport: str,
                           season: int,
                           team_id: Optional[int] = None) -> pd.DataFrame:
        """
        Get team statistics for a season.
        
        Args:
            sport: Sport key
            season: Season year
            team_id: Optional specific team
            
        Returns:
            DataFrame with team statistics
        """
        cache_key = f"team_stats_{sport}_{season}_{team_id}"
        
        # Try cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        try:
            api_client = self.api_clients[sport]
            league_id = self._get_default_league_id(sport)
            
            if team_id:
                # Get stats for specific team
                stats_df = api_client.get_team_statistics(team_id, season, league_id)
            else:
                # Get stats for all teams
                stats_df = api_client.get_all_team_stats_for_season(season, league_id)
            
            if not stats_df.empty:
                # Cache the result
                self._save_to_cache(cache_key, stats_df, Settings.CACHE_DURATIONS['team_stats'])
            
            return stats_df
            
        except Exception as e:
            logger.error(f"Error fetching team stats for {sport}: {e}")
            return pd.DataFrame()
    
    def get_player_statistics(self,
                             sport: str,
                             season: int,
                             player_id: Optional[int] = None,
                             team_id: Optional[int] = None) -> pd.DataFrame:
        """
        Get player statistics for a season.
        
        Args:
            sport: Sport key
            season: Season year
            player_id: Optional specific player
            team_id: Optional team filter
            
        Returns:
            DataFrame with player statistics
        """
        cache_key = f"player_stats_{sport}_{season}_{player_id}_{team_id}"
        
        # Try cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        try:
            api_client = self.api_clients[sport]
            league_id = self._get_default_league_id(sport)
            
            if player_id:
                # Get stats for specific player
                stats_df = api_client.get_player_statistics(player_id, season, team_id, league_id)
            else:
                # Get all players for team or league
                players_df = api_client.get_players(team_id=team_id, season=season)
                
                if players_df.empty:
                    return pd.DataFrame()
                
                # Get stats for all players (this could be slow)
                all_stats = []
                for _, player in players_df.head(50).iterrows():  # Limit to prevent overload
                    player_stats = api_client.get_player_statistics(
                        player['player_id'], season, team_id, league_id
                    )
                    if not player_stats.empty:
                        all_stats.append(player_stats)
                
                stats_df = pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()
            
            if not stats_df.empty:
                # Cache the result
                self._save_to_cache(cache_key, stats_df, Settings.CACHE_DURATIONS['player_stats'])
            
            return stats_df
            
        except Exception as e:
            logger.error(f"Error fetching player stats for {sport}: {e}")
            return pd.DataFrame()
    
    def fetch_historical_data(self, 
                             sport: str,
                             seasons: Optional[List[int]] = None,
                             include_rolling_stats: bool = True) -> pd.DataFrame:
        """
        Fetch and merge historical data from database for model training.
        
        Args:
            sport: Sport key
            seasons: Optional list of seasons, defaults to last 3 seasons
            include_rolling_stats: Whether to include rolling statistics
            
        Returns:
            DataFrame ready for feature engineering
        """
        logger.info(f"📊 Fetching historical data for {sport.upper()}")
        
        if seasons is None:
            current_year = datetime.now().year
            seasons = [current_year - 2, current_year - 1, current_year]
        
        cache_key = f"historical_{sport}_{'_'.join(map(str, seasons))}_{include_rolling_stats}"
        
        # Try cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            logger.info(f"✅ Loaded historical data from cache")
            return cached_data
        
        try:
            # Get data from database
            historical_df = self.databases[sport].get_historical_data(seasons)
            
            if historical_df.empty:
                logger.warning(f"No historical data found for {sport}")
                return pd.DataFrame()
            
            # Add rolling statistics if requested
            if include_rolling_stats:
                historical_df = self._add_rolling_features(historical_df, sport)
            
            # Handle missing values
            historical_df = handle_missing_values(historical_df, strategy='smart')
            
            # Validate data quality
            quality_report = validate_data_quality(historical_df)
            logger.info(f"📋 Data quality: {historical_df.shape[0]} rows, "
                       f"{len(quality_report['high_missing_columns'])} high-missing columns")
            
            # Cache the processed data
            self._save_to_cache(cache_key, historical_df, Settings.CACHE_DURATIONS['historical_data'])
            
            logger.info(f"✅ Fetched historical data: {historical_df.shape}")
            return historical_df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {sport}: {e}")
            return pd.DataFrame()
    
    # ============= DATA PROCESSING METHODS =============
    
    def _add_rolling_features(self, df: pd.DataFrame, sport: str) -> pd.DataFrame:
        """Add rolling statistics features."""
        sport_config = Settings.SPORT_CONFIGS.get(sport, {})
        key_stats = sport_config.get('key_stats', [])
        
        if not key_stats:
            return df
        
        # Get rolling windows for this sport
        windows = Settings.ROLLING_WINDOWS.get(sport, {'short': 5, 'medium': 10, 'long': 20})
        
        # Add rolling averages for key statistics
        for window_name, window_size in windows.items():
            df = calculate_rolling_averages(
                df=df,
                columns=key_stats,
                window_sizes=[window_size],
                group_by='team_id' if 'team_id' in df.columns else None
            )
        
        return df
    
    def _merge_games_with_odds(self, games_df: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
        """Merge games DataFrame with odds data."""
        if games_df.empty or odds_df.empty:
            return games_df
        
        # Create matching keys
        games_df['match_key'] = games_df['home_team_name'] + '_vs_' + games_df['away_team_name']
        odds_df['match_key'] = odds_df['home_team'] + '_vs_' + odds_df['away_team']
        
        # Pivot odds data to get one row per game
        odds_pivot = odds_df.pivot_table(
            index=['match_key', 'game_id'],
            columns=['market', 'team'],
            values='odds',
            aggfunc='mean'
        ).reset_index()
        
        # Flatten column names
        odds_pivot.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                             for col in odds_pivot.columns]
        
        # Merge with games
        merged_df = pd.merge(games_df, odds_pivot, on='match_key', how='left')
        
        # Clean up
        merged_df = merged_df.drop('match_key', axis=1)
        
        return merged_df
    
    def _fetch_daily_player_stats(self, sport: str, games_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Fetch player stats for games on a specific day."""
        if games_df.empty:
            return []
        
        # This is a simplified version - in practice, you'd need to map
        # games to actual player performances, which might require
        # additional API calls or data sources
        
        player_stats = []
        
        # Example: fetch key players from teams playing today
        unique_teams = set(games_df['home_team_id'].dropna()) | set(games_df['away_team_id'].dropna())
        
        for team_id in list(unique_teams)[:5]:  # Limit to prevent API overload
            try:
                team_players = self.get_player_statistics(
                    sport=sport,
                    season=datetime.now().year,
                    team_id=int(team_id)
                )
                
                if not team_players.empty:
                    player_stats.extend(team_players.to_dict('records'))
                    
            except Exception as e:
                logger.warning(f"Error fetching players for team {team_id}: {e}")
        
        return player_stats
    
    def _get_default_league_id(self, sport: str) -> Optional[int]:
        """Get default league ID for a sport."""
        # These would need to be determined from the API or configured
        default_leagues = {
            'nba': 12,  # NBA league ID (example)
            'mlb': 1,   # MLB league ID (example)
            'nfl': 1    # NFL league ID (example)
        }
        return default_leagues.get(sport)
    
    # ============= CACHING METHODS =============
    
    def _get_from_cache(self, key: str) -> Optional[pd.DataFrame]:
        """Get data from cache."""
        if not self.use_redis or not self.redis_client:
            return None
        
        try:
            cached_json = self.redis_client.get(key)
            if cached_json:
                data = json.loads(cached_json)
                return pd.DataFrame(data)
        except Exception as e:
            logger.warning(f"Cache read error for {key}: {e}")
        
        return None
    
    def _save_to_cache(self, key: str, df: pd.DataFrame, ttl: int):
        """Save data to cache."""
        if not self.use_redis or not self.redis_client or df.empty:
            return
        
        try:
            # Convert DataFrame to JSON
            data_json = df.to_json(orient='records')
            self.redis_client.setex(key, ttl, data_json)
        except Exception as e:
            logger.warning(f"Cache write error for {key}: {e}")
    
    def clear_cache(self, pattern: Optional[str] = None):
        """Clear cache entries."""
        if not self.use_redis or not self.redis_client:
            return
        
        try:
            if pattern:
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
                    logger.info(f"Cleared {len(keys)} cache entries matching '{pattern}'")
            else:
                self.redis_client.flushdb()
                logger.info("Cleared entire cache")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    # ============= UTILITY METHODS =============
    
    def get_data_summary(self, sport: str) -> Dict[str, Any]:
        """Get summary of available data for a sport."""
        try:
            summary = self.databases[sport].get_data_summary()
            
            # Add API connectivity status
            summary['api_status'] = {
                'sports_api': self._test_api_connectivity(sport),
                'odds_api': self.odds_api.validate_api_key()
            }
            
            # Add cache status
            summary['cache_status'] = {
                'redis_available': self.use_redis,
                'cache_keys': len(self.redis_client.keys('*')) if self.use_redis else 0
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting data summary for {sport}: {e}")
            return {'error': str(e)}
    
    def _test_api_connectivity(self, sport: str) -> bool:
        """Test API connectivity for a sport."""
        try:
            api_client = self.api_clients[sport]
            seasons = api_client.get_seasons()
            return len(seasons) > 0
        except:
            return False
    
    def bulk_ingest(self, 
                   sports: List[str],
                   start_date: str,
                   end_date: str,
                   max_workers: int = 3) -> Dict[str, Any]:
        """
        Bulk ingest data for multiple sports using parallel processing.
        
        Args:
            sports: List of sports to process
            start_date: Start date
            end_date: End date  
            max_workers: Maximum parallel workers
            
        Returns:
            Dictionary with results for each sport
        """
        logger.info(f"🚀 Starting bulk ingestion for {sports}")
        
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit ingestion tasks
            future_to_sport = {
                executor.submit(
                    self.ingest_historical_data, 
                    sport, 
                    start_date, 
                    end_date
                ): sport for sport in sports
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_sport):
                sport = future_to_sport[future]
                try:
                    results[sport] = future.result()
                    logger.info(f"✅ Completed ingestion for {sport.upper()}")
                except Exception as e:
                    error_msg = f"Error in bulk ingestion for {sport}: {e}"
                    logger.error(error_msg)
                    results[sport] = {'error': error_msg}
        
        logger.info(f"✅ Bulk ingestion complete: {results}")
        return results
