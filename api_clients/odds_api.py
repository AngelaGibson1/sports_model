import requests
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from loguru import logger
import time

from config.settings import Settings
from utils.api_helpers import APIHelper, APIError, create_odds_api_headers

class OddsAPIClient:
    """Client for fetching odds data from The Odds API."""
    
    def __init__(self):
        self.base_url = Settings.ODDS_API_BASE_URL
        self.api_key = Settings.ODDS_API_KEY
        self.headers = create_odds_api_headers()
        self.api_helper = APIHelper(Settings.API_RATE_LIMITS['odds_api'])
        
        # Sport key mappings for The Odds API - with aliases
        self.sport_keys = {
            'nba': 'basketball_nba',
            'mlb': 'baseball_mlb', 
            'baseball': 'baseball_mlb',
            'baseball_mlb': 'baseball_mlb',
            'nfl': 'americanfootball_nfl'
        }
        
        # Market mappings
        self.market_types = {
            'moneyline': 'h2h',
            'spread': 'spreads',
            'totals': 'totals',
            'player_props': 'player_props'
        }
        
        if not self.api_key:
            logger.warning("Odds API key not configured")
    
    def _normalize_sport_key(self, sport_key: str) -> str:
        """Normalize sport key to canonical form."""
        key = str(sport_key).lower().strip()
        return self.sport_keys.get(key, key)
    
    def get_odds(self, 
                sport: str,
                markets: List[str] = ['h2h'],
                regions: str = 'us',
                bookmakers: Optional[List[str]] = None,
                odds_format: str = 'american',
                date_format: str = 'iso',
                commence_time_from: Optional[str] = None,
                commence_time_to: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch odds for a specific sport with proper v4 API formatting.
        
        Args:
            sport: Sport key ('nba', 'mlb', 'nfl')
            markets: List of markets to fetch
            regions: Comma-separated list of regions
            bookmakers: Optional list of specific bookmakers
            odds_format: Odds format ('american', 'decimal')
            date_format: Date format ('iso', 'unix')
            commence_time_from: ISO format start time
            commence_time_to: ISO format end time
            
        Returns:
            DataFrame with flattened odds data
        """
        if not self.api_key:
            logger.error("Odds API key not configured")
            return pd.DataFrame()
        
        sport_key = self._normalize_sport_key(sport)
        if not sport_key:
            logger.error(f"Unsupported sport: {sport}")
            return pd.DataFrame()
        
        url = f"{self.base_url}/sports/{sport_key}/odds"
        
        # Required parameters for v4 API
        params = {
            'apiKey': self.api_key,  # Critical: API key must be included
            'regions': regions,      # Required: us, uk, eu, au
            'oddsFormat': odds_format, # Required: american, decimal
            'dateFormat': date_format  # Required: iso, unix
        }
        
        # Handle markets parameter
        if markets and len(markets) > 0:
            # Filter out invalid markets - only use valid ones
            valid_markets = [m for m in markets if m in ['h2h', 'spreads', 'totals']]
            if valid_markets:
                params['markets'] = ','.join(valid_markets)
            else:
                params['markets'] = 'h2h'  # Default to moneyline only
        else:
            params['markets'] = 'h2h'  # Default to moneyline
        
        # Optional parameters
        if bookmakers and len(bookmakers) > 0:
            params['bookmakers'] = ','.join(bookmakers)
        if commence_time_from:
            params['commenceTimeFrom'] = commence_time_from
        if commence_time_to:
            params['commenceTimeTo'] = commence_time_to
        
        logger.info(f"🎯 Calling Odds API: {url}")
        logger.info(f"📋 Parameters: {dict(params)}")  # Log for debugging
        
        try:
            # Use direct requests instead of api_helper to ensure parameters are passed correctly
            response = requests.get(url, params=params, timeout=30)
            
            # Log response details for debugging
            logger.info(f"📊 Response status: {response.status_code}")
            if response.status_code != 200:
                logger.error(f"❌ API Error {response.status_code}: {response.text}")
                return pd.DataFrame()
            
            # Log API usage from headers
            remaining = response.headers.get('x-requests-remaining', 'unknown')
            logger.info(f"💳 API requests remaining: {remaining}")
            
            data = response.json()
            
            if not data:
                logger.warning("⚪ Odds API returned empty response")
                return pd.DataFrame()
                
            return self._parse_odds_response(data, sport)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Network error fetching odds for {sport}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"❌ Error fetching odds for {sport}: {e}")
            return pd.DataFrame()
    
    def _parse_odds_response(self, data: Union[List, Dict], sport: str) -> pd.DataFrame:
        """Parse odds API response into flattened DataFrame."""
        if not data:
            return pd.DataFrame()
        
        rows = []
        
        # Handle both list and dict responses
        if isinstance(data, dict):
            events = data.get('data', data.get('events', []))
        else:
            events = data
        
        for event in events:
            if not isinstance(event, dict):
                continue
            
            # Extract game information
            game_id = event.get('id', '')
            home_team = event.get('home_team', '')
            away_team = event.get('away_team', '')
            commence_time = event.get('commence_time', '')
            completed = event.get('completed', False)
            
            # Parse bookmaker odds
            for bookmaker in event.get('bookmakers', []):
                bookmaker_name = bookmaker.get('title', 'unknown')
                last_update = bookmaker.get('last_update', '')
                
                for market in bookmaker.get('markets', []):
                    market_key = market.get('key', '')
                    
                    # Process outcomes for each market
                    for outcome in market.get('outcomes', []):
                        team_name = outcome.get('name', '')
                        odds = outcome.get('price', 0)
                        point = outcome.get('point', None)
                        
                        # Determine if this is home or away team
                        team = None
                        if team_name == home_team:
                            team = 'home'
                        elif team_name == away_team:
                            team = 'away'
                        else:
                            # For totals market, use the outcome name (Over/Under)
                            team = team_name.lower() if team_name.lower() in ['over', 'under'] else team_name
                        
                        # Create row for this odds entry
                        row = {
                            'game_id': game_id,
                            'sport': sport,
                            'home_team': home_team,
                            'away_team': away_team,
                            'commence_time': commence_time,
                            'completed': completed,
                            'bookmaker': bookmaker_name,
                            'market': market_key,  # This will be 'h2h' for moneyline
                            'team': team,
                            'team_name': team_name,
                            'odds': odds,
                            'last_update': last_update
                        }
                        
                        # Add point for spreads/totals
                        if point is not None:
                            row['point'] = point
                        
                        rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            
            # Convert timestamps
            if 'commence_time' in df.columns:
                df['commence_time'] = pd.to_datetime(df['commence_time'], errors='coerce')
            if 'last_update' in df.columns:
                df['last_update'] = pd.to_datetime(df['last_update'], errors='coerce')
            
            logger.info(f"✅ Parsed {len(df)} odds rows from {len(events)} games")
            return df
        
        logger.warning("⚪ No odds data found in response")
        return pd.DataFrame()
    
    def get_game_odds(self, 
                     sport: str,
                     game_id: str,
                     markets: List[str] = ['h2h', 'spreads', 'totals']) -> Dict[str, Any]:
        """
        Fetch odds for a specific game.
        
        Args:
            sport: Sport key
            game_id: Specific game ID
            markets: Markets to fetch
            
        Returns:
            Dictionary with game odds data
        """
        sport_key = self._normalize_sport_key(sport)
        if not sport_key:
            return {}
        
        url = f"{self.base_url}/sports/{sport_key}/events/{game_id}/odds"
        
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': ','.join(markets),
            'oddsFormat': 'american',
            'dateFormat': 'iso'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"❌ Game odds API error {response.status_code}: {response.text}")
                return {}
            
        except Exception as e:
            logger.error(f"❌ Error fetching game odds for {game_id}: {e}")
            return {}
    
    def get_api_usage(self) -> Dict[str, Any]:
        """
        Get current API usage statistics.
        
        Returns:
            Dictionary with usage information
        """
        url = f"{self.base_url}/sports"
        params = {'apiKey': self.api_key}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            
            usage_info = {
                'status_code': response.status_code,
                'requests_used': response.headers.get('x-requests-used'),
                'requests_remaining': response.headers.get('x-requests-remaining'),
                'reset_time': response.headers.get('x-ratelimit-reset'),
                'quota_limit': response.headers.get('x-ratelimit-limit')
            }
            
            return usage_info
            
        except Exception as e:
            logger.error(f"❌ Error checking API usage: {e}")
            return {'error': str(e)}
    
    def validate_api_key(self) -> bool:
        """
        Validate that the API key is working.
        
        Returns:
            True if API key is valid
        """
        if not self.api_key:
            return False
        
        try:
            usage = self.get_api_usage()
            return usage.get('status_code') == 200
        except:
            return False
