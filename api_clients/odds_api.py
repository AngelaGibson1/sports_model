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
        
        # Sport key mappings for The Odds API
        self.sport_keys = {
            'nba': 'basketball_nba',
            'mlb': 'baseball_mlb', 
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
    
    def get_odds(self, 
                sport: str,
                markets: List[str] = ['h2h', 'spreads', 'totals'],
                regions: str = 'us',
                bookmakers: Optional[List[str]] = None,
                commence_time_from: Optional[str] = None,
                commence_time_to: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch odds for a specific sport.
        
        Args:
            sport: Sport key ('nba', 'mlb', 'nfl')
            markets: List of markets to fetch
            regions: Comma-separated list of regions
            bookmakers: Optional list of specific bookmakers
            commence_time_from: ISO format start time
            commence_time_to: ISO format end time
            
        Returns:
            DataFrame with odds data
        """
        if not self.api_key:
            logger.error("Odds API key not configured")
            return pd.DataFrame()
        
        sport_key = self.sport_keys.get(sport.lower())
        if not sport_key:
            logger.error(f"Unsupported sport: {sport}")
            return pd.DataFrame()
        
        url = f"{self.base_url}/sports/{sport_key}/odds"
        
        params = {
            'apiKey': self.api_key,
            'regions': regions,
            'markets': ','.join(markets),
            'oddsFormat': 'american',
            'dateFormat': 'iso'
        }
        
        if bookmakers:
            params['bookmakers'] = ','.join(bookmakers)
        if commence_time_from:
            params['commenceTimeFrom'] = commence_time_from
        if commence_time_to:
            params['commenceTimeTo'] = commence_time_to
        
        try:
            data = self.api_helper.make_request(url, {}, params)
            
            # Log API usage
            if isinstance(data, dict) and 'remaining' in str(data):
                logger.info(f"Odds API requests remaining: check response headers")
            
            return self._parse_odds_response(data, sport)
            
        except APIError as e:
            logger.error(f"Error fetching odds for {sport}: {e}")
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
        sport_key = self.sport_keys.get(sport.lower())
        if not sport_key:
            return {}
        
        url = f"{self.base_url}/sports/{sport_key}/events/{game_id}/odds"
        
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': ','.join(markets),
            'oddsFormat': 'american'
        }
        
        try:
            data = self.api_helper.make_request(url, {}, params)
            return data if isinstance(data, dict) else {}
            
        except APIError as e:
            logger.error(f"Error fetching game odds for {game_id}: {e}")
            return {}
    
    def get_player_props(self, 
                        sport: str,
                        game_id: Optional[str] = None,
                        player_name: Optional[str] = None,
                        prop_type: str = 'player_points') -> pd.DataFrame:
        """
        Fetch player proposition bets.
        
        Args:
            sport: Sport key
            game_id: Optional specific game
            player_name: Optional specific player
            prop_type: Type of prop bet
            
        Returns:
            DataFrame with player props
        """
        sport_key = self.sport_keys.get(sport.lower())
        if not sport_key:
            return pd.DataFrame()
        
        if game_id:
            url = f"{self.base_url}/sports/{sport_key}/events/{game_id}/odds"
        else:
            url = f"{self.base_url}/sports/{sport_key}/odds"
        
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': 'player_props',
            'oddsFormat': 'american'
        }
        
        try:
            data = self.api_helper.make_request(url, {}, params)
            return self._parse_player_props_response(data, player_name, prop_type)
            
        except APIError as e:
            logger.error(f"Error fetching player props: {e}")
            return pd.DataFrame()
    
    def get_bookmakers(self) -> List[Dict[str, str]]:
        """
        Get list of available bookmakers.
        
        Returns:
            List of bookmaker information
        """
        url = f"{self.base_url}/bookmakers"
        
        params = {'apiKey': self.api_key}
        
        try:
            data = self.api_helper.make_request(url, {}, params)
            return data if isinstance(data, list) else []
            
        except APIError as e:
            logger.error(f"Error fetching bookmakers: {e}")
            return []
    
    def _parse_odds_response(self, data: Union[List, Dict], sport: str) -> pd.DataFrame:
        """Parse odds API response into DataFrame."""
        if not data:
            return pd.DataFrame()
        
        games_list = []
        
        # Handle both list and dict responses
        if isinstance(data, dict):
            games = data.get('data', data.get('events', []))
        else:
            games = data
        
        for game in games:
            if not isinstance(game, dict):
                continue
            
            game_info = {
                'game_id': game.get('id', ''),
                'sport': sport,
                'home_team': game.get('home_team', ''),
                'away_team': game.get('away_team', ''),
                'commence_time': game.get('commence_time', ''),
                'completed': game.get('completed', False)
            }
            
            # Parse bookmaker odds
            for bookmaker in game.get('bookmakers', []):
                bookmaker_name = bookmaker.get('title', 'unknown')
                
                for market in bookmaker.get('markets', []):
                    market_key = market.get('key', '')
                    
                    if market_key == 'h2h':  # Moneyline
                        for outcome in market.get('outcomes', []):
                            team = outcome.get('name', '')
                            odds = outcome.get('price', 0)
                            
                            if team == game_info['home_team']:
                                games_list.append({
                                    **game_info,
                                    'bookmaker': bookmaker_name,
                                    'market': 'moneyline',
                                    'team': 'home',
                                    'team_name': team,
                                    'odds': odds,
                                    'last_update': market.get('last_update', '')
                                })
                            elif team == game_info['away_team']:
                                games_list.append({
                                    **game_info,
                                    'bookmaker': bookmaker_name,
                                    'market': 'moneyline',
                                    'team': 'away',
                                    'team_name': team,
                                    'odds': odds,
                                    'last_update': market.get('last_update', '')
                                })
                    
                    elif market_key == 'spreads':  # Point spread
                        for outcome in market.get('outcomes', []):
                            team = outcome.get('name', '')
                            odds = outcome.get('price', 0)
                            point = outcome.get('point', 0)
                            
                            team_type = 'home' if team == game_info['home_team'] else 'away'
                            
                            games_list.append({
                                **game_info,
                                'bookmaker': bookmaker_name,
                                'market': 'spread',
                                'team': team_type,
                                'team_name': team,
                                'odds': odds,
                                'point': point,
                                'last_update': market.get('last_update', '')
                            })
                    
                    elif market_key == 'totals':  # Over/Under
                        for outcome in market.get('outcomes', []):
                            over_under = outcome.get('name', '')  # 'Over' or 'Under'
                            odds = outcome.get('price', 0)
                            point = outcome.get('point', 0)
                            
                            games_list.append({
                                **game_info,
                                'bookmaker': bookmaker_name,
                                'market': 'totals',
                                'team': over_under.lower(),
                                'team_name': over_under,
                                'odds': odds,
                                'point': point,
                                'last_update': market.get('last_update', '')
                            })
        
        if games_list:
            df = pd.DataFrame(games_list)
            # Convert commence_time to datetime
            if 'commence_time' in df.columns:
                df['commence_time'] = pd.to_datetime(df['commence_time'], errors='coerce')
            if 'last_update' in df.columns:
                df['last_update'] = pd.to_datetime(df['last_update'], errors='coerce')
            return df
        
        return pd.DataFrame()
    
    def _parse_player_props_response(self, 
                                   data: Union[List, Dict], 
                                   player_name: Optional[str] = None,
                                   prop_type: str = 'player_points') -> pd.DataFrame:
        """Parse player props response into DataFrame."""
        if not data:
            return pd.DataFrame()
        
        props_list = []
        
        # Handle both list and dict responses
        if isinstance(data, dict):
            games = data.get('data', data.get('events', []))
        else:
            games = data
        
        for game in games:
            if not isinstance(game, dict):
                continue
            
            game_info = {
                'game_id': game.get('id', ''),
                'home_team': game.get('home_team', ''),
                'away_team': game.get('away_team', ''),
                'commence_time': game.get('commence_time', '')
            }
            
            # Parse player props from bookmakers
            for bookmaker in game.get('bookmakers', []):
                bookmaker_name = bookmaker.get('title', 'unknown')
                
                for market in bookmaker.get('markets', []):
                    market_key = market.get('key', '')
                    
                    # Only process player prop markets
                    if 'player' not in market_key.lower():
                        continue
                    
                    for outcome in market.get('outcomes', []):
                        player = outcome.get('description', '')
                        over_under = outcome.get('name', '')  # 'Over' or 'Under'
                        odds = outcome.get('price', 0)
                        point = outcome.get('point', 0)
                        
                        # Filter by player name if specified
                        if player_name and player_name.lower() not in player.lower():
                            continue
                        
                        # Filter by prop type if specified
                        if prop_type and prop_type not in market_key:
                            continue
                        
                        props_list.append({
                            **game_info,
                            'bookmaker': bookmaker_name,
                            'market': market_key,
                            'player': player,
                            'bet_type': over_under,
                            'odds': odds,
                            'line': point,
                            'last_update': market.get('last_update', '')
                        })
        
        if props_list:
            df = pd.DataFrame(props_list)
            # Convert timestamps
            if 'commence_time' in df.columns:
                df['commence_time'] = pd.to_datetime(df['commence_time'], errors='coerce')
            if 'last_update' in df.columns:
                df['last_update'] = pd.to_datetime(df['last_update'], errors='coerce')
            return df
        
        return pd.DataFrame()
    
    def get_market_averages(self, 
                           sport: str,
                           market: str = 'h2h',
                           min_bookmakers: int = 3) -> pd.DataFrame:
        """
        Calculate average odds across bookmakers for each game.
        
        Args:
            sport: Sport key
            market: Market type ('h2h', 'spreads', 'totals')
            min_bookmakers: Minimum number of bookmakers required
            
        Returns:
            DataFrame with average odds
        """
        odds_df = self.get_odds(sport, markets=[market])
        
        if odds_df.empty:
            return pd.DataFrame()
        
        # Filter by market
        market_df = odds_df[odds_df['market'] == market.replace('h2h', 'moneyline')]
        
        # Group by game and team, calculate averages
        if market == 'h2h':
            # For moneyline, group by game and team
            avg_odds = (market_df.groupby(['game_id', 'team', 'team_name'])
                       .agg({
                           'odds': ['mean', 'std', 'count'],
                           'home_team': 'first',
                           'away_team': 'first',
                           'commence_time': 'first'
                       })
                       .reset_index())
            
            # Flatten column names
            avg_odds.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                               for col in avg_odds.columns]
            
            # Filter by minimum bookmakers
            avg_odds = avg_odds[avg_odds['odds_count'] >= min_bookmakers]
            
        else:
            # For spreads and totals, include point values
            avg_odds = (market_df.groupby(['game_id', 'team', 'team_name', 'point'])
                       .agg({
                           'odds': ['mean', 'std', 'count'],
                           'home_team': 'first',
                           'away_team': 'first',
                           'commence_time': 'first'
                       })
                       .reset_index())
            
            avg_odds.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                               for col in avg_odds.columns]
            avg_odds = avg_odds[avg_odds['odds_count'] >= min_bookmakers]
        
        return avg_odds
    
    def calculate_implied_probabilities(self, odds_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate implied probabilities from American odds.
        
        Args:
            odds_df: DataFrame with odds data
            
        Returns:
            DataFrame with implied probabilities added
        """
        df = odds_df.copy()
        
        if 'odds' not in df.columns:
            return df
        
        def american_to_probability(odds):
            if pd.isna(odds) or odds == 0:
                return None
            
            if odds > 0:
                # Positive odds: +150 means bet $100 to win $150
                return 100 / (odds + 100)
            else:
                # Negative odds: -150 means bet $150 to win $100
                return abs(odds) / (abs(odds) + 100)
        
        df['implied_probability'] = df['odds'].apply(american_to_probability)
        
        # Calculate vig (bookmaker margin) for moneyline bets
        if 'market' in df.columns:
            moneyline_games = df[df['market'] == 'moneyline'].groupby('game_id')
            
            vig_data = []
            for game_id, game_data in moneyline_games:
                if len(game_data) >= 2:  # Need both home and away odds
                    total_prob = game_data['implied_probability'].sum()
                    vig = total_prob - 1.0 if total_prob > 1.0 else 0.0
                    
                    vig_data.append({
                        'game_id': game_id,
                        'total_implied_prob': total_prob,
                        'vig': vig,
                        'fair_home_prob': None,
                        'fair_away_prob': None
                    })
                    
                    # Calculate fair probabilities (removing vig)
                    if vig > 0:
                        home_prob = game_data[game_data['team'] == 'home']['implied_probability'].iloc[0]
                        away_prob = game_data[game_data['team'] == 'away']['implied_probability'].iloc[0]
                        
                        vig_data[-1]['fair_home_prob'] = home_prob / total_prob
                        vig_data[-1]['fair_away_prob'] = away_prob / total_prob
            
            if vig_data:
                vig_df = pd.DataFrame(vig_data)
                df = df.merge(vig_df, on='game_id', how='left')
        
        return df
    
    def find_arbitrage_opportunities(self, sport: str) -> pd.DataFrame:
        """
        Find arbitrage betting opportunities.
        
        Args:
            sport: Sport key
            
        Returns:
            DataFrame with arbitrage opportunities
        """
        odds_df = self.get_odds(sport, markets=['h2h'])
        
        if odds_df.empty:
            return pd.DataFrame()
        
        # Filter moneyline bets only
        ml_df = odds_df[odds_df['market'] == 'moneyline']
        
        arbitrage_opps = []
        
        # Group by game
        for game_id, game_data in ml_df.groupby('game_id'):
            home_odds = game_data[game_data['team'] == 'home']
            away_odds = game_data[game_data['team'] == 'away']
            
            if home_odds.empty or away_odds.empty:
                continue
            
            # Find best odds for each team
            best_home = home_odds.loc[home_odds['odds'].idxmax()]
            best_away = away_odds.loc[away_odds['odds'].idxmax()]
            
            # Calculate implied probabilities
            home_prob = 100 / (best_home['odds'] + 100) if best_home['odds'] > 0 else abs(best_home['odds']) / (abs(best_home['odds']) + 100)
            away_prob = 100 / (best_away['odds'] + 100) if best_away['odds'] > 0 else abs(best_away['odds']) / (abs(best_away['odds']) + 100)
            
            total_prob = home_prob + away_prob
            
            # Arbitrage exists if total probability < 1.0
            if total_prob < 1.0:
                profit_margin = (1.0 - total_prob) * 100
                
                arbitrage_opps.append({
                    'game_id': game_id,
                    'home_team': best_home['home_team'],
                    'away_team': best_home['away_team'],
                    'home_odds': best_home['odds'],
                    'away_odds': best_away['odds'],
                    'home_bookmaker': best_home['bookmaker'],
                    'away_bookmaker': best_away['bookmaker'],
                    'home_stake_pct': home_prob / total_prob * 100,
                    'away_stake_pct': away_prob / total_prob * 100,
                    'profit_margin': profit_margin,
                    'commence_time': best_home['commence_time']
                })
        
        return pd.DataFrame(arbitrage_opps)
    
    def get_api_usage(self) -> Dict[str, Any]:
        """
        Get current API usage statistics.
        
        Returns:
            Dictionary with usage information
        """
        # Make a simple request to check headers
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
            logger.error(f"Error checking API usage: {e}")
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
