import requests
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from loguru import logger

from config.settings import Settings
from utils.api_helpers import APIHelper, APIError, create_api_sports_headers, get_all_pages

class SportsAPIClient:
    """Client for fetching sports data from API-Sports directly."""
    
    def __init__(self, sport: str):
        """
        Initialize API client for a specific sport.
        
        Args:
            sport: Sport type ('nba', 'mlb', 'nfl')
        """
        self.sport = sport.lower()
        
        # Set correct base URLs based on working discoveries
        if self.sport == 'nba':
            # NBA uses Basketball API, not dedicated NBA API
            self.base_url = "https://v1.basketball.api-sports.io"
            self.api_type = 'basketball'
        elif self.sport == 'mlb':
            self.base_url = "https://v1.baseball.api-sports.io" 
            self.api_type = 'baseball'
        elif self.sport == 'nfl':
            self.base_url = "https://v1.american-football.api-sports.io"
            self.api_type = 'american-football'
        else:
            raise ValueError(f"Unsupported sport: {sport}")
        
        # Create headers for direct API-Sports access
        self.headers = create_api_sports_headers()
        self.api_helper = APIHelper(Settings.API_RATE_LIMITS.get('api_sports', Settings.API_RATE_LIMITS['rapidapi']))
        
        # Set correct league IDs and season formats
        self.default_league_id = self._get_default_league_id()
        self.season_format = self._get_season_format()
        
        logger.info(f"✅ Initialized {self.sport.upper()} API client - API: {self.api_type}, League: {self.default_league_id}")
    
    def _get_default_league_id(self) -> Optional[int]:
        """Get default league ID for the sport based on working discoveries."""
        league_ids = {
            'nba': 12,    # NBA is league 12 in Basketball API ✅
            'mlb': 1,     # MLB works with league=1 ✅  
            'nfl': 1      # NFL works with league=1 ✅
        }
        return league_ids.get(self.sport, 1)
    
    def _get_season_format(self) -> str:
        """Get season format for the sport."""
        formats = {
            'nba': 'hyphenated',  # "2023-2024" format
            'mlb': 'year',        # 2024 format
            'nfl': 'year'         # 2024 format
        }
        return formats.get(self.sport, 'year')
    
    def _format_season(self, season: Union[int, str]) -> Union[str, int]:
        """Format season according to sport requirements with improved error handling."""
        if self.season_format == 'hyphenated':
            # NBA: Convert 2024 -> "2023-2024" 
            # Handle both int and string inputs
            if isinstance(season, str):
                # If already formatted like "2023-2024", return as-is
                if '-' in season:
                    return season
                # If string number like "2024", convert to int first
                try:
                    season = int(season)
                except ValueError:
                    logger.warning(f"Cannot convert season '{season}' to int, returning as-is")
                    return season  # Return as-is if can't convert
            
            return f"{season-1}-{season}"
        else:
            # MLB/NFL: Use year as-is, convert to int if string
            if isinstance(season, str):
                try:
                    return int(season)
                except ValueError:
                    logger.warning(f"Cannot convert season '{season}' to int, returning as-is")
                    return season
            return season
    
    def get_leagues(self, season: Optional[int] = None) -> pd.DataFrame:
        """
        Get available leagues for the sport.
        
        Args:
            season: Optional season year
            
        Returns:
            DataFrame with league information
        """
        url = f"{self.base_url}/leagues"
        params = {}
        
        if season:
            params['season'] = self._format_season(season)
        
        try:
            data = self.api_helper.make_request(url, self.headers, params)
            return self._parse_leagues_response(data)
        except APIError as e:
            logger.error(f"Error fetching leagues: {e}")
            return pd.DataFrame()
    
    def get_teams(self, 
                  league_id: Optional[int] = None,
                  season: Optional[int] = None,
                  team_id: Optional[int] = None) -> pd.DataFrame:
        """
        Get teams for the sport.
        
        Args:
            league_id: Optional league ID filter
            season: Optional season filter
            team_id: Optional specific team ID
            
        Returns:
            DataFrame with team information
        """
        url = f"{self.base_url}/teams"
        params = {}
        
        # Use working parameter logic
        if self.sport == 'nba':
            # NBA: Use Basketball API with league 12 and hyphenated season
            if not team_id:  # Only add league for general team queries
                params['league'] = league_id or self.default_league_id
            if season:
                params['season'] = self._format_season(season)
            if team_id:
                params['id'] = team_id
                
        elif self.sport == 'mlb':
            # MLB: Use league 1 and year season format
            if not team_id:
                params['league'] = league_id or self.default_league_id
            if season:
                params['season'] = self._format_season(season)
            if team_id:
                params['id'] = team_id
                
        elif self.sport == 'nfl':
            # NFL: Use league 1 and year season format  
            if not team_id:
                params['league'] = league_id or self.default_league_id
            if season:
                params['season'] = self._format_season(season)
            if team_id:
                params['id'] = team_id
        
        try:
            data = self.api_helper.make_request(url, self.headers, params)
            return self._parse_teams_response(data)
        except APIError as e:
            logger.error(f"Error fetching teams: {e}")
            return pd.DataFrame()
    
    def get_standings(self,
                     league_id: Optional[int] = None,
                     season: Optional[int] = None,
                     group: Optional[str] = None) -> pd.DataFrame:
        """
        Get league standings using working logic.
        
        Args:
            league_id: League ID (uses default if not specified)
            season: Season year
            group: Optional group/conference filter
            
        Returns:
            DataFrame with standings
        """
        url = f"{self.base_url}/standings"
        params = {}
        
        # Use the exact working parameters
        if self.sport == 'nba':
            # NBA: Basketball API with league 12 and hyphenated season
            params['league'] = league_id or self.default_league_id
            if season:
                params['season'] = self._format_season(season)
                
        elif self.sport == 'mlb':
            # MLB: league 1 and year season
            params['league'] = league_id or self.default_league_id
            if season:
                params['season'] = self._format_season(season)
                
        elif self.sport == 'nfl':
            # NFL: league 1 and year season
            params['league'] = league_id or self.default_league_id
            if season:
                params['season'] = self._format_season(season)
        
        if group:
            params['group'] = group
        
        try:
            data = self.api_helper.make_request(url, self.headers, params)
            return self._parse_standings_response(data)
        except APIError as e:
            logger.error(f"Error fetching standings: {e}")
            return pd.DataFrame()
    
    def get_team_statistics(self,
                           team_id: int,
                           season: Union[int, str],
                           league_id: Optional[int] = None) -> pd.DataFrame:
        """
        Get team statistics for a season with improved parameter handling.
        
        Args:
            team_id: Team ID
            season: Season year or string
            league_id: Optional league ID
            
        Returns:
            DataFrame with team statistics
        """
        url = f"{self.base_url}/teams/statistics"
        
        # Format season properly
        formatted_season = self._format_season(season)
        
        params = {
            'team': team_id,
            'season': formatted_season,
            'league': league_id or self.default_league_id
        }
        
        try:
            data = self.api_helper.make_request(url, self.headers, params)
            
            # Check for error responses
            if isinstance(data, dict) and 'error' in data:
                logger.warning(f"API error for team stats: {data['error']}")
                return pd.DataFrame()
            
            return self._parse_team_stats_response(data, team_id, season)
        except APIError as e:
            logger.error(f"Error fetching team statistics: {e}")
            return pd.DataFrame()
    
    def get_players(self,
                   team_id: Optional[int] = None,
                   season: Optional[Union[int, str]] = None,
                   player_id: Optional[int] = None,
                   search: Optional[str] = None,
                   page: int = 1) -> pd.DataFrame:
        """
        Get players with improved parameter handling and pagination.
        
        Args:
            team_id: Optional team filter
            season: Optional season filter
            player_id: Optional specific player ID
            search: Optional player name search
            page: Page number for pagination
            
        Returns:
            DataFrame with player information
        """
        url = f"{self.base_url}/players"
        params = {}
        
        # Try different parameter combinations based on what works for each sport
        if player_id:
            params['id'] = player_id
        elif search:
            params['search'] = search
        else:
            # Sport-specific parameter strategies
            if self.sport == 'nba':
                # NBA Basketball API approach
                if team_id:
                    params['team'] = team_id
                if season:
                    params['season'] = self._format_season(season)
                # Add league for NBA
                params['league'] = self.default_league_id
                
            elif self.sport == 'mlb':
                # MLB approach - try without league first
                if team_id:
                    params['team'] = team_id
                if season:
                    params['season'] = self._format_season(season)
                
            elif self.sport == 'nfl':
                # NFL approach
                if team_id:
                    params['team'] = team_id
                if season:
                    params['season'] = self._format_season(season)
                params['league'] = self.default_league_id
        
        # Add pagination
        if page > 1:
            params['page'] = page
        
        try:
            data = self.api_helper.make_request(url, self.headers, params)
            
            # If no results and we used league parameter, try without it
            if (isinstance(data, dict) and 
                data.get('response', []) == [] and 
                'league' in params and 
                not player_id and not search):
                
                logger.debug(f"Retrying players request without league parameter")
                params_without_league = {k: v for k, v in params.items() if k != 'league'}
                data = self.api_helper.make_request(url, self.headers, params_without_league)
            
            return self._parse_players_response(data)
        except APIError as e:
            logger.error(f"Error fetching players: {e}")
            return pd.DataFrame()
    
    def get_player_statistics(self,
                             player_id: int,
                             season: Union[int, str],
                             team_id: Optional[int] = None,
                             league_id: Optional[int] = None) -> pd.DataFrame:
        """
        Get player statistics for a season.
        
        Args:
            player_id: Player ID
            season: Season year
            team_id: Optional team filter
            league_id: Optional league filter
            
        Returns:
            DataFrame with player statistics
        """
        url = f"{self.base_url}/players/statistics"
        params = {
            'player': player_id,
            'season': self._format_season(season)
        }
        
        # Add league parameter
        params['league'] = league_id or self.default_league_id
        
        if team_id:
            params['team'] = team_id
        
        try:
            data = self.api_helper.make_request(url, self.headers, params)
            return self._parse_player_stats_response(data, player_id, season)
        except APIError as e:
            logger.error(f"Error fetching player statistics: {e}")
            return pd.DataFrame()
    
    def get_games(self,
                  date: Optional[str] = None,
                  season: Optional[Union[int, str]] = None,
                  league_id: Optional[int] = None,
                  team_id: Optional[int] = None,
                  game_id: Optional[int] = None,
                  live: bool = False,
                  from_date: Optional[str] = None,
                  to_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get games with enhanced date range support.
        
        Args:
            date: Optional date filter (YYYY-MM-DD)
            season: Optional season filter
            league_id: Optional league filter
            team_id: Optional team filter
            game_id: Optional specific game ID
            live: Get live games only
            from_date: Start date for range
            to_date: End date for range
            
        Returns:
            DataFrame with game information
        """
        if live:
            url = f"{self.base_url}/games/live"
        else:
            url = f"{self.base_url}/games"
        
        params = {
            'league': league_id or self.default_league_id
        }
        
        # Handle date parameters
        if date:
            params['date'] = date
        elif from_date and to_date:
            params['from'] = from_date
            params['to'] = to_date
        
        if season:
            params['season'] = self._format_season(season)
        if team_id:
            params['team'] = team_id
        if game_id:
            params['id'] = game_id
        
        try:
            data = self.api_helper.make_request(url, self.headers, params)
            return self._parse_games_response(data)
        except APIError as e:
            logger.error(f"Error fetching games: {e}")
            return pd.DataFrame()
    
    def get_head_to_head(self,
                        team1_id: int,
                        team2_id: int,
                        limit: int = 10) -> pd.DataFrame:
        """
        Get head-to-head matchup history between two teams.
        
        Args:
            team1_id: First team ID
            team2_id: Second team ID
            limit: Maximum number of games to return
            
        Returns:
            DataFrame with H2H game history
        """
        url = f"{self.base_url}/games/h2h"
        params = {
            'h2h': f"{team1_id}-{team2_id}",
            'league': self.default_league_id
        }
        
        try:
            data = self.api_helper.make_request(url, self.headers, params)
            h2h_df = self._parse_games_response(data)
            
            # Limit results if specified
            if not h2h_df.empty and limit:
                h2h_df = h2h_df.head(limit)
            
            return h2h_df
        except APIError as e:
            logger.error(f"Error fetching H2H data: {e}")
            return pd.DataFrame()
    
    def get_injuries(self,
                    team_id: Optional[int] = None,
                    player_id: Optional[int] = None,
                    season: Optional[Union[int, str]] = None) -> pd.DataFrame:
        """
        Get injury information (if available for the sport).
        
        Args:
            team_id: Optional team filter
            player_id: Optional player filter
            season: Optional season filter
            
        Returns:
            DataFrame with injury information
        """
        # Check if sport supports injuries endpoint
        if self.sport not in ['nfl', 'nba']:
            logger.warning(f"Injuries endpoint not typically available for {self.sport}")
            return pd.DataFrame()
        
        url = f"{self.base_url}/injuries"
        params = {
            'league': self.default_league_id
        }
        
        if team_id:
            params['team'] = team_id
        if player_id:
            params['player'] = player_id
        if season:
            params['season'] = self._format_season(season)
        
        try:
            data = self.api_helper.make_request(url, self.headers, params)
            return self._parse_injuries_response(data)
        except APIError as e:
            logger.error(f"Error fetching injuries: {e}")
            return pd.DataFrame()
    
    def get_seasons(self) -> List[Union[int, str]]:
        """
        Get available seasons for the sport.
        
        Returns:
            List of available season years/strings
        """
        url = f"{self.base_url}/seasons"
        
        try:
            data = self.api_helper.make_request(url, self.headers, {})
            
            # Parse response based on structure
            if isinstance(data, dict) and 'response' in data:
                return data['response']
            elif isinstance(data, list):
                return data
            else:
                return []
        except APIError as e:
            logger.error(f"Error fetching seasons: {e}")
            return []
    
    def check_data_availability(self) -> Dict[str, Any]:
        """
        Check what data is actually available for this sport.
        
        Returns:
            Dictionary with data availability information
        """
        availability = {
            'sport': self.sport,
            'seasons_with_teams': [],
            'seasons_with_games': [],
            'seasons_with_stats': [],
            'current_season': None,
            'working_endpoints': []
        }
        
        try:
            # Get available seasons
            seasons = self.get_seasons()
            if not seasons:
                return availability
            
            # Test recent seasons for data availability
            test_seasons = seasons[-5:] if len(seasons) >= 5 else seasons
            
            for season in test_seasons:
                # Check teams
                teams = self.get_teams(season=season)
                if not teams.empty:
                    availability['seasons_with_teams'].append(season)
                    if 'teams' not in availability['working_endpoints']:
                        availability['working_endpoints'].append('teams')
                
                # Check games (try recent date)
                if season in [2023, 2024, '2023-2024', '2024-2025']:
                    games = self.get_games(season=season)
                    if not games.empty:
                        availability['seasons_with_games'].append(season)
                        if 'games' not in availability['working_endpoints']:
                            availability['working_endpoints'].append('games')
                        
                    # Check team stats with first team
                    if not teams.empty:
                        first_team_id = teams.iloc[0].get('team_id', teams.iloc[0].get('id'))
                        if first_team_id:
                            stats = self.get_team_statistics(first_team_id, season)
                            if not stats.empty:
                                availability['seasons_with_stats'].append(season)
                                if 'team_stats' not in availability['working_endpoints']:
                                    availability['working_endpoints'].append('team_stats')
            
            # Check standings (usually works)
            standings = self.get_standings()
            if not standings.empty:
                availability['working_endpoints'].append('standings')
            
            # Determine current season
            current_year = datetime.now().year
            if self.sport == 'nba':
                # NBA season spans years
                current_candidates = [f"{current_year-1}-{current_year}", current_year-1, current_year]
            else:
                current_candidates = [current_year, current_year-1]
            
            for candidate in current_candidates:
                if candidate in availability['seasons_with_teams']:
                    availability['current_season'] = candidate
                    break
            
            return availability
            
        except Exception as e:
            logger.error(f"Error checking data availability: {e}")
            availability['error'] = str(e)
            return availability
    
    def get_all_team_stats_for_season(self,
                                     season: Union[int, str],
                                     league_id: Optional[int] = None) -> pd.DataFrame:
        """
        Get statistics for all teams in a season.
        
        Args:
            season: Season year
            league_id: Optional league ID
            
        Returns:
            DataFrame with all team statistics
        """
        # First get all teams
        teams_df = self.get_teams(league_id=league_id, season=season)
        
        if teams_df.empty:
            logger.warning(f"No teams found for {self.sport} season {season}")
            return pd.DataFrame()
        
        all_stats = []
        logger.info(f"Fetching stats for {len(teams_df)} teams...")
        
        for i, (_, team) in enumerate(teams_df.iterrows()):
            team_id = team.get('team_id', team.get('id'))
            team_name = team.get('name', 'Unknown')
            
            if team_id:
                try:
                    stats_df = self.get_team_statistics(team_id, season, league_id)
                    if not stats_df.empty:
                        all_stats.append(stats_df)
                        logger.debug(f"✅ Got stats for {team_name}")
                    else:
                        logger.debug(f"⚠️ No stats for {team_name}")
                except Exception as e:
                    logger.warning(f"❌ Error getting stats for {team_name}: {e}")
            
            # Progress update
            if (i + 1) % 5 == 0:
                logger.info(f"Progress: {i + 1}/{len(teams_df)} teams processed")
        
        if all_stats:
            combined_df = pd.concat(all_stats, ignore_index=True)
            logger.info(f"✅ Combined stats for {len(all_stats)} teams")
            return combined_df
        else:
            logger.warning("No team statistics found")
            return pd.DataFrame()
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test API connection and return status.
        
        Returns:
            Dictionary with connection test results
        """
        try:
            # Test with seasons endpoint (usually lightweight)
            seasons = self.get_seasons()
            
            # Test with teams endpoint - use recent season for testing
            test_season = 2024 if self.sport != 'nba' else 2024
            teams_df = self.get_teams(season=test_season)
            
            return {
                'status': 'success',
                'api_reachable': True,
                'seasons_available': len(seasons) if seasons else 0,
                'teams_found': len(teams_df),
                'sport': self.sport,
                'base_url': self.base_url,
                'default_league_id': self.default_league_id,
                'season_format': self.season_format,
                'api_type': self.api_type
            }
        except Exception as e:
            return {
                'status': 'error',
                'api_reachable': False,
                'error': str(e),
                'sport': self.sport,
                'base_url': self.base_url
            }
    
    # PARSING METHODS WITH IMPROVED NBA HANDLING
    
    def _parse_nba_basketball_response(self, data: Dict[str, Any], data_type: str) -> pd.DataFrame:
        """Improved NBA Basketball API response parsing based on working logic."""
        if not data or 'response' not in data:
            return pd.DataFrame()
        
        response = data['response']
        if not response:
            return pd.DataFrame()
        
        if data_type == 'standings':
            return self._parse_nba_standings_fixed(response)
        elif data_type == 'games':
            return self._parse_nba_games_fixed(response)
        elif data_type == 'teams':
            return self._parse_nba_teams_fixed(response)
        else:
            return pd.DataFrame()

    def _parse_nba_standings_fixed(self, response: List) -> pd.DataFrame:
        """Parse NBA standings using Basketball API format with improved error handling."""
        # Handle string responses (error messages)
        if isinstance(response, str):
            logger.warning(f"NBA standings API returned string response: {response}")
            return pd.DataFrame()
        
        # Handle None responses
        if response is None:
            logger.warning(f"NBA standings API returned None response")
            return pd.DataFrame()
        
        # Handle non-list responses
        if not isinstance(response, list):
            logger.warning(f"NBA standings response is not a list: {type(response)}")
            return pd.DataFrame()
        
        standings_list = []
        
        for standing_group in response:
            if isinstance(standing_group, list):
                # List of teams (conference/division)
                for team_standing in standing_group:
                    if isinstance(team_standing, dict):
                        # Extract data using Basketball API structure - safely
                        team_data = team_standing.get('team', {}) if isinstance(team_standing.get('team'), dict) else {}
                        group_data = team_standing.get('group', {}) if isinstance(team_standing.get('group'), dict) else {}
                        
                        # Extract wins/losses from games structure
                        wins = 'N/A'
                        losses = 'N/A'
                        
                        if 'games' in team_standing:
                            games = team_standing['games']
                            if isinstance(games, dict):
                                if 'win' in games:
                                    wins = games['win'].get('total', games['win']) if isinstance(games['win'], dict) else games['win']
                                if 'lose' in games:
                                    losses = games['lose'].get('total', games['lose']) if isinstance(games['lose'], dict) else games['lose']
                        
                        # Fallback to direct fields
                        if wins == 'N/A':
                            wins = team_standing.get('wins', team_standing.get('won', 'N/A'))
                        if losses == 'N/A':
                            losses = team_standing.get('losses', team_standing.get('lost', 'N/A'))
                        
                        standings_list.append({
                            'team_id': team_data.get('id') if team_data else None,
                            'team_name': team_data.get('name') if team_data else None,
                            'position': team_standing.get('position'),
                            'group': group_data.get('name', 'Conference') if group_data else 'Conference',
                            'wins': wins,
                            'losses': losses
                        })
                    elif isinstance(team_standing, str):
                        logger.warning(f"Skipping string standing entry: {team_standing}")
                    else:
                        logger.warning(f"Skipping unexpected standing type: {type(team_standing)}")
            elif isinstance(standing_group, str):
                logger.warning(f"Skipping string standing group: {standing_group}")
            else:
                logger.warning(f"Skipping unexpected standing group type: {type(standing_group)}")
        
        return pd.DataFrame(standings_list)
    
    def _parse_nba_games_fixed(self, response: List) -> pd.DataFrame:
        """Parse NBA games using Basketball API format with improved error handling."""
        # Handle string responses (error messages)
        if isinstance(response, str):
            logger.warning(f"NBA games API returned string response: {response}")
            return pd.DataFrame()
        
        # Handle None responses
        if response is None:
            logger.warning(f"NBA games API returned None response")
            return pd.DataFrame()
        
        # Handle non-list responses
        if not isinstance(response, list):
            logger.warning(f"NBA games response is not a list: {type(response)}")
            return pd.DataFrame()
        
        games_list = []
        
        for game in response:
            if isinstance(game, dict):
                # Handle Basketball API game structure - safely extract nested data
                teams = game.get('teams', {}) if isinstance(game.get('teams'), dict) else {}
                scores = game.get('scores', {}) if isinstance(game.get('scores'), dict) else {}
                status = game.get('status', {}) if isinstance(game.get('status'), dict) else {}
                venue = game.get('venue', {}) if isinstance(game.get('venue'), dict) else {}
                league = game.get('league', {}) if isinstance(game.get('league'), dict) else {}
                
                games_list.append({
                    'game_id': game.get('id'),
                    'date': game.get('date'),
                    'time': game.get('time'),
                    'timestamp': game.get('timestamp'),
                    'status': status.get('long', status.get('short', 'Unknown')) if status else 'Unknown',
                    'home_team_id': teams.get('home', {}).get('id') if teams.get('home') and isinstance(teams.get('home'), dict) else None,
                    'home_team_name': teams.get('home', {}).get('name') if teams.get('home') and isinstance(teams.get('home'), dict) else None,
                    'away_team_id': teams.get('away', {}).get('id') if teams.get('away') and isinstance(teams.get('away'), dict) else None,
                    'away_team_name': teams.get('away', {}).get('name') if teams.get('away') and isinstance(teams.get('away'), dict) else None,
                    'home_score': scores.get('home', {}).get('total') if scores.get('home') and isinstance(scores.get('home'), dict) else None,
                    'away_score': scores.get('away', {}).get('total') if scores.get('away') and isinstance(scores.get('away'), dict) else None,
                    'league_id': league.get('id') if league else None,
                    'season': league.get('season') if league else None,
                    'venue': venue.get('name') if venue else None,
                    'city': venue.get('city') if venue else None
                })
            elif isinstance(game, str):
                logger.warning(f"Skipping string game entry: {game}")
            else:
                logger.warning(f"Skipping unexpected game type: {type(game)}")
        
        return pd.DataFrame(games_list)

    def _parse_nba_teams_fixed(self, response: List) -> pd.DataFrame:
        """Parse NBA teams using Basketball API format with improved error handling."""
        # Handle string responses (error messages)
        if isinstance(response, str):
            logger.warning(f"NBA teams API returned string response: {response}")
            return pd.DataFrame()
        
        # Handle None responses
        if response is None:
            logger.warning(f"NBA teams API returned None response")
            return pd.DataFrame()
        
        # Handle non-list responses
        if not isinstance(response, list):
            logger.warning(f"NBA teams response is not a list: {type(response)}")
            return pd.DataFrame()
        
        teams_list = []
        
        for team in response:
            if isinstance(team, dict):
                teams_list.append({
                    'team_id': team.get('id'),
                    'name': team.get('name'),
                    'code': team.get('code'),
                    'logo': team.get('logo'),
                    'city': team.get('city'),
                    'country': team.get('country'),
                    'founded': team.get('founded'),
                    'national': team.get('national', False)
                })
            elif isinstance(team, str):
                logger.warning(f"Skipping string team entry: {team}")
        
        return pd.DataFrame(teams_list)
    
    def _parse_standings_response(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Parse standings API response with improved NBA handling."""
        if not data or 'response' not in data:
            return pd.DataFrame()
        
        response = data['response']
        if not response:
            return pd.DataFrame()
        
        # Handle string responses (error messages)
        if isinstance(response, str):
            logger.warning(f"Standings API returned string response: {response}")
            return pd.DataFrame()
        
        # Use improved parsing for NBA (Basketball API)
        if self.sport == 'nba':
            return self._parse_nba_standings_fixed(response)
        
        standings_list = []
        
        # Handle different response structures by sport
        if self.sport == 'nfl':
            # NFL: Direct list of team standings
            for team_standing in response:
                if isinstance(team_standing, dict):
                    standings_list.append({
                        'team_id': team_standing.get('team', {}).get('id'),
                        'team_name': team_standing.get('team', {}).get('name'),
                        'position': team_standing.get('position'),
                        'conference': team_standing.get('conference'),
                        'division': team_standing.get('division'),
                        'wins': team_standing.get('won'),
                        'losses': team_standing.get('lost'),
                        'ties': team_standing.get('ties', 0)
                    })
                
        elif self.sport == 'mlb':
            # MLB: List of lists (divisions)
            for standing_group in response:
                if isinstance(standing_group, list):
                    for team_standing in standing_group:
                        if isinstance(team_standing, dict):
                            # Extract wins/losses from games.win.total structure
                            wins = 'N/A'
                            losses = 'N/A'
                            
                            if 'games' in team_standing:
                                games = team_standing['games']
                                if isinstance(games, dict):
                                    if 'win' in games and isinstance(games['win'], dict):
                                        wins = games['win'].get('total', 'N/A')
                                    if 'lose' in games and isinstance(games['lose'], dict):
                                        losses = games['lose'].get('total', 'N/A')
                            
                            standings_list.append({
                                'team_id': team_standing.get('team', {}).get('id'),
                                'team_name': team_standing.get('team', {}).get('name'),
                                'position': team_standing.get('position'),
                                'group': team_standing.get('group', {}).get('name'),
                                'wins': wins,
                                'losses': losses
                            })
        
        return pd.DataFrame(standings_list)
    
    def _parse_leagues_response(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Parse leagues API response."""
        if not data or 'response' not in data:
            return pd.DataFrame()
        
        leagues = data['response']
        if not leagues:
            return pd.DataFrame()
        
        leagues_list = []
        for league in leagues:
            if isinstance(league, dict):
                leagues_list.append({
                    'league_id': league.get('id'),
                    'name': league.get('name'),
                    'type': league.get('type'),
                    'logo': league.get('logo'),
                    'country': league.get('country', {}).get('name'),
                    'country_code': league.get('country', {}).get('code')
                })
        
        return pd.DataFrame(leagues_list)
    
    def _parse_teams_response(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Parse teams API response with improved NBA handling."""
        if not data or 'response' not in data:
            return pd.DataFrame()
        
        teams = data['response']
        if not teams:
            return pd.DataFrame()
        
        # Handle string responses (error messages)
        if isinstance(teams, str):
            logger.warning(f"Teams API returned string response: {teams}")
            return pd.DataFrame()
        
        # Use improved parsing for NBA (Basketball API)
        if self.sport == 'nba':
            return self._parse_nba_teams_fixed(teams)
        
        teams_list = []
        for team in teams:
            if isinstance(team, dict):
                teams_list.append({
                    'team_id': team.get('id'),
                    'name': team.get('name'),
                    'code': team.get('code'),
                    'logo': team.get('logo'),
                    'city': team.get('city'),
                    'country': team.get('country'),
                    'founded': team.get('founded'),
                    'national': team.get('national', False)
                })
        
        return pd.DataFrame(teams_list)
    
    def _parse_team_stats_response(self, data: Dict[str, Any], team_id: int, season: Union[int, str]) -> pd.DataFrame:
        """Parse team statistics response with better error handling."""
        if not data or 'response' not in data:
            return pd.DataFrame()
        
        response = data['response']
        if not response:
            return pd.DataFrame()
        
        # Handle string responses (error messages)
        if isinstance(response, str):
            logger.warning(f"Team stats API returned string response: {response}")
            return pd.DataFrame()
        
        # Handle different response structures by sport
        try:
            if self.sport == 'nba':
                return self._parse_nba_team_stats(response, team_id, season)
            elif self.sport == 'mlb':
                return self._parse_mlb_team_stats(response, team_id, season)
            elif self.sport == 'nfl':
                return self._parse_nfl_team_stats(response, team_id, season)
            else:
                # Generic parsing with error handling
                stats_list = []
                if isinstance(response, list):
                    for stat in response:
                        if isinstance(stat, dict):
                            stat_data = {
                                'team_id': team_id,
                                'season': season,
                                **stat
                            }
                            stats_list.append(stat_data)
                return pd.DataFrame(stats_list)
        except Exception as e:
            logger.error(f"Error parsing team stats for {self.sport}: {e}")
            return pd.DataFrame()
    
    def _parse_nba_team_stats(self, response: List[Dict], team_id: int, season: Union[int, str]) -> pd.DataFrame:
        """Parse NBA-specific team stats with error handling."""
        if not response or not isinstance(response, list):
            return pd.DataFrame()
        
        # NBA stats typically nested under 'statistics'
        for team_data in response:
            if isinstance(team_data, dict) and team_data.get('team', {}).get('id') == team_id:
                stats = team_data.get('statistics', [])
                if stats and isinstance(stats, list):
                    stat_data = {
                        'team_id': team_id,
                        'season': season,
                        **stats[0]  # Usually one stats record per team
                    }
                    return pd.DataFrame([stat_data])
        
        return pd.DataFrame()
    
    def _parse_mlb_team_stats(self, response: List[Dict], team_id: int, season: Union[int, str]) -> pd.DataFrame:
        """Parse MLB-specific team stats."""
        # Similar structure to NBA, adapt as needed
        return self._parse_nba_team_stats(response, team_id, season)
    
    def _parse_nfl_team_stats(self, response: List[Dict], team_id: int, season: Union[int, str]) -> pd.DataFrame:
        """Parse NFL-specific team stats."""
        # Similar structure, adapt as needed
        return self._parse_nba_team_stats(response, team_id, season)
    
    def _parse_players_response(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Parse players API response with better error handling."""
        if not data or 'response' not in data:
            return pd.DataFrame()
        
        players = data['response']
        if not players:
            return pd.DataFrame()
        
        # Handle error responses
        if isinstance(players, str):
            logger.warning(f"Players API returned string: {players}")
            return pd.DataFrame()
        
        if not isinstance(players, list):
            logger.warning(f"Players response is not a list: {type(players)}")
            return pd.DataFrame()
        
        players_list = []
        for player in players:
            if isinstance(player, dict):
                players_list.append({
                    'player_id': player.get('id'),
                    'name': player.get('name'),
                    'firstname': player.get('firstname'),
                    'lastname': player.get('lastname'),
                    'age': player.get('age'),
                    'height': player.get('height'),
                    'weight': player.get('weight'),
                    'country': player.get('country'),
                    'position': player.get('position'),
                    'photo': player.get('photo')
                })
        
        return pd.DataFrame(players_list)
    
    def _parse_player_stats_response(self, data: Dict[str, Any], player_id: int, season: Union[int, str]) -> pd.DataFrame:
        """Parse player statistics response."""
        if not data or 'response' not in data:
            return pd.DataFrame()
        
        response = data['response']
        if not response:
            return pd.DataFrame()
        
        stats_list = []
        for player_data in response:
            if isinstance(player_data, dict) and player_data.get('player', {}).get('id') == player_id:
                statistics = player_data.get('statistics', [])
                for stat in statistics:
                    if isinstance(stat, dict):
                        stat_data = {
                            'player_id': player_id,
                            'season': season,
                            'team_id': stat.get('team', {}).get('id'),
                            **stat
                        }
                        stats_list.append(stat_data)
        
        return pd.DataFrame(stats_list)
    
    def _parse_games_response(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Parse games API response with improved NBA handling."""
        if not data or 'response' not in data:
            return pd.DataFrame()
        
        games = data['response']
        if not games:
            return pd.DataFrame()
        
        # Handle string responses (error messages)
        if isinstance(games, str):
            logger.warning(f"Games API returned string response: {games}")
            return pd.DataFrame()
        
        # Use improved parsing for NBA (Basketball API)
        if self.sport == 'nba':
            return self._parse_nba_games_fixed(games)
        
        games_list = []
        for game in games:
            if isinstance(game, dict):
                games_list.append({
                    'game_id': game.get('id'),
                    'date': game.get('date'),
                    'time': game.get('time'),
                    'timestamp': game.get('timestamp'),
                    'status': game.get('status', {}).get('long'),
                    'home_team_id': game.get('teams', {}).get('home', {}).get('id'),
                    'home_team_name': game.get('teams', {}).get('home', {}).get('name'),
                    'away_team_id': game.get('teams', {}).get('away', {}).get('id'),
                    'away_team_name': game.get('teams', {}).get('away', {}).get('name'),
                    'home_score': game.get('scores', {}).get('home', {}).get('total'),
                    'away_score': game.get('scores', {}).get('away', {}).get('total'),
                    'league_id': game.get('league', {}).get('id'),
                    'season': game.get('league', {}).get('season'),
                    'venue': game.get('venue', {}).get('name'),
                    'city': game.get('venue', {}).get('city')
                })
        
        return pd.DataFrame(games_list)
    
    def _parse_injuries_response(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Parse injuries API response."""
        if not data or 'response' not in data:
            return pd.DataFrame()
        
        injuries = data['response']
        if not injuries:
            return pd.DataFrame()
        
        injuries_list = []
        for injury in injuries:
            if isinstance(injury, dict):
                injuries_list.append({
                    'player_id': injury.get('player', {}).get('id'),
                    'player_name': injury.get('player', {}).get('name'),
                    'team_id': injury.get('team', {}).get('id'),
                    'team_name': injury.get('team', {}).get('name'),
                    'type': injury.get('type'),
                    'reason': injury.get('reason'),
                    'date': injury.get('date')
                })
        
        return pd.DataFrame(injuries_list)
