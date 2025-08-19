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
        
        # FIXED: Set correct base URLs based on our working discoveries
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
        
        # FIXED: Set correct league IDs and season formats based on our working script
        self.default_league_id = self._get_default_league_id()
        self.season_format = self._get_season_format()
        
        logger.info(f"✅ Initialized {self.sport.upper()} API client - API: {self.api_type}, League: {self.default_league_id}")
    
    def _get_default_league_id(self) -> Optional[int]:
        """Get default league ID for the sport based on our working discoveries."""
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
    
    def _format_season(self, season: int) -> Union[str, int]:
        """Format season according to sport requirements."""
        if self.season_format == 'hyphenated':
            # NBA: Convert 2024 -> "2023-2024"
            return f"{season-1}-{season}"
        else:
            # MLB/NFL: Use year as-is
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
        
        # FIXED: Use working parameter logic from our script
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
        Get league standings using our working logic.
        
        Args:
            league_id: League ID (uses default if not specified)
            season: Season year
            group: Optional group/conference filter
            
        Returns:
            DataFrame with standings
        """
        url = f"{self.base_url}/standings"
        params = {}
        
        # FIXED: Use the exact working parameters from our script
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
                           season: int,
                           league_id: Optional[int] = None) -> pd.DataFrame:
        """
        Get team statistics for a season.
        
        Args:
            team_id: Team ID
            season: Season year
            league_id: Optional league ID
            
        Returns:
            DataFrame with team statistics
        """
        url = f"{self.base_url}/teams/statistics"
        
        params = {
            'team': team_id,
            'season': self._format_season(season)
        }
        
        # Add league parameter for all sports now that we know NBA works with it
        params['league'] = league_id or self.default_league_id
        
        try:
            data = self.api_helper.make_request(url, self.headers, params)
            return self._parse_team_stats_response(data, team_id, season)
        except APIError as e:
            logger.error(f"Error fetching team statistics: {e}")
            return pd.DataFrame()
    
    def get_players(self,
                   team_id: Optional[int] = None,
                   season: Optional[int] = None,
                   player_id: Optional[int] = None,
                   search: Optional[str] = None) -> pd.DataFrame:
        """
        Get players for the sport.
        
        Args:
            team_id: Optional team filter
            season: Optional season filter
            player_id: Optional specific player ID
            search: Optional player name search
            
        Returns:
            DataFrame with player information
        """
        url = f"{self.base_url}/players"
        params = {}
        
        if team_id:
            params['team'] = team_id
        if season:
            params['season'] = self._format_season(season)
        if player_id:
            params['id'] = player_id
        if search:
            params['search'] = search
        
        # Add league for consistency
        if not player_id and not search:  # Don't add league for specific lookups
            params['league'] = self.default_league_id
        
        try:
            data = self.api_helper.make_request(url, self.headers, params)
            return self._parse_players_response(data)
        except APIError as e:
            logger.error(f"Error fetching players: {e}")
            return pd.DataFrame()
    
    def get_player_statistics(self,
                             player_id: int,
                             season: int,
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
                  season: Optional[int] = None,
                  league_id: Optional[int] = None,
                  team_id: Optional[int] = None,
                  game_id: Optional[int] = None,
                  live: bool = False) -> pd.DataFrame:
        """
        Get games/schedule information.
        
        Args:
            date: Optional date filter (YYYY-MM-DD)
            season: Optional season filter
            league_id: Optional league filter
            team_id: Optional team filter
            game_id: Optional specific game ID
            live: Get live games only
            
        Returns:
            DataFrame with game information
        """
        if live:
            url = f"{self.base_url}/games/live"
        else:
            url = f"{self.base_url}/games"
        
        params = {}
        
        # Use consistent parameter logic
        params['league'] = league_id or self.default_league_id
        
        if date:
            params['date'] = date
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
                    season: Optional[int] = None) -> pd.DataFrame:
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
    
    def get_all_team_stats_for_season(self,
                                     season: int,
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
            test_season = 2024 if self.sport != 'nba' else 2023  # NBA uses 2023-2024
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
    
    # FIXED: Updated parsing methods to handle our working response structures
    def _parse_standings_response(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Parse standings API response using our working logic."""
        if not data or 'response' not in data:
            return pd.DataFrame()
        
        response = data['response']
        if not response:
            return pd.DataFrame()
        
        standings_list = []
        
        # Handle different response structures by sport
        if self.sport == 'nfl':
            # NFL: Direct list of team standings
            for team_standing in response:
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
                        
        elif self.sport == 'nba':
            # NBA: Similar to MLB structure from Basketball API
            for standing_group in response:
                if isinstance(standing_group, list):
                    for team_standing in standing_group:
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
                            'team_id': team_standing.get('team', {}).get('id'),
                            'team_name': team_standing.get('team', {}).get('name'),
                            'position': team_standing.get('position'),
                            'group': team_standing.get('group', {}).get('name'),
                            'wins': wins,
                            'losses': losses
                        })
        
        return pd.DataFrame(standings_list)
    
    # Keep existing parsing methods for other responses
    def _parse_leagues_response(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Parse leagues API response."""
        if not data or 'response' not in data:
            return pd.DataFrame()
        
        leagues = data['response']
        if not leagues:
            return pd.DataFrame()
        
        leagues_list = []
        for league in leagues:
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
        """Parse teams API response."""
        if not data or 'response' not in data:
            return pd.DataFrame()
        
        teams = data['response']
        if not teams:
            return pd.DataFrame()
        
        teams_list = []
        for team in teams:
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
    
    def _parse_team_stats_response(self, data: Dict[str, Any], team_id: int, season: int) -> pd.DataFrame:
        """Parse team statistics response."""
        if not data or 'response' not in data:
            return pd.DataFrame()
        
        response = data['response']
        if not response:
            return pd.DataFrame()
        
        # Handle different response structures by sport
        if self.sport == 'nba':
            return self._parse_nba_team_stats(response, team_id, season)
        elif self.sport == 'mlb':
            return self._parse_mlb_team_stats(response, team_id, season)
        elif self.sport == 'nfl':
            return self._parse_nfl_team_stats(response, team_id, season)
        else:
            # Generic parsing
            stats_list = []
            for stat in response:
                stat_data = {
                    'team_id': team_id,
                    'season': season,
                    **stat
                }
                stats_list.append(stat_data)
            return pd.DataFrame(stats_list)
    
    def _parse_nba_team_stats(self, response: List[Dict], team_id: int, season: int) -> pd.DataFrame:
        """Parse NBA-specific team stats."""
        if not response:
            return pd.DataFrame()
        
        # NBA stats typically nested under 'statistics'
        for team_data in response:
            if team_data.get('team', {}).get('id') == team_id:
                stats = team_data.get('statistics', [])
                if stats:
                    stat_data = {
                        'team_id': team_id,
                        'season': season,
                        **stats[0]  # Usually one stats record per team
                    }
                    return pd.DataFrame([stat_data])
        
        return pd.DataFrame()
    
    def _parse_mlb_team_stats(self, response: List[Dict], team_id: int, season: int) -> pd.DataFrame:
        """Parse MLB-specific team stats."""
        # Similar structure to NBA, adapt as needed
        return self._parse_nba_team_stats(response, team_id, season)
    
    def _parse_nfl_team_stats(self, response: List[Dict], team_id: int, season: int) -> pd.DataFrame:
        """Parse NFL-specific team stats."""
        # Similar structure, adapt as needed
        return self._parse_nba_team_stats(response, team_id, season)
    
    def _parse_players_response(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Parse players API response."""
        if not data or 'response' not in data:
            return pd.DataFrame()
        
        players = data['response']
        if not players:
            return pd.DataFrame()
        
        players_list = []
        for player in players:
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
    
    def _parse_player_stats_response(self, data: Dict[str, Any], player_id: int, season: int) -> pd.DataFrame:
        """Parse player statistics response."""
        if not data or 'response' not in data:
            return pd.DataFrame()
        
        response = data['response']
        if not response:
            return pd.DataFrame()
        
        stats_list = []
        for player_data in response:
            if player_data.get('player', {}).get('id') == player_id:
                statistics = player_data.get('statistics', [])
                for stat in statistics:
                    stat_data = {
                        'player_id': player_id,
                        'season': season,
                        'team_id': stat.get('team', {}).get('id'),
                        **stat
                    }
                    stats_list.append(stat_data)
        
        return pd.DataFrame(stats_list)
    
    def _parse_games_response(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Parse games API response."""
        if not data or 'response' not in data:
            return pd.DataFrame()
        
        games = data['response']
        if not games:
            return pd.DataFrame()
        
        games_list = []
        for game in games:
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
