# api_clients/espn_api.py
"""
ESPN API client for reliable sports data retrieval.
Based on ESPN's free, undocumented JSON endpoints.
Integrates with existing SportsAPIClient architecture.
"""

import os
import time
import json
import pandas as pd
import requests
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from loguru import logger


class ESPNAPIClient:
    """
    ESPN API client that matches the SportsAPIClient interface.
    """
    
    def __init__(self, sport: str = 'mlb'):
        """Initialize ESPN client for specified sport."""
        self.sport = sport.lower()
        
        # ESPN endpoints (no auth required)
        base_url = f"https://site.web.api.espn.com/apis/common/v3/sports"
        
        if self.sport == 'mlb':
            sport_path = "baseball/mlb"
        elif self.sport == 'nba':
            sport_path = "basketball/nba"
        elif self.sport == 'nfl':
            sport_path = "football/nfl"
        else:
            raise ValueError(f"Sport {sport} not supported by ESPN client")
        
        self.team_stats_url = f"{base_url}/{sport_path}/statistics/byteam"
        self.player_stats_url = f"{base_url}/{sport_path}/statistics/byathlete"
        self.teams_url = f"{base_url}/{sport_path}/teams"
        
        # Create session with proper headers
        self.session = self._create_session()
        
        logger.info(f"✅ ESPN API client initialized for {self.sport.upper()}")
    
    def _create_session(self) -> requests.Session:
        """Create HTTP session with proper headers."""
        s = requests.Session()
        s.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*"
        })
        return s
    
    def _get_json(self, url: str, params: Dict[str, Any], tries: int = 4, backoff: float = 0.75) -> Dict[str, Any]:
        """GET with retry/backoff for rate limiting."""
        for attempt in range(1, tries + 1):
            try:
                r = self.session.get(url, params=params, timeout=30)
                if r.status_code == 200:
                    return r.json()
                if r.status_code in (429, 500, 502, 503, 504):
                    sleep_for = backoff * (2 ** (attempt - 1))
                    logger.debug(f"Rate limited, sleeping {sleep_for}s...")
                    time.sleep(sleep_for)
                    continue
                r.raise_for_status()
            except Exception as e:
                if attempt == tries:
                    raise
                logger.debug(f"Request failed (attempt {attempt}): {e}")
                time.sleep(backoff)
        
        # Final attempt
        r = self.session.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    
    def _flatten_categories(self, categories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Flatten ESPN stats categories into a flat dictionary."""
        out: Dict[str, Any] = {}
        if not isinstance(categories, list):
            return out
        
        for cat in categories:
            cat_key = (
                cat.get("name") or 
                cat.get("abbreviation") or 
                cat.get("displayName") or 
                "category"
            ).strip().lower().replace(" ", "_")
            
            for stat in cat.get("stats", []):
                stat_key = (
                    stat.get("name") or 
                    stat.get("abbreviation") or 
                    stat.get("displayName") or 
                    "value"
                ).strip().replace(" ", "_")
                
                # Prefer numeric value, fallback to display value
                val = stat.get("value")
                if val is None:
                    val = stat.get("displayValue")
                
                out[f"{cat_key}.{stat_key}"] = val
        
        return out
    
    def _get_current_season(self) -> int:
        """Get current season based on sport calendar."""
        now = datetime.now()
        
        if self.sport == 'mlb':
            # MLB season runs March-October within same calendar year
            return now.year
        elif self.sport == 'nba':
            # NBA season runs October-June across calendar years
            if now.month >= 10:
                return now.year
            else:
                return now.year - 1
        elif self.sport == 'nfl':
            # NFL season runs September-February across calendar years
            if now.month >= 9:
                return now.year
            else:
                return now.year - 1
        else:
            return now.year
    
    # ============= MAIN API METHODS (match SportsAPIClient interface) =============
    
    def get_teams(self, season: Optional[int] = None, league_id: Optional[int] = None, **kwargs) -> pd.DataFrame:
        """
        Get teams data from ESPN.
        
        Args:
            season: Season year (defaults to current season)
            league_id: Ignored for ESPN (no league concept)
            **kwargs: Additional parameters for compatibility
            
        Returns:
            DataFrame with team information
        """
        if season is None:
            season = self._get_current_season()
        
        logger.debug(f"Fetching {self.sport.upper()} teams for season {season}")
        
        try:
            # First try getting teams from team stats endpoint
            teams_df = self._get_teams_from_stats(season)
            
            # If that fails, try direct teams endpoint
            if teams_df.empty:
                teams_df = self._get_teams_direct(season)
            
            if not teams_df.empty:
                logger.info(f"✅ Found {len(teams_df)} teams from ESPN")
            else:
                logger.warning(f"No teams found for {self.sport.upper()} season {season}")
            
            return teams_df
            
        except Exception as e:
            logger.error(f"Error fetching teams from ESPN: {e}")
            return pd.DataFrame()
    
    def _get_teams_from_stats(self, season: int, seasontype: int = 2) -> pd.DataFrame:
        """Get teams from team statistics endpoint."""
        params = {"season": season, "seasontype": seasontype}
        data = self._get_json(self.team_stats_url, params)
        
        rows: List[Dict[str, Any]] = []
        
        # ESPN can return different shapes
        container = data.get("splits") or data.get("results") or data.get("teams") or []
        if not isinstance(container, list):
            container = []
        
        for item in container:
            team = item.get("team") or item.get("entity") or {}
            team_info = {
                "team_id": team.get("id"),
                "name": team.get("displayName") or team.get("name"),
                "code": team.get("abbreviation") or team.get("shortDisplayName"),
                "city": self._extract_city(team.get("displayName") or ""),
                "country": "USA",  # ESPN is primarily US sports
                "logo": team.get("logo", ""),
                "uid": team.get("uid"),
                "season": season
            }
            if team_info["team_id"]:
                rows.append(team_info)
        
        return pd.DataFrame(rows)
    
    def _get_teams_direct(self, season: int) -> pd.DataFrame:
        """Get teams from direct teams endpoint."""
        try:
            params = {"season": season} if season else {}
            data = self._get_json(self.teams_url, params)
            
            rows: List[Dict[str, Any]] = []
            
            # Handle different response structures
            teams = data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
            if not teams:
                teams = data.get("teams", [])
            
            for team_data in teams:
                team = team_data.get("team", team_data)
                team_info = {
                    "team_id": team.get("id"),
                    "name": team.get("displayName") or team.get("name"),
                    "code": team.get("abbreviation"),
                    "city": self._extract_city(team.get("displayName") or ""),
                    "country": "USA",
                    "logo": team.get("logo", ""),
                    "uid": team.get("uid"),
                    "season": season
                }
                if team_info["team_id"]:
                    rows.append(team_info)
            
            return pd.DataFrame(rows)
            
        except Exception as e:
            logger.debug(f"Direct teams endpoint failed: {e}")
            return pd.DataFrame()
    
    def _extract_city(self, display_name: str) -> str:
        """Extract city from team display name."""
        if not display_name:
            return ""
        
        # Common patterns: "New York Yankees" -> "New York"
        words = display_name.split()
        if len(words) >= 2:
            # Take all but last word as city for most cases
            return " ".join(words[:-1])
        return ""
    
    def get_players(self, team_id: Optional[int] = None, season: Optional[int] = None, 
                   player_id: Optional[int] = None, search: Optional[str] = None, 
                   page: int = 1, **kwargs) -> pd.DataFrame:
        """
        Get players data from ESPN.
        
        Args:
            team_id: Optional team filter
            season: Season year (defaults to current season)
            player_id: Optional specific player ID (not supported by ESPN)
            search: Optional search term (not supported by ESPN)
            page: Page number (handled internally by ESPN)
            **kwargs: Additional parameters for compatibility
            
        Returns:
            DataFrame with player information and stats
        """
        if season is None:
            season = self._get_current_season()
        
        logger.debug(f"Fetching {self.sport.upper()} players for season {season}")
        
        try:
            # For MLB, try multiple categories to get comprehensive player list
            if self.sport == 'mlb':
                categories = ['batting', 'pitching']
            else:
                categories = ['general']  # For NBA/NFL, use general category
            
            all_players = []
            
            for cat in categories:
                players_df = self._get_players_by_category(season, cat)
                if not players_df.empty:
                    all_players.append(players_df)
            
            if all_players:
                # Combine and deduplicate players
                combined_df = pd.concat(all_players, ignore_index=True)
                
                # Remove duplicates based on player_id, keeping first occurrence
                combined_df = combined_df.drop_duplicates(subset=['player_id'], keep='first')
                
                # Filter by team if specified
                if team_id:
                    combined_df = combined_df[combined_df['team_id'] == str(team_id)]
                
                logger.info(f"✅ Found {len(combined_df)} players from ESPN")
                return combined_df
            else:
                logger.warning(f"No players found for {self.sport.upper()} season {season}")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching players from ESPN: {e}")
            return pd.DataFrame()
    
    def _get_players_by_category(self, season: int, category: str, 
                                seasontype: int = 2, limit: int = 5000) -> pd.DataFrame:
        """Get players for a specific category with pagination."""
        rows: List[Dict[str, Any]] = []
        page = 1
        
        while True:
            params = {
                "season": season,
                "year": season,
                "seasontype": seasontype,
                "category": category,
                "page": page,
                "limit": limit,
                "lang": "en",
                "region": "us",
                "contentorigin": "espn",
                "isqualified": "false",  # Get all players, not just qualified
            }
            
            try:
                data = self._get_json(self.player_stats_url, params)
                
                # Handle different response structures
                items = data.get("athletes") or data.get("results") or []
                if not items:
                    break
                
                for item in items:
                    athlete = item.get("athlete") or item.get("player") or {}
                    team = item.get("team") or athlete.get("team") or {}
                    
                    # Extract basic player info
                    player_info = {
                        "player_id": athlete.get("id"),
                        "name": athlete.get("displayName") or athlete.get("shortName"),
                        "firstname": athlete.get("firstName", ""),
                        "lastname": athlete.get("lastName", ""),
                        "age": athlete.get("age"),
                        "height": athlete.get("height"),
                        "weight": athlete.get("weight"),
                        "country": athlete.get("birthPlace", {}).get("country", ""),
                        "position": athlete.get("position", {}).get("abbreviation", ""),
                        "photo": athlete.get("headshot", {}).get("href", ""),
                        "team_id": team.get("id"),
                        "season": season,
                        "category": category
                    }
                    
                    # Split full name if first/last not available
                    if not player_info["firstname"] and player_info["name"]:
                        name_parts = player_info["name"].split()
                        if len(name_parts) >= 2:
                            player_info["firstname"] = name_parts[0]
                            player_info["lastname"] = " ".join(name_parts[1:])
                    
                    # Flatten stats categories
                    categories = item.get("categories") or data.get("categories") or []
                    stats = self._flatten_categories(categories)
                    
                    combined_row = {**player_info, **stats}
                    rows.append(combined_row)
                
                # Check if we got fewer items than limit (last page)
                if len(items) < limit:
                    break
                    
                page += 1
                time.sleep(0.1)  # Be nice to ESPN's servers
                
            except Exception as e:
                logger.debug(f"Error fetching page {page} for category {category}: {e}")
                break
        
        return pd.DataFrame(rows)
    
    def get_player_statistics(self, player_id: int, season: Union[int, str], 
                             team_id: Optional[int] = None, league_id: Optional[int] = None) -> pd.DataFrame:
        """
        Get specific player statistics.
        Note: ESPN endpoints are more suited for bulk data retrieval.
        """
        if isinstance(season, str):
            try:
                season = int(season)
            except ValueError:
                season = self._get_current_season()
        
        logger.debug(f"Fetching stats for player {player_id}")
        
        try:
            # For individual player stats, we need to get all players and filter
            # This is not as efficient as dedicated player stat endpoints
            categories = ['batting', 'pitching'] if self.sport == 'mlb' else ['general']
            
            for category in categories:
                players_df = self._get_players_by_category(season, category, limit=1000)
                
                if not players_df.empty:
                    player_stats = players_df[players_df['player_id'] == str(player_id)]
                    if not player_stats.empty:
                        return player_stats
            
            logger.warning(f"No stats found for player {player_id}")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching player statistics: {e}")
            return pd.DataFrame()
    
    def get_team_statistics(self, team_id: int, season: Union[int, str], 
                           league_id: Optional[int] = None) -> pd.DataFrame:
        """Get team statistics from ESPN."""
        if isinstance(season, str):
            try:
                season = int(season)
            except ValueError:
                season = self._get_current_season()
        
        try:
            teams_df = self._get_teams_from_stats(season)
            
            if not teams_df.empty:
                team_stats = teams_df[teams_df['team_id'] == str(team_id)]
                return team_stats
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching team statistics: {e}")
            return pd.DataFrame()
    
    def get_seasons(self) -> List[Union[int, str]]:
        """Get available seasons (last 5 years)."""
        current = self._get_current_season()
        return list(range(current - 4, current + 1))
    
    def test_connection(self) -> Dict[str, Any]:
        """Test ESPN API connection."""
        try:
            current_season = self._get_current_season()
            teams = self.get_teams(season=current_season)
            
            return {
                'status': 'success' if not teams.empty else 'warning',
                'message': f'Found {len(teams)} teams' if not teams.empty else 'No teams found',
                'season_tested': current_season,
                'sport': self.sport,
                'api_type': 'ESPN'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'sport': self.sport,
                'api_type': 'ESPN'
            } 
