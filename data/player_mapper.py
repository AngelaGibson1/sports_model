import pandas as pd
from datetime import datetime
from typing import Optional, Dict, List
from loguru import logger
from data.api_client import DataSourceClient
from config.settings import Settings


class PlayerMapper:
    """
    Manages the mapping between player names, IDs, teams, and current stats from different APIs.
    This class is dynamic and works for multiple sports (MLB, NBA, NFL).
    
    Note: Requires COMMON_STATS configuration in Settings for consistent stat mapping across sports.
    Add to your Settings file:
    
    COMMON_STATS = {
        'games_played': ['games', 'gp', 'g'],
        'minutes_played': ['minutes', 'min', 'mp'],
        'points': ['points', 'pts', 'p'],
        'assists': ['assists', 'ast', 'a'],
        'turnovers': ['turnovers', 'to', 'tov']
    }
    """
    
    def __init__(self, sport: str):
        """
        Initialize the player mapper for a specific sport.
        
        Args:
            sport: The sport key ('nba', 'mlb', 'nfl').
            
        Raises:
            ValueError: If sport is not supported.
        """
        self.sport = sport.lower()
        
        # Validate sport is supported
        if self.sport not in Settings.SPORT_CONFIGS:
            raise ValueError(f"Unsupported sport: {sport}. Supported: {list(Settings.SPORT_CONFIGS.keys())}")
        
        self.sport_config = Settings.SPORT_CONFIGS[self.sport]
        self.api_client = DataSourceClient(sport=self.sport)
        self.league_id = self.api_client.sports_client.default_league_id
        
        # Build maps on initialization
        self.team_map = self._build_team_map()
        self.player_map = self._build_player_map()
    
    def _get_current_season(self) -> int:
        """
        Get current season based on sport-specific calendar.
        
        Returns:
            Current season year based on sport calendar.
        """
        now = datetime.now()
        sport_config = self.sport_config
        
        if sport_config['season_start_month'] > sport_config['season_end_month']:
            # Season crosses calendar year (NBA, NFL)
            if now.month >= sport_config['season_start_month']:
                return now.year
            else:
                return now.year - 1
        else:
            # Season within calendar year (MLB)
            return now.year
    
    def _standardize_position(self, position: str) -> str:
        """
        Standardize positions across sports.
        
        Args:
            position: Raw position from API.
            
        Returns:
            Standardized position string.
        """
        if not position:
            return position
            
        position_maps = {
            'nba': {
                'Guard': 'G', 'Point Guard': 'PG', 'Shooting Guard': 'SG',
                'Forward': 'F', 'Power Forward': 'PF', 'Small Forward': 'SF',
                'Center': 'C', 'Forward-Center': 'F-C', 'Guard-Forward': 'G-F'
            },
            'nfl': {
                'Quarterback': 'QB', 'Running Back': 'RB', 'Fullback': 'FB',
                'Wide Receiver': 'WR', 'Tight End': 'TE', 'Offensive Line': 'OL',
                'Defensive Line': 'DL', 'Linebacker': 'LB', 'Defensive Back': 'DB',
                'Cornerback': 'CB', 'Safety': 'S', 'Kicker': 'K', 'Punter': 'P'
            },
            'mlb': {
                'Pitcher': 'P', 'Starting Pitcher': 'SP', 'Relief Pitcher': 'RP',
                'Catcher': 'C', 'First Baseman': '1B', 'Second Baseman': '2B',
                'Third Baseman': '3B', 'Shortstop': 'SS', 'Left Field': 'LF',
                'Center Field': 'CF', 'Right Field': 'RF', 'Outfielder': 'OF',
                'Infielder': 'IF', 'Designated Hitter': 'DH'
            }
        }
        
        sport_map = position_maps.get(self.sport, {})
        return sport_map.get(position, position)
    
    def _build_player_map(self) -> pd.DataFrame:
        """
        Fetches all players for the specified sport and creates a comprehensive map.
        
        Returns:
            A DataFrame with player names, IDs, and team information.
        """
        logger.info(f"Building {self.sport.upper()} player map...")
        current_season = self._get_current_season()
        
        try:
            all_players = self.api_client.sports_client.get_players(season=current_season)
            
            if all_players.empty:
                logger.warning(f"No player data found for {self.sport.upper()}. Player mapping will be unavailable.")
                return pd.DataFrame(columns=[
                    'player_id', 'full_name', 'first_name', 'last_name', 
                    'team_id', 'team_name', 'team_abbreviation', 'league_id', 'position', 'jersey_number'
                ])
                
            # Create a full name for easy matching
            all_players['full_name'] = all_players['firstname'] + ' ' + all_players['lastname']
            
            # Select and rename columns in a single, clean step
            player_map_df = all_players.rename(columns={
                'firstname': 'first_name',
                'lastname': 'last_name',
                'jersey': 'jersey_number'
            })
            
            # Add league_id and team_name from the pre-built team map
            player_map_df['league_id'] = self.league_id
            team_info = self.team_map.set_index('team_id')[['team_name', 'abbreviation']].to_dict('index')
            player_map_df['team_name'] = player_map_df['team_id'].map(lambda x: team_info.get(x, {}).get('team_name'))
            player_map_df['team_abbreviation'] = player_map_df['team_id'].map(lambda x: team_info.get(x, {}).get('abbreviation'))

            # Standardize positions
            if 'position' in player_map_df.columns:
                player_map_df['position'] = player_map_df['position'].apply(self._standardize_position)

            # Ensure we only keep the desired columns
            player_map_df = player_map_df[[
                'player_id', 'full_name', 'first_name', 'last_name', 
                'team_id', 'team_name', 'team_abbreviation', 'league_id', 'position', 'jersey_number'
            ]].copy()
            
            logger.info(f"✅ Player map built with {len(player_map_df)} players for {self.sport.upper()}.")
            return player_map_df
            
        except Exception as e:
            logger.error(f"Error building player map for {self.sport.upper()}: {e}")
            return pd.DataFrame(columns=[
                'player_id', 'full_name', 'first_name', 'last_name', 
                'team_id', 'team_name', 'team_abbreviation', 'league_id', 'position', 'jersey_number'
            ])
    
    def _build_team_map(self) -> pd.DataFrame:
        """
        Builds a mapping of team IDs to team names and information.
        
        Handles API response inconsistencies where team name columns vary:
        - MLB API uses 'name' for team names
        - Other sports may use 'full_name' or 'team_name'
        - Abbreviations may be in 'abbreviation', 'code', or 'abbr' fields
        
        Returns:
            DataFrame with team mapping information including name, city, and abbreviation.
        """
        logger.info(f"Building {self.sport.upper()} team map...")
        
        try:
            teams = self.api_client.sports_client.get_teams()
            if not teams.empty:
                # Handle inconsistent team name columns across different sport APIs
                team_columns = ['team_id', 'name', 'city']
                
                # Try to find team name column (different APIs use different column names)
                if 'name' not in teams.columns:
                    if 'full_name' in teams.columns:
                        teams = teams.rename(columns={'full_name': 'name'})
                    elif 'team_name' in teams.columns:
                        teams = teams.rename(columns={'team_name': 'name'})
                
                # Try to find abbreviation column (also inconsistent across APIs)
                abbr_column = None
                for col in ['abbreviation', 'code', 'abbr', 'short_code']:
                    if col in teams.columns:
                        abbr_column = col
                        break
                
                if abbr_column:
                    team_columns.append(abbr_column)
                
                # Select available columns
                available_columns = [col for col in team_columns if col in teams.columns]
                team_map_df = teams[available_columns].copy()
                
                # Standardize column names
                rename_dict = {'name': 'team_name'}
                if abbr_column:
                    rename_dict[abbr_column] = 'abbreviation'
                
                team_map_df = team_map_df.rename(columns=rename_dict)
                team_map_df['league_id'] = self.league_id
                
                # Ensure we have required columns even if missing from API
                required_columns = ['team_id', 'team_name', 'city', 'abbreviation', 'league_id']
                for col in required_columns:
                    if col not in team_map_df.columns:
                        team_map_df[col] = None
                
                # Validate we have expected number of teams
                expected_teams = self.sport_config['team_count']
                if len(team_map_df) != expected_teams:
                    logger.warning(f"Expected {expected_teams} teams for {self.sport.upper()}, got {len(team_map_df)}")
                
                logger.info(f"✅ Team map built with {len(team_map_df)} teams.")
                return team_map_df
            else:
                logger.warning(f"No team data found for {self.sport.upper()}.")
                return pd.DataFrame(columns=['team_id', 'team_name', 'city', 'abbreviation', 'league_id'])
                
        except Exception as e:
            logger.error(f"Error building team map for {self.sport.upper()}: {e}")
            return pd.DataFrame(columns=['team_id', 'team_name', 'city', 'abbreviation', 'league_id'])
    
    def get_player_id(self, name: str) -> Optional[int]:
        """
        Finds a player's ID given their full name.
        
        Args:
            name: The full name of the player (e.g., 'Shohei Ohtani').
            
        Returns:
            The player's ID, or None if not found.
        """
        match = self.player_map[self.player_map['full_name'].str.lower() == name.lower()]
        if not match.empty:
            return match.iloc[0]['player_id']
        return None
    
    def get_player_names(self, player_id: int) -> Optional[Dict]:
        """
        Finds a player's names given their ID.
        
        Args:
            player_id: The ID of the player.
            
        Returns:
            A dictionary with the player's names, or None if not found.
        """
        match = self.player_map[self.player_map['player_id'] == player_id]
        if not match.empty:
            return {
                'full_name': match.iloc[0]['full_name'],
                'first_name': match.iloc[0]['first_name'],
                'last_name': match.iloc[0]['last_name']
            }
        return None
    
    def get_player_info(self, name: str = None, player_id: int = None) -> Optional[Dict]:
        """
        Gets comprehensive player information including team and league details.
        
        Args:
            name: The full name of the player (optional if player_id provided).
            player_id: The ID of the player (optional if name provided).
            
        Returns:
            Dictionary with player info including name, team, position, etc.
        """
        if name:
            match = self.player_map[self.player_map['full_name'].str.lower() == name.lower()]
        elif player_id:
            match = self.player_map[self.player_map['player_id'] == player_id]
        else:
            return None
        
        if not match.empty:
            player_row = match.iloc[0]
            return {
                'player_id': player_row['player_id'],
                'full_name': player_row['full_name'],
                'first_name': player_row['first_name'],
                'last_name': player_row['last_name'],
                'team_id': player_row['team_id'],
                'team_name': player_row['team_name'],
                'team_abbreviation': player_row['team_abbreviation'],
                'league_id': player_row['league_id'],
                'position': player_row['position'],
                'jersey_number': player_row['jersey_number'],
                'sport': self.sport
            }
        return None
    
    def get_current_stats(self, player_id: int, season: int = None) -> Optional[Dict]:
        """
        Fetches and cleans current season stats for a specific player based on the sport.
        Uses config-driven stat mapping for consistency.
        
        Args:
            player_id: The ID of the player.
            season: The season year (defaults to current season based on sport calendar).
            
        Returns:
            Dictionary with current season stats, or None if not found.
        """
        if season is None:
            season = self._get_current_season()
        
        try:
            logger.info(f"Fetching stats for player {player_id} in season {season}")
            stats_df = self.api_client.sports_client.get_player_statistics(
                player_id=player_id, 
                season=season
            )
            
            if stats_df.empty:
                logger.warning(f"No stats found for player {player_id} in season {season}")
                return None
            
            # Use the latest stats entry if multiple are returned
            latest_stats = stats_df.iloc[-1].to_dict()
            
            # Base stats structure
            cleaned_stats = {
                'season': season, 
                'sport': self.sport,
                'raw_stats': latest_stats
            }
            
            # Use sport config for key stats - dynamic extraction
            key_stats = self.sport_config['key_stats']
            for stat in key_stats:
                # Try multiple variations of stat names (case insensitive)
                stat_value = None
                for key in [stat.lower(), stat.upper(), stat]:
                    if key in latest_stats:
                        stat_value = latest_stats[key]
                        break
                
                # Store with consistent lowercase key
                cleaned_stats[stat.lower().replace('/', '_per_').replace('%', '_pct')] = stat_value or 0
            
            # Add common stats that most sports share (from Settings)
            common_stats = getattr(Settings, 'COMMON_STATS', {
                'games_played': ['games', 'gp', 'g'],
                'minutes_played': ['minutes', 'min', 'mp'],
                'points': ['points', 'pts', 'p'],
                'assists': ['assists', 'ast', 'a'],
                'turnovers': ['turnovers', 'to', 'tov']
            })
            
            for stat_name, possible_keys in common_stats.items():
                for key in possible_keys:
                    if key in latest_stats:
                        cleaned_stats[stat_name] = latest_stats[key]
                        break
                else:
                    cleaned_stats[stat_name] = 0
            
            logger.info(f"✅ Retrieved stats for player {player_id}")
            return cleaned_stats
                
        except Exception as e:
            logger.error(f"Error fetching stats for player {player_id}: {e}")
            return None
    
    def get_player_with_stats(self, name: str = None, player_id: int = None, season: int = None) -> Optional[Dict]:
        """
        Gets comprehensive player information including current stats.
        
        Args:
            name: The full name of the player (optional if player_id provided).
            player_id: The ID of the player (optional if name provided).
            season: The season year (defaults to current season based on sport calendar).
            
        Returns:
            Dictionary with player info and current stats.
        """
        player_info = self.get_player_info(name=name, player_id=player_id)
        
        if not player_info:
            return None
        
        stats = self.get_current_stats(player_info['player_id'], season)
        
        result = player_info.copy()
        if stats:
            result['current_stats'] = stats
        else:
            result['current_stats'] = None
        
        return result
    
    def search_players(self, partial_name: str, limit: int = 10) -> List[Dict]:
        """
        Searches for players by partial name match.
        
        Args:
            partial_name: Partial player name to search for.
            limit: Maximum number of results to return.
            
        Returns:
            List of dictionaries with matching player information.
        """
        matches = self.player_map[
            self.player_map['full_name'].str.lower().str.contains(
                partial_name.lower(), na=False
            )
        ].head(limit)
        
        results = []
        for _, player in matches.iterrows():
            results.append({
                'player_id': player['player_id'],
                'full_name': player['full_name'],
                'first_name': player['first_name'],
                'last_name': player['last_name'],
                'team_name': player['team_name'],
                'team_abbreviation': player['team_abbreviation'],
                'position': player['position'],
                'league_id': player['league_id'],
                'sport': self.sport
            })
        
        return results
    
    def get_team_roster(self, team_name: str = None, team_id: int = None) -> List[Dict]:
        """
        Gets all players for a specific team.
        
        Args:
            team_name: Name of the team (optional if team_id provided).
            team_id: ID of the team (optional if team_name provided).
            
        Returns:
            List of dictionaries with player information for the team.
        """
        if team_name:
            team_players = self.player_map[
                self.player_map['team_name'].str.lower() == team_name.lower()
            ]
        elif team_id:
            team_players = self.player_map[self.player_map['team_id'] == team_id]
        else:
            return []
        
        results = []
        for _, player in team_players.iterrows():
            results.append({
                'player_id': player['player_id'],
                'full_name': player['full_name'],
                'first_name': player['first_name'],
                'last_name': player['last_name'],
                'position': player['position'],
                'jersey_number': player['jersey_number'],
                'team_abbreviation': player['team_abbreviation'],
                'league_id': player['league_id'],
                'sport': self.sport
            })
        
        return results
    
    def get_sport_info(self) -> Dict:
        """
        Get information about the current sport configuration.
        
        Returns:
            Dictionary with sport configuration details.
        """
        return {
            'sport': self.sport,
            'league_id': self.league_id,
            'config': self.sport_config,
            'current_season': self._get_current_season(),
            'total_teams': len(self.team_map),
            'total_players': len(self.player_map)
        }
    
    def validate_season(self, season: int) -> bool:
        """
        Validate if a season is reasonable for this sport.
        
        Args:
            season: Season year to validate.
            
        Returns:
            True if season is valid, False otherwise.
        """
        current_season = self._get_current_season()
        # Allow seasons from 10 years ago to 1 year in the future
        return (current_season - 10) <= season <= (current_season + 1)
    
    def get_position_groups(self) -> Dict[str, List[str]]:
        """
        Get position groups for the current sport.
        
        Returns:
            Dictionary mapping position groups to individual positions.
        """
        position_groups = {
            'nba': {
                'Guards': ['PG', 'SG', 'G', 'G-F'],
                'Forwards': ['SF', 'PF', 'F', 'F-C'],
                'Centers': ['C']
            },
            'nfl': {
                'Offense': ['QB', 'RB', 'FB', 'WR', 'TE', 'OL'],
                'Defense': ['DL', 'LB', 'DB', 'CB', 'S'],
                'Special Teams': ['K', 'P']
            },
            'mlb': {
                'Pitchers': ['P', 'SP', 'RP'],
                'Catchers': ['C'],
                'Infielders': ['1B', '2B', '3B', 'SS', 'IF'],
                'Outfielders': ['LF', 'CF', 'RF', 'OF'],
                'Designated Hitters': ['DH']
            }
        }
        
        return position_groups.get(self.sport, {})
    
    def refresh_data(self):
        """
        Refreshes the player and team mappings by fetching fresh data from the API.
        """
        logger.info(f"Refreshing {self.sport.upper()} player and team data...")
        self.team_map = self._build_team_map()
        self.player_map = self._build_player_map()
        logger.info("✅ Data refresh completed.")


# Convenience factory function for easy instantiation
def create_player_mapper(sport: str) -> PlayerMapper:
    """
    Factory function to create a PlayerMapper instance with validation.
    
    Args:
        sport: Sport key ('nba', 'mlb', 'nfl').
        
    Returns:
        Initialized PlayerMapper instance.
        
    Raises:
        ValueError: If sport is not supported.
    """
    return PlayerMapper(sport)
