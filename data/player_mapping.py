import pandas as pd
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
from loguru import logger

# FIXED: Import the correct existing clients
from api_clients.unified_api_client import DataSourceClient
from api_clients.sports_api import SportsAPIClient

from config.settings import Settings


class EnhancedPlayerMapper:
    """
    Enhanced PlayerMapper that prioritizes local CSV data from ingestion scripts.
    
    Data Source Priority:
    1. Local CSV files from data ingestion (fastest, most reliable)
    2. ESPN API (reliable backup)
    3. Sports API (fallback)
    
    Features:
    - Loads from data/latest/{sport}_players_latest.csv first
    - Smart cache invalidation based on file age
    - Automatic fallback to API sources
    - Comprehensive player mapping for ML models
    - Fast lookup performance for model inference
    """
    
    def __init__(self, sport: str, auto_build: bool = True, csv_max_age_hours: int = 24):
        """
        Initialize the enhanced player mapper.
        
        Args:
            sport: The sport key ('nba', 'mlb', 'nfl').
            auto_build: Whether to automatically build maps on initialization.
            csv_max_age_hours: Maximum age of CSV files before refreshing from API (default 24 hours).
        """
        self.sport = sport.lower()
        self.csv_max_age_hours = csv_max_age_hours
        
        # Validate sport
        if self.sport not in Settings.SPORT_CONFIGS:
            raise ValueError(f"Unsupported sport: {sport}. Supported: {list(Settings.SPORT_CONFIGS.keys())}")
        
        self.sport_config = Settings.SPORT_CONFIGS[self.sport]
        
        # Set up paths for CSV files
        self.project_root = Path(__file__).parent.parent  # Go up from data/ to project root
        self.data_dir = self.project_root / "data" / "latest"
        self.players_csv_path = self.data_dir / f"{self.sport}_players_latest.csv"
        self.teams_csv_path = self.data_dir / f"{self.sport}_teams_latest.csv"
        
        # FIXED: Initialize API client for fallback using existing clients
        try:
            self.api_client = DataSourceClient(sport=self.sport)
            logger.info(f"✅ Using DataSourceClient for {self.sport.upper()}")
        except Exception as e:
            logger.warning(f"DataSourceClient failed, falling back to SportsAPIClient: {e}")
            self.api_client = SportsAPIClient(sport=self.sport)
        
        # Initialize empty maps
        self.team_map = pd.DataFrame()
        self.player_map = pd.DataFrame()
        
        # Track data sources used
        self.data_sources_used = {
            'teams': 'unknown',
            'players': 'unknown',
            'last_updated': None
        }
        
        logger.info(f"🏀 Enhanced PlayerMapper initialized for {self.sport.upper()}")
        logger.info(f"📂 Looking for CSV files in: {self.data_dir}")
        
        # Build maps on initialization if requested
        if auto_build:
            logger.info(f"🚀 Auto-building {self.sport.upper()} player mapper...")
            self.refresh_data()
        else:
            logger.info(f"📋 {self.sport.upper()} player mapper initialized (auto_build=False)")
    
    def _check_csv_file_age(self, file_path: Path) -> Dict[str, Any]:
        """Check if CSV file exists and its age."""
        if not file_path.exists():
            return {'exists': False, 'age_hours': None, 'is_fresh': False}
        
        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
        age_hours = (datetime.now() - file_time).total_seconds() / 3600
        is_fresh = age_hours <= self.csv_max_age_hours
        
        return {
            'exists': True,
            'age_hours': age_hours,
            'is_fresh': is_fresh,
            'last_modified': file_time
        }
    
    def _load_csv_data(self, file_path: Path, data_type: str) -> pd.DataFrame:
        """Load and validate CSV data."""
        try:
            if not file_path.exists():
                logger.warning(f"📂 CSV file not found: {file_path}")
                return pd.DataFrame()
            
            df = pd.read_csv(file_path)
            logger.info(f"✅ Loaded {len(df)} {data_type} from {file_path.name}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading {data_type} CSV from {file_path}: {e}")
            return pd.DataFrame()
    
    def _build_team_map(self) -> pd.DataFrame:
        """Build team mapping prioritizing CSV data."""
        logger.info(f"🏟️ Building {self.sport.upper()} team map...")
        
        # Check CSV file first
        csv_status = self._check_csv_file_age(self.teams_csv_path)
        
        if csv_status['exists'] and csv_status['is_fresh']:
            logger.info(f"📂 Using fresh CSV team data (age: {csv_status['age_hours']:.1f} hours)")
            teams_df = self._load_csv_data(self.teams_csv_path, 'teams')
            
            if not teams_df.empty:
                self.data_sources_used['teams'] = f"CSV ({csv_status['last_modified'].strftime('%Y-%m-%d %H:%M')})"
                return self._process_team_data_from_csv(teams_df)
        
        # Fallback to API
        logger.info("🌐 CSV team data not available or stale, trying API...")
        try:
            current_season = self._get_current_season()
            teams_df = self.api_client.get_teams(season=current_season)
            
            if not teams_df.empty:
                api_source = type(self.api_client).__name__
                self.data_sources_used['teams'] = f'{api_source}_API'
                logger.info(f"✅ Found {len(teams_df)} teams from {api_source}")
                return self._process_team_data_from_api(teams_df)
        except Exception as e:
            logger.error(f"❌ API team fetch failed: {e}")
        
        logger.warning(f"No team data found for {self.sport.upper()}.")
        return self._get_empty_team_dataframe()
    
    def _build_player_map(self) -> pd.DataFrame:
        """Build player mapping prioritizing CSV data."""
        logger.info(f"👥 Building {self.sport.upper()} player map...")
        
        # Check CSV file first
        csv_status = self._check_csv_file_age(self.players_csv_path)
        
        if csv_status['exists'] and csv_status['is_fresh']:
            logger.info(f"📂 Using fresh CSV player data (age: {csv_status['age_hours']:.1f} hours)")
            players_df = self._load_csv_data(self.players_csv_path, 'players')
            
            if not players_df.empty:
                self.data_sources_used['players'] = f"CSV ({csv_status['last_modified'].strftime('%Y-%m-%d %H:%M')})"
                return self._process_player_data_from_csv(players_df)
        
        # Fallback to API
        logger.info("🌐 CSV player data not available or stale, trying API...")
        try:
            current_season = self._get_current_season()
            players_df = self.api_client.get_players(season=current_season)
            
            if not players_df.empty:
                api_source = type(self.api_client).__name__
                self.data_sources_used['players'] = f'{api_source}_API'
                logger.info(f"✅ Found {len(players_df)} players from {api_source}")
                return self._process_player_data_from_api(players_df)
        except Exception as e:
            logger.error(f"❌ API player fetch failed: {e}")
        
        logger.warning(f"No player data found for {self.sport.upper()}.")
        return self._get_empty_player_dataframe()
    
    def _process_team_data_from_csv(self, teams_df: pd.DataFrame) -> pd.DataFrame:
        """Process team data from ingested CSV files."""
        # CSV files from your ingestion should already be well-structured
        # Just need to standardize column names for PlayerMapper consistency
        
        # Common column mapping from your ingestion scripts
        column_mapping = {
            'id': 'team_id',
            'name': 'team_name', 
            'display_name': 'team_name',
            'abbreviation': 'abbreviation',
            'abbr': 'abbreviation',
            'code': 'abbreviation'
        }
        
        # Rename columns that exist
        for old_col, new_col in column_mapping.items():
            if old_col in teams_df.columns and new_col not in teams_df.columns:
                teams_df = teams_df.rename(columns={old_col: new_col})
        
        # Ensure required columns exist
        required_columns = ['team_id', 'team_name', 'city', 'abbreviation', 'league_id']
        for col in required_columns:
            if col not in teams_df.columns:
                if col == 'league_id':
                    teams_df[col] = self._get_league_id()
                elif col == 'city' and 'team_city' in teams_df.columns:
                    teams_df[col] = teams_df['team_city']
                else:
                    teams_df[col] = None
        
        # Select and clean data
        available_columns = [col for col in required_columns if col in teams_df.columns]
        team_map_df = teams_df[available_columns].copy()
        
        # Remove duplicates and clean
        team_map_df = team_map_df.drop_duplicates(subset=['team_id'], keep='first')
        
        logger.info(f"✅ Processed {len(team_map_df)} teams from CSV")
        return team_map_df
    
    def _process_player_data_from_csv(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """Process player data from ingested CSV files."""
        logger.info(f"🔧 Processing {len(players_df)} players from CSV...")
        
        # Column mapping from your ingestion scripts to PlayerMapper format
        column_mapping = {
            'id': 'player_id',
            'espn_player_id': 'player_id',
            'name': 'full_name',
            'display_name': 'full_name',
            'first_name': 'first_name',
            'last_name': 'last_name',
            'firstname': 'first_name', 
            'lastname': 'last_name',
            'team_abbr': 'team_abbreviation',
            'abbr': 'team_abbreviation',
            'jersey': 'jersey_number',
            'jersey_number': 'jersey_number'
        }
        
        # Rename columns that exist
        for old_col, new_col in column_mapping.items():
            if old_col in players_df.columns and new_col not in players_df.columns:
                players_df = players_df.rename(columns={old_col: new_col})
        
        # Create full_name if missing
        if 'full_name' not in players_df.columns:
            if 'first_name' in players_df.columns and 'last_name' in players_df.columns:
                players_df['full_name'] = (players_df['first_name'].astype(str) + ' ' + 
                                         players_df['last_name'].astype(str)).str.strip()
            elif 'name' in players_df.columns:
                players_df['full_name'] = players_df['name']
        
        # Split full_name if first/last missing
        if 'full_name' in players_df.columns:
            missing_names = (players_df['first_name'].isna() | players_df['first_name'].eq('')) & players_df['full_name'].notna()
            for idx in players_df[missing_names].index:
                full_name = str(players_df.loc[idx, 'full_name'])
                if ' ' in full_name:
                    name_parts = full_name.split()
                    players_df.loc[idx, 'first_name'] = name_parts[0]
                    players_df.loc[idx, 'last_name'] = ' '.join(name_parts[1:])
        
        # Generate player_id if missing
        if 'player_id' not in players_df.columns or players_df['player_id'].isna().all():
            if 'full_name' in players_df.columns and 'team_abbreviation' in players_df.columns:
                players_df['player_id'] = (players_df['full_name'].str.replace(' ', '_').str.lower() + 
                                         '_' + players_df['team_abbreviation'].str.lower())
            elif 'full_name' in players_df.columns:
                players_df['player_id'] = players_df['full_name'].str.replace(' ', '_').str.lower()
        
        # Ensure required columns exist
        required_columns = [
            'player_id', 'full_name', 'first_name', 'last_name', 
            'team_id', 'team_name', 'team_abbreviation', 'league_id', 'position', 'jersey_number'
        ]
        
        for col in required_columns:
            if col not in players_df.columns:
                if col == 'league_id':
                    players_df[col] = self._get_league_id()
                else:
                    players_df[col] = None
        
        # Add team info from team map if available
        if not self.team_map.empty and 'team_id' in players_df.columns:
            team_info = self.team_map.set_index('team_id')[['team_name', 'abbreviation']].to_dict('index')
            
            def map_team_info(team_id, field):
                if pd.isna(team_id):
                    return None
                return team_info.get(str(team_id), {}).get(field)
            
            # Fill missing team info from team map
            if players_df['team_name'].isna().any():
                players_df['team_name'] = players_df['team_name'].fillna(
                    players_df['team_id'].apply(lambda x: map_team_info(x, 'team_name'))
                )
            if players_df['team_abbreviation'].isna().any():
                players_df['team_abbreviation'] = players_df['team_abbreviation'].fillna(
                    players_df['team_id'].apply(lambda x: map_team_info(x, 'abbreviation'))
                )
        
        # Standardize positions
        if 'position' in players_df.columns:
            players_df['position'] = players_df['position'].apply(self._standardize_position)
        
        # Select final columns
        available_columns = [col for col in required_columns if col in players_df.columns]
        processed_df = players_df[available_columns].copy()
        
        # Remove duplicates
        if 'player_id' in processed_df.columns:
            processed_df = processed_df.drop_duplicates(subset=['player_id'], keep='first')
        else:
            processed_df = processed_df.drop_duplicates(subset=['full_name'], keep='first')
        
        logger.info(f"✅ Processed {len(processed_df)} unique players from CSV")
        return processed_df
    
    def _process_team_data_from_api(self, teams_df: pd.DataFrame) -> pd.DataFrame:
        """Process team data from API (fallback method)."""
        # Similar to your original method but for API data
        # Handle different column naming from ESPN vs Sports API
        
        if 'team_id' not in teams_df.columns:
            if 'id' in teams_df.columns:
                teams_df = teams_df.rename(columns={'id': 'team_id'})
        
        if 'team_name' not in teams_df.columns:
            if 'name' in teams_df.columns:
                teams_df = teams_df.rename(columns={'name': 'team_name'})
            elif 'displayName' in teams_df.columns:
                teams_df = teams_df.rename(columns={'displayName': 'team_name'})
        
        if 'abbreviation' not in teams_df.columns:
            for col in ['abbr', 'code', 'shortDisplayName']:
                if col in teams_df.columns:
                    teams_df = teams_df.rename(columns={col: 'abbreviation'})
                    break
        
        required_columns = ['team_id', 'team_name', 'city', 'abbreviation', 'league_id']
        for col in required_columns:
            if col not in teams_df.columns:
                if col == 'league_id':
                    teams_df[col] = self._get_league_id()
                else:
                    teams_df[col] = None
        
        available_columns = [col for col in required_columns if col in teams_df.columns]
        return teams_df[available_columns].copy()
    
    def _process_player_data_from_api(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """Process player data from API (fallback method)."""
        # Similar processing for API data
        if 'player_id' not in players_df.columns:
            if 'id' in players_df.columns:
                players_df = players_df.rename(columns={'id': 'player_id'})
        
        if 'full_name' not in players_df.columns:
            if 'firstname' in players_df.columns and 'lastname' in players_df.columns:
                players_df['full_name'] = players_df['firstname'].astype(str) + ' ' + players_df['lastname'].astype(str)
            elif 'name' in players_df.columns:
                players_df['full_name'] = players_df['name']
        
        # Ensure required columns exist
        required_columns = [
            'player_id', 'full_name', 'first_name', 'last_name', 
            'team_id', 'team_name', 'team_abbreviation', 'league_id', 'position', 'jersey_number'
        ]
        
        for col in required_columns:
            if col not in players_df.columns:
                if col == 'league_id':
                    players_df[col] = self._get_league_id()
                else:
                    players_df[col] = None
        
        available_columns = [col for col in required_columns if col in players_df.columns]
        return players_df[available_columns].copy()
    
    def check_csv_status(self) -> Dict[str, Any]:
        """Check status of CSV files."""
        players_status = self._check_csv_file_age(self.players_csv_path)
        teams_status = self._check_csv_file_age(self.teams_csv_path)
        
        return {
            'players_csv': {
                'path': str(self.players_csv_path),
                'exists': players_status['exists'],
                'age_hours': players_status.get('age_hours'),
                'is_fresh': players_status.get('is_fresh', False),
                'last_modified': players_status.get('last_modified')
            },
            'teams_csv': {
                'path': str(self.teams_csv_path),
                'exists': teams_status['exists'],
                'age_hours': teams_status.get('age_hours'),
                'is_fresh': teams_status.get('is_fresh', False),
                'last_modified': teams_status.get('last_modified')
            },
            'csv_max_age_hours': self.csv_max_age_hours
        }
    
    def force_api_refresh(self):
        """Force refresh from API, ignoring CSV files."""
        logger.info(f"🔄 Forcing API refresh for {self.sport.upper()} (ignoring CSV files)...")
        
        try:
            # Get teams from API
            current_season = self._get_current_season()
            teams_df = self.api_client.get_teams(season=current_season)
            
            if not teams_df.empty:
                self.team_map = self._process_team_data_from_api(teams_df)
                api_source = type(self.api_client).__name__
                self.data_sources_used['teams'] = f'{api_source}_API (forced)'
                logger.info(f"✅ Refreshed {len(self.team_map)} teams from API")
            
            # Get players from API
            players_df = self.api_client.get_players(season=current_season)
            
            if not players_df.empty:
                self.player_map = self._process_player_data_from_api(players_df)
                api_source = type(self.api_client).__name__
                self.data_sources_used['players'] = f'{api_source}_API (forced)'
                logger.info(f"✅ Refreshed {len(self.player_map)} players from API")
            
        except Exception as e:
            logger.error(f"❌ Forced API refresh failed: {e}")
    
    # ============= UTILITY METHODS =============
    
    def _get_current_season(self) -> int:
        """Get current season based on sport-specific calendar."""
        now = datetime.now()
        sport_config = self.sport_config
        
        if self.sport == 'mlb':
            return now.year
        
        if sport_config['season_start_month'] > sport_config['season_end_month']:
            if now.month >= sport_config['season_start_month']:
                return now.year
            else:
                return now.year - 1
        else:
            return now.year
    
    def _get_league_id(self) -> int:
        """Get league ID with multiple fallback attempts."""
        defaults = {'nba': 12, 'mlb': 1, 'nfl': 1}
        return defaults.get(self.sport, 1)
    
    def _standardize_position(self, position: str) -> str:
        """Standardize positions across sports."""
        if not position or pd.isna(position):
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
    
    def _get_empty_player_dataframe(self) -> pd.DataFrame:
        """Create an empty DataFrame with consistent player mapping columns."""
        return pd.DataFrame(columns=[
            'player_id', 'full_name', 'first_name', 'last_name', 
            'team_id', 'team_name', 'team_abbreviation', 'league_id', 'position', 'jersey_number'
        ])
    
    def _get_empty_team_dataframe(self) -> pd.DataFrame:
        """Create an empty DataFrame with consistent team mapping columns."""
        return pd.DataFrame(columns=['team_id', 'team_name', 'city', 'abbreviation', 'league_id'])
    
    def refresh_data(self):
        """Enhanced refresh with CSV priority."""
        logger.info(f"🔄 Starting data refresh for {self.sport.upper()}...")
        
        # Show CSV status
        csv_status = self.check_csv_status()
        logger.info(f"📂 CSV Status:")
        logger.info(f"   Players: {'✅ Fresh' if csv_status['players_csv']['is_fresh'] else '❌ Stale/Missing'}")
        logger.info(f"   Teams: {'✅ Fresh' if csv_status['teams_csv']['is_fresh'] else '❌ Stale/Missing'}")
        
        # Refresh team map first
        self.team_map = self._build_team_map()
        
        # Refresh player map
        self.player_map = self._build_player_map()
        
        # Update timestamp
        self.data_sources_used['last_updated'] = datetime.now()
        
        # Summary
        logger.info("🎯 Data refresh completed:")
        logger.info(f"   Teams: {len(self.team_map)} (source: {self.data_sources_used['teams']})")
        logger.info(f"   Players: {len(self.player_map)} (source: {self.data_sources_used['players']})")
        
        status = '✅ Ready for ML models' if not self.player_map.empty else '⚠️ Limited functionality'
        logger.info(f"   Status: {status}")
    
    # ============= PLAYER LOOKUP METHODS =============
    
    def get_player_id(self, name: str) -> Optional[str]:
        """Finds a player's ID given their full name."""
        if self.player_map.empty:
            logger.warning("Player map is empty. Cannot lookup player ID.")
            return None
            
        match = self.player_map[self.player_map['full_name'].str.lower() == name.lower()]
        if not match.empty:
            return match.iloc[0]['player_id']
        return None
    
    def get_player_info(self, name: str = None, player_id: str = None) -> Optional[Dict]:
        """Gets comprehensive player information."""
        if self.player_map.empty:
            logger.warning("Player map is empty. Cannot lookup player info.")
            return None
            
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
    
    def search_players(self, partial_name: str, limit: int = 10) -> List[Dict]:
        """Searches for players by partial name match."""
        if self.player_map.empty:
            logger.warning("Player map is empty. Cannot search players.")
            return []
            
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
    
    def get_team_players(self, team_name: str = None, team_id: str = None) -> List[Dict]:
        """Get all players for a specific team."""
        if self.player_map.empty:
            logger.warning("Player map is empty. Cannot get team players.")
            return []
        
        if team_name:
            matches = self.player_map[
                self.player_map['team_name'].str.lower().str.contains(team_name.lower(), na=False)
            ]
        elif team_id:
            matches = self.player_map[self.player_map['team_id'] == team_id]
        else:
            return []
        
        results = []
        for _, player in matches.iterrows():
            results.append({
                'player_id': player['player_id'],
                'full_name': player['full_name'],
                'position': player['position'],
                'jersey_number': player['jersey_number'],
                'team_name': player['team_name'],
                'team_abbreviation': player['team_abbreviation']
            })
        
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary including CSV status."""
        csv_status = self.check_csv_status()
        
        return {
            'sport': self.sport,
            'current_season': self._get_current_season(),
            'teams_loaded': len(self.team_map),
            'players_loaded': len(self.player_map),
            'league_id': self._get_league_id(),
            'data_sources_used': self.data_sources_used,
            'csv_status': csv_status,
            'status': 'ready' if not self.player_map.empty else 'limited',
            'last_updated': self.data_sources_used.get('last_updated'),
            'csv_max_age_hours': self.csv_max_age_hours
        }


# Factory function
def create_enhanced_player_mapper(sport: str, auto_build: bool = True, csv_max_age_hours: int = 24) -> EnhancedPlayerMapper:
    """
    Factory function to create an Enhanced PlayerMapper with CSV integration.
    
    Args:
        sport: Sport key ('nba', 'mlb', 'nfl').
        auto_build: Whether to automatically build maps.
        csv_max_age_hours: Maximum age of CSV files before refreshing from API.
        
    Returns:
        Initialized EnhancedPlayerMapper instance.
    """
    return EnhancedPlayerMapper(sport, auto_build=auto_build, csv_max_age_hours=csv_max_age_hours)


# Test function
def test_csv_player_mapper(sport: str = 'nba') -> Dict[str, Any]:
    """Test Enhanced PlayerMapper with CSV integration."""
    print(f"🧪 Testing CSV-Enhanced PlayerMapper for {sport.upper()}")
    print("=" * 60)
    
    results = {
        'sport': sport,
        'tests': {},
        'summary': 'unknown'
    }
    
    try:
        # Create mapper
        print("1️⃣ Creating Enhanced PlayerMapper...")
        mapper = EnhancedPlayerMapper(sport, auto_build=False)
        results['tests']['creation'] = 'success'
        
        # Check CSV status
        print("2️⃣ Checking CSV file status...")
        csv_status = mapper.check_csv_status()
        results['tests']['csv_status'] = csv_status
        
        print(f"   Players CSV: {'✅ Found' if csv_status['players_csv']['exists'] else '❌ Missing'}")
        print(f"   Teams CSV: {'✅ Found' if csv_status['teams_csv']['exists'] else '❌ Missing'}")
        
        # Build maps
        print("3️⃣ Building data maps...")
        mapper.refresh_data()
        
        # Get summary
        summary = mapper.get_summary()
        results['tests']['summary'] = summary
        
        print(f"📊 Results:")
        print(f"   Teams: {summary['teams_loaded']} (source: {summary['data_sources_used']['teams']})")
        print(f"   Players: {summary['players_loaded']} (source: {summary['data_sources_used']['players']})")
        
        # Test player search
        if summary['players_loaded'] > 0:
            print("4️⃣ Testing player search...")
            search_results = mapper.search_players('james', limit=3)
            results['tests']['search'] = len(search_results)
            print(f"   Found {len(search_results)} players matching 'james'")
            
            if search_results:
                player = search_results[0]
                print(f"   Sample: {player['full_name']} ({player['team_name']}) - {player['position']}")
                
                # Test detailed lookup
                player_info = mapper.get_player_info(player['full_name'])
                if player_info:
                    print(f"   Detailed info: {player_info['full_name']} - ID: {player_info['player_id']}")
        
        # Determine overall status
        if summary['players_loaded'] > 0:
            results['summary'] = 'success'
            print("✅ All tests passed! CSV integration working.")
        elif summary['teams_loaded'] > 0:
            results['summary'] = 'partial'
            print("⚠️ Teams loaded but no players")
        else:
            results['summary'] = 'failed'
            print("❌ No data loaded")
            
    except Exception as e:
        results['tests']['error'] = str(e)
        results['summary'] = 'error'
        print(f"❌ Test failed: {e}")
    
    return results


if __name__ == "__main__":
    # Test CSV-enhanced PlayerMapper
    for sport in ['nba', 'mlb', 'nfl']:
        test_results = test_csv_player_mapper(sport)
        print(f"\n{sport.upper()} CSV Test Results: {test_results['summary']}")
        print("-" * 40)
