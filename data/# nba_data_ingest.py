# nba_data_ingest.py
"""
NBA Data Ingestion Script for Model Training - ESPN Hybrid Version
Uses ESPN API for teams and web scraping for comprehensive player rosters.
Retrieves full team rosters from ESPN.com to get all players, not just 25.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger
import time
import requests
from bs4 import BeautifulSoup
import re

# Import your ESPN API client
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # Add parent directory to path

from api_clients.espn_api import ESPNAPIClient


class NBADataIngest:
    """NBA data ingestion using ESPN API + web scraping for comprehensive player data."""
    
    def __init__(self, output_dir: str = "data"):
        """
        Initialize NBA data ingestion.
        
        Args:
            output_dir: Base directory for logs and data files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create latest data directory for ML training files
        self.latest_dir = self.output_dir / "latest"
        self.latest_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ESPN client for teams
        self.client = ESPNAPIClient(sport='nba')
        
        # Setup logging
        log_file = self.output_dir / f"nba_ingest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger.add(log_file, rotation="10 MB", retention="30 days")
        
        # Enhanced ESPN team abbreviations mapping for web scraping
        self.team_abbreviations = {
            'atlanta': 'atl', 'boston': 'bos', 'brooklyn': 'bkn', 'charlotte': 'cha',
            'chicago': 'chi', 'cleveland': 'cle', 'dallas': 'dal', 'denver': 'den',
            'detroit': 'det', 'golden state': 'gs', 'houston': 'hou', 'indiana': 'ind',
            'los angeles clippers': 'lac', 'los angeles lakers': 'lal', 'memphis': 'mem',
            'miami': 'mia', 'milwaukee': 'mil', 'minnesota': 'min', 'new orleans': 'no',
            'new york': 'ny', 'oklahoma city': 'okc', 'orlando': 'orl', 'philadelphia': 'phi',
            'phoenix': 'phx', 'portland': 'por', 'sacramento': 'sac', 'san antonio': 'sa',
            'toronto': 'tor', 'utah': 'utah', 'washington': 'wsh',
            # Alternative names/variations
            'hawks': 'atl', 'celtics': 'bos', 'nets': 'bkn', 'hornets': 'cha',
            'bulls': 'chi', 'cavaliers': 'cle', 'mavericks': 'dal', 'nuggets': 'den',
            'pistons': 'det', 'warriors': 'gs', 'rockets': 'hou', 'pacers': 'ind',
            'clippers': 'lac', 'lakers': 'lal', 'grizzlies': 'mem', 'heat': 'mia',
            'bucks': 'mil', 'timberwolves': 'min', 'pelicans': 'no', 'knicks': 'ny',
            'thunder': 'okc', 'magic': 'orl', '76ers': 'phi', 'suns': 'phx',
            'trail blazers': 'por', 'blazers': 'por', 'kings': 'sac', 'spurs': 'sa',
            'raptors': 'tor', 'jazz': 'utah', 'wizards': 'wsh'
        }
        
        logger.info("🏀 NBA Data Ingestion initialized with ESPN API + Web Scraping")
    
    def get_seasons_to_process(self) -> List[int]:
        """Get the last 5 NBA seasons including current."""
        try:
            seasons = self.client.get_seasons()
            if len(seasons) >= 5:
                return seasons[-5:]
            else:
                return seasons
        except Exception as e:
            logger.warning(f"Could not get seasons from API: {e}")
            # Fallback
            current_date = datetime.now()
            if current_date.month >= 10:  # NBA season starts in October
                current_season = current_date.year
            else:
                current_season = current_date.year - 1
            
            return list(range(current_season - 4, current_season + 1))
    
    def fetch_teams_data(self, seasons: List[int]) -> pd.DataFrame:
        """Fetch NBA team data for all seasons using ESPN API."""
        logger.info(f"📊 Fetching NBA team data for seasons: {seasons}")
        
        all_teams = []
        
        for season in seasons:
            try:
                logger.info(f"Getting teams for {season}")
                teams = self.client.get_teams(season=season)
                
                if not teams.empty:
                    teams['season'] = season
                    teams['data_source'] = 'ESPN_API'
                    teams['ingestion_date'] = datetime.now()
                    all_teams.append(teams)
                    logger.info(f"✅ Found {len(teams)} teams for {season}")
                else:
                    logger.warning(f"❌ No teams found for {season}")
                
                time.sleep(0.5)  # Be respectful to API
                
            except Exception as e:
                logger.error(f"❌ Error fetching teams for {season}: {e}")
                continue
        
        if all_teams:
            combined_teams = pd.concat(all_teams, ignore_index=True)
            logger.info(f"✅ Total teams collected: {len(combined_teams)}")
            return self.clean_team_data(combined_teams)
        else:
            logger.error("❌ No team data collected")
            return pd.DataFrame()
    
    def get_nba_team_roster(self, url: str) -> List[Dict]:
        """
        Fetch NBA roster from a given ESPN URL.
        """
        try:
            logger.debug(f"Fetching roster from: {url}")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            players = []
            
            # Look for the main roster table
            table = soup.find('table', class_='Table')
            if not table:
                logger.debug("No roster table found on page.")
                return []
                
            tbody = table.find('tbody')
            if not tbody:
                logger.debug("No tbody found in roster table.")
                return []
                
            rows = tbody.find_all('tr')
            
            for row in rows:
                cells = row.find_all('td')
                if len(cells) < 4:  # Check for expected number of columns
                    continue
                
                try:
                    player_data = {}
                    
                    # Jersey number (first column)
                    jersey_text = cells[0].get_text(strip=True)
                    if jersey_text.isdigit():
                        player_data['jersey_number'] = jersey_text
                    
                    # Name and URL (second column - includes player ID)
                    name_cell = cells[1].find('a')
                    if name_cell:
                        player_data['name'] = name_cell.get_text(strip=True)
                        href = name_cell.get('href', '')
                        id_match = re.search(r'/id/(\d+)/', href)
                        if id_match:
                            player_data['espn_player_id'] = id_match.group(1)
                    else:
                        player_data['name'] = cells[1].get_text(strip=True)
                    
                    # Position (third column)
                    player_data['position'] = cells[2].get_text(strip=True)
                    
                    # Additional columns if available: Height, Weight, Age, College
                    if len(cells) > 3:
                        player_data['height'] = cells[3].get_text(strip=True)
                    if len(cells) > 4:
                        player_data['weight'] = cells[4].get_text(strip=True)
                    if len(cells) > 5:
                        player_data['age'] = cells[5].get_text(strip=True)
                    if len(cells) > 6:
                        player_data['college'] = cells[6].get_text(strip=True)
                    
                    # Only add if we have a name
                    if player_data.get('name'):
                        players.append(player_data)
                
                except Exception as e:
                    logger.debug(f"Error parsing player row: {e}")
                    continue
            
            return players
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error fetching roster from {url}: {e}")
            return []
        except Exception as e:
            logger.warning(f"Error parsing roster from {url}: {e}")
            return []
    
    def fetch_players_data(self, seasons: List[int], teams_df: pd.DataFrame) -> pd.DataFrame:
        """Fetch comprehensive NBA player data by scraping team rosters from ESPN."""
        logger.info(f"🏀 Fetching NBA player data via ESPN web scraping for seasons: {seasons}")
        
        if teams_df.empty:
            logger.error("❌ No teams data available for player fetching")
            return pd.DataFrame()
        
        all_players = []
        
        # Get current season teams for roster scraping
        # We'll use the most recent season's team list, as rosters are year-specific on the site.
        current_season = max(seasons) if seasons else datetime.now().year
        current_season_teams = teams_df[teams_df['season'] == current_season]
        
        if current_season_teams.empty:
            # Fall back to any available season
            current_season_teams = teams_df[teams_df['season'] == teams_df['season'].max()]
        
        if current_season_teams.empty:
            logger.error("❌ No teams found for any season, cannot scrape rosters.")
            return pd.DataFrame()
        
        logger.info(f"Scraping rosters for {len(current_season_teams)} teams from season {current_season_teams['season'].iloc[0]}")
        
        successful_teams = 0
        total_players_found = 0
        
        for _, team in current_season_teams.iterrows():
            team_id = team.get('team_id')
            team_name = team.get('name', '').lower()
            team_city = team.get('city', '').lower()
            
            # Find team abbreviation directly from the mapping
            team_abbr = self.team_abbreviations.get(team_city, None)
            if not team_abbr:
                team_abbr = self.team_abbreviations.get(team_name, None)
            
            # Try combinations if direct lookup fails
            if not team_abbr:
                full_team_name = f"{team_city} {team_name}".strip()
                for full_name, abbr in self.team_abbreviations.items():
                    if full_name in full_team_name or full_team_name in full_name:
                        team_abbr = abbr
                        break
            
            if not team_abbr:
                logger.warning(f"⚪ Could not find abbreviation for {team_city} {team_name}, skipping.")
                continue
            
            try:
                # Create proper team name for URL (no redundant city names)
                display_team_name = team.get('name', team_abbr).replace(' ', '-').lower()
                
                logger.debug(f"Fetching roster for {team_city} {team_name} ({team_abbr})")
                
                # Use the correct, simplified URL for scraping
                roster_url = f"https://www.espn.com/nba/team/roster/_/name/{team_abbr}/{display_team_name}"
                
                # Get current roster
                roster = self.get_nba_team_roster(roster_url)
                
                if roster:
                    # Convert to DataFrame and add metadata
                    team_players_df = pd.DataFrame(roster)
                    
                    # Add team metadata
                    team_players_df['team_id'] = team_id
                    team_players_df['team_abbr'] = team_abbr
                    team_players_df['team_name'] = team.get('name', '')
                    team_players_df['team_city'] = team.get('city', '')
                    team_players_df['fetch_method'] = 'espn_web_scraping'
                    team_players_df['data_source'] = 'ESPN_Website'
                    team_players_df['ingestion_date'] = datetime.now()
                    
                    # Assign this roster to all seasons
                    for season in seasons:
                        season_players = team_players_df.copy()
                        season_players['season'] = season
                        all_players.append(season_players)
                    
                    successful_teams += 1
                    total_players_found += len(roster)
                    logger.info(f"✅ {team_city} {team_name}: {len(roster)} players found.")
                else:
                    logger.warning(f"⚪ {team_city} {team_name}: No players found on roster page.")
                
                # Rate limiting
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"❌ Error scraping roster for {team_city} {team_name}: {e}")
                continue
        
        logger.info(f"📊 Web scraping results: {successful_teams}/{len(current_season_teams)} teams, {total_players_found} total players")
        
        if not all_players:
            logger.error("❌ No player data collected from web scraping")
            return pd.DataFrame()
        
        # Combine and deduplicate
        combined_players = pd.concat(all_players, ignore_index=True)
        
        # Remove duplicates by name and season
        initial_total = len(combined_players)
        combined_players = combined_players.drop_duplicates(
            subset=['name', 'season', 'team_abbr'], keep='first'
        )
        final_total = len(combined_players)
        
        logger.info(f"✅ FINAL PLAYER RESULTS:")
        logger.info(f"📊 Total player records: {initial_total} → {final_total}")
        logger.info(f"👥 Unique players: {combined_players['name'].nunique()}")
        logger.info(f"📈 Seasons covered: {sorted(combined_players['season'].unique())}")
        
        return self.clean_player_data(combined_players)
    
    def clean_team_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean NBA team data for consistency."""
        if df.empty:
            return df
        
        # Standardize column names
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        # Clean text fields
        text_cols = ['name', 'code', 'city', 'country']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Add conference and division information
        if 'city' in df.columns or 'name' in df.columns:
            df['conference'] = df.apply(self._assign_nba_conference, axis=1)
            df['division'] = df.apply(self._assign_nba_division, axis=1)
        
        return df
    
    def clean_player_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize NBA player data for ML training."""
        if df.empty:
            return df
        
        # Standardize column names
        df.columns = [col.lower().replace(' ', '_').replace('.', '_') for col in df.columns]
        
        # Clean text fields
        text_cols = ['name', 'position', 'team_name', 'team_abbr']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace('nan', '')
        
        # Generate player_id from name and team (since we don't get IDs from scraping consistently)
        if 'name' in df.columns and 'team_abbr' in df.columns:
            df['player_id'] = (df['name'].str.replace(' ', '_').str.lower() + 
                             '_' + df['team_abbr'].str.lower())
        elif 'name' in df.columns:
            df['player_id'] = df['name'].str.replace(' ', '_').str.lower()
        
        # NBA position standardization
        if 'position' in df.columns:
            df['position_standard'] = df['position'].apply(self._standardize_nba_position)
            df['is_guard'] = df['position_standard'].isin(['PG', 'SG', 'G'])
            df['is_forward'] = df['position_standard'].isin(['SF', 'PF', 'F'])
            df['is_center'] = df['position_standard'].isin(['C'])
            df['is_big_man'] = df['position_standard'].isin(['PF', 'C', 'F-C'])
            df['is_perimeter'] = df['position_standard'].isin(['PG', 'SG', 'SF', 'G', 'G-F'])
        
        # Physical attributes
        if 'height' in df.columns:
            df['height_inches'] = df['height'].apply(self._convert_height_to_inches)
        
        if 'weight' in df.columns:
            df['weight_lbs'] = pd.to_numeric(df['weight'], errors='coerce')
            if 'height_inches' in df.columns:
                df['bmi'] = df.apply(
                    lambda x: self._calculate_bmi(x['weight_lbs'], x['height_inches']), 
                    axis=1
                )
        
        # Convert additional fields from improved scraping
        if 'jersey_number' in df.columns:
            df['jersey_number'] = pd.to_numeric(df['jersey_number'], errors='coerce')
        
        if 'age' in df.columns:
            df['age_numeric'] = pd.to_numeric(df['age'], errors='coerce')
        
        # Clean college field
        if 'college' in df.columns:
            df['college'] = df['college'].astype(str).str.strip()
            df['college'] = df['college'].replace('nan', '')
            df['college'] = df['college'].replace('--', '')
        
        # Use ESPN player ID if available
        if 'espn_player_id' in df.columns:
            # Prefer ESPN ID when available, fall back to generated ID
            df['player_id'] = df['espn_player_id'].fillna(df['player_id'])
        
        # Career progression features
        if 'season' in df.columns:
            df['is_current_season'] = df['season'] == df['season'].max()
            
            if 'player_id' in df.columns:
                df = df.sort_values(['player_id', 'season'])
                df['career_year'] = df.groupby('player_id').cumcount() + 1
        
        return df
    
    def _assign_nba_conference(self, row) -> str:
        """Assign NBA conference based on team."""
        eastern_teams = {
            'atlanta', 'boston', 'brooklyn', 'charlotte', 'chicago', 'cleveland',
            'detroit', 'indiana', 'miami', 'milwaukee', 'new york', 'orlando',
            'philadelphia', 'toronto', 'washington'
        }
        
        team_identifiers = [
            str(row.get('city', '')).lower(),
            str(row.get('name', '')).lower()
        ]
        
        for identifier in team_identifiers:
            if any(team in identifier for team in eastern_teams):
                return 'Eastern'
        
        return 'Western'
    
    def _assign_nba_division(self, row) -> str:
        """Assign NBA division based on team."""
        divisions = {
            'Atlantic': ['boston', 'brooklyn', 'new york', 'philadelphia', 'toronto'],
            'Central': ['chicago', 'cleveland', 'detroit', 'indiana', 'milwaukee'],
            'Southeast': ['atlanta', 'charlotte', 'miami', 'orlando', 'washington'],
            'Northwest': ['denver', 'minnesota', 'oklahoma', 'portland', 'utah'],
            'Pacific': ['golden state', 'los angeles', 'phoenix', 'sacramento'],
            'Southwest': ['dallas', 'houston', 'memphis', 'new orleans', 'san antonio']
        }
        
        team_identifiers = [
            str(row.get('city', '')).lower(),
            str(row.get('name', '')).lower()
        ]
        
        for division, cities in divisions.items():
            for identifier in team_identifiers:
                if any(city in identifier for city in cities):
                    return division
        
        return 'Unknown'
    
    def _standardize_nba_position(self, position: str) -> str:
        """Standardize NBA position abbreviations."""
        if pd.isna(position) or position == '':
            return 'Unknown'
        
        pos = str(position).upper().strip()
        
        position_map = {
            'POINT GUARD': 'PG', 'PG': 'PG', 'G': 'PG',
            'SHOOTING GUARD': 'SG', 'SG': 'SG', 
            'SMALL FORWARD': 'SF', 'SF': 'SF',
            'POWER FORWARD': 'PF', 'PF': 'PF', 'F': 'PF',
            'CENTER': 'C', 'C': 'C',
            'GUARD': 'G', 'FORWARD': 'F', 'F-C': 'F-C', 'G-F': 'G-F'
        }
        
        return position_map.get(pos, pos)
    
    def _convert_height_to_inches(self, height: str) -> Optional[float]:
        """Convert height from feet-inches format to total inches."""
        if pd.isna(height) or height == '':
            return None
        
        try:
            height_str = str(height).strip()
            
            if "'" in height_str and '"' in height_str:
                # Format: 6'2"
                parts = height_str.replace('"', '').split("'")
                feet = int(parts[0])
                inches = int(parts[1]) if parts[1] else 0
                return feet * 12 + inches
            elif '-' in height_str:
                # Format: 6-2
                parts = height_str.split('-')
                feet = int(parts[0])
                inches = int(parts[1]) if len(parts) > 1 else 0
                return feet * 12 + inches
            else:
                # Try to parse as total inches
                return float(height_str)
        except:
            return None
    
    def _calculate_bmi(self, weight_lbs: float, height_inches: float) -> Optional[float]:
        """Calculate BMI from weight (lbs) and height (inches)."""
        if pd.isna(weight_lbs) or pd.isna(height_inches) or height_inches <= 0:
            return None
        
        try:
            weight_kg = weight_lbs * 0.453592
            height_m = height_inches * 0.0254
            return weight_kg / (height_m ** 2)
        except:
            return None
    
    def create_training_features(self, players_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for NBA ML training."""
        if players_df.empty:
            return players_df
        
        logger.info("🔧 Creating NBA training features")
        
        # Merge team information
        if not teams_df.empty and 'team_id' in players_df.columns:
            team_mapping = teams_df[['team_id', 'season', 'name', 'city', 'conference', 'division']].drop_duplicates()
            team_mapping.columns = ['team_id', 'season', 'team_name_api', 'team_city_api', 'team_conference', 'team_division']
            
            players_df = players_df.merge(
                team_mapping,
                on=['team_id', 'season'],
                how='left'
            )
        
        # Career progression features
        if 'player_id' in players_df.columns and 'season' in players_df.columns:
            players_df = players_df.sort_values(['player_id', 'season'])
            
            # Career statistics
            players_df['seasons_played'] = players_df.groupby('player_id').cumcount() + 1
            players_df['career_seasons'] = players_df.groupby('player_id')['season'].transform('count')
            
            # Age-related features
            if 'age_numeric' in players_df.columns:
                players_df['is_rookie'] = players_df['seasons_played'] == 1
                players_df['is_veteran'] = players_df['age_numeric'] >= 30
                players_df['prime_years'] = (players_df['age_numeric'] >= 25) & (players_df['age_numeric'] <= 29)
        
        return players_df
    
    def validate_data_quality(self, df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """Validate NBA data quality for ML training."""
        validation = {
            'data_type': data_type,
            'total_records': len(df),
            'missing_data': {},
            'data_quality_score': 0,
            'issues': []
        }
        
        if df.empty:
            validation['issues'].append("No data available")
            return validation
        
        # Check for missing data in key columns
        key_columns = {
            'players': ['name', 'season', 'team_abbr'],
            'teams': ['team_id', 'name', 'season']
        }
        
        if data_type in key_columns:
            for col in key_columns[data_type]:
                if col in df.columns:
                    missing_pct = df[col].isna().mean() * 100
                    validation['missing_data'][col] = missing_pct
                    if missing_pct > 10:
                        validation['issues'].append(f"High missing data in {col}: {missing_pct:.1f}%")
        
        # Calculate quality score
        quality_score = 100 - len(validation['issues']) * 10
        validation['data_quality_score'] = max(0, quality_score)
        
        return validation
    
    def save_data(self, teams_df: pd.DataFrame, players_df: pd.DataFrame) -> Dict[str, str]:
        """Save processed NBA data in multiple formats."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_files = {}
        
        logger.info("💾 Saving processed NBA data")
        
        try:
            # Save teams data
            if not teams_df.empty:
                teams_parquet = self.latest_dir / f"nba_teams_{timestamp}.parquet"
                teams_df.to_parquet(teams_parquet, index=False)
                saved_files['teams_parquet'] = str(teams_parquet)
                
                teams_csv = self.latest_dir / "nba_teams_latest.csv"
                teams_df.to_csv(teams_csv, index=False)
                saved_files['teams_csv'] = str(teams_csv)
                
                logger.info(f"✅ Teams data saved: {len(teams_df)} records")
            
            # Save players data
            if not players_df.empty:
                players_parquet = self.latest_dir / f"nba_players_{timestamp}.parquet"
                players_df.to_parquet(players_parquet, index=False)
                saved_files['players_parquet'] = str(players_parquet)
                
                players_csv = self.latest_dir / "nba_players_latest.csv"
                players_df.to_csv(players_csv, index=False)
                saved_files['players_csv'] = str(players_csv)
                
                logger.info(f"✅ Players data saved: {len(players_df)} records")
            
            # Save metadata
            metadata = {
                'ingestion_date': datetime.now().isoformat(),
                'seasons_processed': sorted(players_df['season'].unique().tolist()) if not players_df.empty else [],
                'total_teams': len(teams_df),
                'total_players': len(players_df),
                'unique_players': players_df['name'].nunique() if 'name' in players_df.columns else 0,
                'data_source': 'ESPN API + Website Scraping',
                'sport': 'NBA',
                'files_created': saved_files
            }
            
            metadata_file = self.latest_dir / f"nba_metadata_{timestamp}.json"
            import json
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            saved_files['metadata'] = str(metadata_file)
            
        except Exception as e:
            logger.error(f"❌ Error saving data: {e}")
            raise
        
        return saved_files
    
    def run_full_ingestion(self) -> Dict[str, Any]:
        """Run complete NBA data ingestion process."""
        logger.info("🚀 Starting NBA data ingestion with ESPN API + Web Scraping")
        start_time = datetime.now()
        
        results = {
            'status': 'unknown',
            'start_time': start_time.isoformat(),
            'seasons_processed': [],
            'teams_count': 0,
            'players_count': 0,
            'unique_players': 0,
            'files_saved': {},
            'data_quality': {},
            'errors': []
        }
        
        try:
            # Get seasons to process
            seasons = self.get_seasons_to_process()
            results['seasons_processed'] = seasons
            logger.info(f"Processing seasons: {seasons}")
            
            # Fetch teams data using ESPN API
            teams_df = self.fetch_teams_data(seasons)
            results['teams_count'] = len(teams_df)
            
            # Fetch players data using web scraping
            players_df = self.fetch_players_data(seasons, teams_df)
            results['players_count'] = len(players_df)
            results['unique_players'] = players_df['name'].nunique() if 'name' in players_df.columns else 0
            
            # Create training features
            if not players_df.empty and not teams_df.empty:
                players_df = self.create_training_features(players_df, teams_df)
            
            # Validate data quality
            results['data_quality']['teams'] = self.validate_data_quality(teams_df, 'teams')
            results['data_quality']['players'] = self.validate_data_quality(players_df, 'players')
            
            # Save data
            if not teams_df.empty or not players_df.empty:
                results['files_saved'] = self.save_data(teams_df, players_df)
                results['status'] = 'success'
            else:
                results['status'] = 'no_data'
                results['errors'].append("No data was retrieved")
            
        except Exception as e:
            logger.error(f"❌ Ingestion failed: {e}")
            results['status'] = 'error'
            results['errors'].append(str(e))
        
        # Calculate duration
        end_time = datetime.now()
        results['end_time'] = end_time.isoformat()
        results['duration_minutes'] = (end_time - start_time).total_seconds() / 60
        
        logger.info(f"🏁 NBA ingestion completed in {results['duration_minutes']:.1f} minutes")
        logger.info(f"Status: {results['status']}")
        logger.info(f"Teams: {results['teams_count']}, Players: {results['players_count']}")
        logger.info(f"Unique Players: {results['unique_players']}")
        
        return results


def main():
    """Main execution function."""
    print("🏀 NBA Data Ingestion for Model Training (ESPN Hybrid Version)")
    print("=" * 65)
    
    # Initialize and run ingestion
    ingest = NBADataIngest()
    results = ingest.run_full_ingestion()
    
    # Print detailed summary
    print(f"\n📊 INGESTION SUMMARY")
    print(f"Status: {results['status']}")
    print(f"Seasons: {results['seasons_processed']}")
    print(f"Teams: {results['teams_count']}")
    print(f"Players: {results['players_count']}")
    print(f"Unique Players: {results['unique_players']}")
    print(f"Duration: {results['duration_minutes']:.1f} minutes")
    
    if results['teams_count'] > 0 and results['players_count'] > 0:
        ratio = results['players_count'] / results['teams_count']
        seasons_count = len(results['seasons_processed'])
        players_per_team_per_season = ratio / seasons_count if seasons_count > 0 else ratio
        
        print(f"Players per team per season: {players_per_team_per_season:.1f}")
        print(f"Total ratio (all seasons): {ratio:.1f}")
        
        if players_per_team_per_season >= 10:
            print("✅ Excellent player coverage from web scraping!")
        elif players_per_team_per_season >= 8:
            print("✅ Good player coverage from web scraping!")
        else:
            print("⚠️  Lower than expected - some teams may have failed to scrape")
            print("   Check logs for failed team URLs or parsing issues")
    
    if results['files_saved']:
        print(f"\n📁 Files saved:")
        for key, path in results['files_saved'].items():
            print(f"  {key}: {path}")
    
    if results['errors']:
        print(f"\n❌ Errors:")
        for error in results['errors']:
            print(f"  {error}")
    
    return results


if __name__ == "__main__":
    main()
