# nfl_data_ingest.py
"""
NFL Data Ingestion Script for Model Training - FIXED VERSION
Retrieves comprehensive player stats, team data, and rosters for the last 5 seasons.
Fixed to use the working SportsAPIClient instead of problematic ESPN endpoints.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger
import time

# Import the working API client
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # Add parent directory to path

from api_clients.sports_api import SportsAPIClient


class NFLDataIngest:
    """NFL data ingestion class optimized for ML training datasets using the working API client."""
    
    def __init__(self, output_dir: str = "data"):
        """
        Initialize NFL data ingestion.
        
        Args:
            output_dir: Base directory for logs and data files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create latest data directory for ML training files
        self.latest_dir = self.output_dir / "latest"
        self.latest_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the working SportsAPIClient for NFL
        self.client = SportsAPIClient('nfl')
        
        # Setup logging
        log_file = self.output_dir / f"nfl_ingest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger.add(log_file, rotation="10 MB", retention="30 days")
        
        logger.info("🏈 NFL Data Ingestion initialized with working API client")
    
    def get_seasons_to_process(self) -> List[int]:
        """Get the last 5 NFL seasons including current."""
        try:
            seasons = self.client.get_seasons()
            logger.info(f"Available seasons from API: {seasons}")
            
            # Filter to get numeric seasons and take the last 5
            numeric_seasons = []
            for season in seasons:
                if isinstance(season, (int, str)):
                    try:
                        year = int(str(season).split('-')[0])  # Handle "2023-2024" format
                        numeric_seasons.append(year)
                    except ValueError:
                        continue
            
            numeric_seasons = sorted(list(set(numeric_seasons)))
            
            if len(numeric_seasons) >= 5:
                return numeric_seasons[-5:]
            else:
                return numeric_seasons
                
        except Exception as e:
            logger.warning(f"Could not get seasons from API: {e}")
            # Fallback: NFL season spans calendar years, starts in September
            current_date = datetime.now()
            if current_date.month >= 9:  # NFL season starts in September
                current_season = current_date.year
            else:
                current_season = current_date.year - 1
            
            return list(range(current_season - 4, current_season + 1))
    
    def clean_player_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize NFL player data for ML training."""
        if df.empty:
            return df
        
        # Standardize column names
        df.columns = [col.lower().replace(' ', '_').replace('.', '_') for col in df.columns]
        
        # NFL-specific numeric columns (including stats that might be present)
        numeric_cols = [col for col in df.columns if any(stat in col.lower() for stat in 
                       ['yards', 'touchdowns', 'tds', 'receptions', 'carries', 'attempts',
                        'completions', 'passing', 'rushing', 'receiving', 'tackles', 'sacks',
                        'interceptions', 'fumbles', 'field_goals', 'extra_points', 'punts',
                        'returns', 'rating', 'qbr', 'completion_percentage', 'yards_per',
                        'average', 'longest', 'games_played', 'games_started', 'points',
                        'age', 'weight', 'jersey_number', 'experience'])]
        
        for col in numeric_cols:
            if col in df.columns:
                # Handle percentage columns
                if any(pct in col.lower() for pct in ['pct', 'percentage', '%']):
                    df[col] = pd.to_numeric(df[col], errors='coerce') / 100
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clean text fields
        text_cols = ['name', 'firstname', 'lastname', 'fullname', 'shortname', 'position', 
                    'position_name', 'status', 'injury_status', 'birth_city', 'birth_state', 
                    'birth_country', 'college', 'college_mascot']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace('nan', '')
        
        # NFL-specific position standardization
        if 'position' in df.columns:
            df['position_standard'] = df['position'].apply(self._standardize_nfl_position)
            df['position_group'] = df['position_standard'].apply(self._get_position_group)
            df['is_offensive'] = df['position_group'].isin(['QB', 'RB', 'WR', 'TE', 'OL'])
            df['is_defensive'] = df['position_group'].isin(['DL', 'LB', 'DB'])
            df['is_special_teams'] = df['position_group'].isin(['K', 'P', 'LS'])
        
        # Add derived features for ML
        if 'height' in df.columns:
            df['height_inches'] = df['height'].apply(self._convert_height_to_inches)
        
        if 'weight' in df.columns:
            df['weight_lbs'] = pd.to_numeric(df['weight'], errors='coerce')
            if 'height_inches' in df.columns:
                df['bmi'] = df.apply(lambda x: self._calculate_bmi(x['weight_lbs'], x['height_inches']), axis=1)
        
        # Experience-based features
        if 'experience' in df.columns:
            df['is_rookie'] = (df['experience'] == 0) | (df['experience'].isna())
            df['is_veteran'] = df['experience'] >= 5
            df['experience_group'] = df['experience'].apply(self._categorize_experience)
        
        # Add season-based features
        if 'season' in df.columns:
            df['career_year'] = df['season'] - df.groupby('player_id')['season'].transform('min') + 1
            df['is_current_season'] = df['season'] == df['season'].max()
        
        # Jersey number insights
        if 'jersey_number' in df.columns:
            df['jersey_number_numeric'] = pd.to_numeric(df['jersey_number'], errors='coerce')
            df['jersey_number_range'] = df['jersey_number_numeric'].apply(self._categorize_jersey_number)
        
        return df
    
    def _categorize_experience(self, years: float) -> str:
        """Categorize player experience level."""
        if pd.isna(years):
            return 'Unknown'
        elif years == 0:
            return 'Rookie'
        elif years <= 2:
            return 'Young'
        elif years <= 5:
            return 'Developing'
        elif years <= 10:
            return 'Veteran'
        else:
            return 'Elder'
    
    def _categorize_jersey_number(self, number: float) -> str:
        """Categorize jersey number by traditional NFL position ranges."""
        if pd.isna(number):
            return 'Unknown'
        
        number = int(number)
        if 1 <= number <= 9:
            return 'Kickers/Punters'
        elif 10 <= number <= 19:
            return 'QBs/Kickers/Punters'
        elif 20 <= number <= 49:
            return 'RBs/DBs'
        elif 50 <= number <= 59:
            return 'Centers/LBs'
        elif 60 <= number <= 79:
            return 'Offensive Line'
        elif 80 <= number <= 89:
            return 'WRs/TEs'
        elif 90 <= number <= 99:
            return 'Defensive Line/LBs'
        else:
            return 'Other'
    
    def _standardize_nfl_position(self, position: str) -> str:
        """Standardize NFL position abbreviations."""
        if pd.isna(position) or position == '':
            return 'Unknown'
        
        pos = str(position).upper().strip()
        
        # Map common variations
        position_map = {
            # Offense
            'QUARTERBACK': 'QB', 'QB': 'QB',
            'RUNNING BACK': 'RB', 'RB': 'RB', 'FULLBACK': 'FB', 'FB': 'FB',
            'WIDE RECEIVER': 'WR', 'WR': 'WR',
            'TIGHT END': 'TE', 'TE': 'TE',
            'CENTER': 'C', 'C': 'C',
            'GUARD': 'G', 'G': 'G', 'LEFT GUARD': 'LG', 'LG': 'LG', 'RIGHT GUARD': 'RG', 'RG': 'RG',
            'TACKLE': 'T', 'T': 'T', 'LEFT TACKLE': 'LT', 'LT': 'LT', 'RIGHT TACKLE': 'RT', 'RT': 'RT',
            'OFFENSIVE LINE': 'OL', 'OL': 'OL',
            
            # Defense
            'DEFENSIVE END': 'DE', 'DE': 'DE',
            'DEFENSIVE TACKLE': 'DT', 'DT': 'DT',
            'NOSE TACKLE': 'NT', 'NT': 'NT',
            'LINEBACKER': 'LB', 'LB': 'LB',
            'MIDDLE LINEBACKER': 'MLB', 'MLB': 'MLB',
            'OUTSIDE LINEBACKER': 'OLB', 'OLB': 'OLB',
            'CORNERBACK': 'CB', 'CB': 'CB',
            'SAFETY': 'S', 'S': 'S',
            'FREE SAFETY': 'FS', 'FS': 'FS',
            'STRONG SAFETY': 'SS', 'SS': 'SS',
            
            # Special Teams
            'KICKER': 'K', 'K': 'K',
            'PUNTER': 'P', 'P': 'P',
            'LONG SNAPPER': 'LS', 'LS': 'LS'
        }
        
        return position_map.get(pos, pos)
    
    def _get_position_group(self, position: str) -> str:
        """Get position group for NFL positions."""
        if pd.isna(position):
            return 'Unknown'
        
        groups = {
            'QB': ['QB'],
            'RB': ['RB', 'FB', 'HB'],
            'WR': ['WR', 'WR/RS', 'WR/PR'],
            'TE': ['TE'],
            'OL': ['C', 'G', 'LG', 'RG', 'T', 'LT', 'RT', 'OL', 'OT', 'OG'],
            'DL': ['DE', 'DT', 'NT', 'DL'],
            'LB': ['LB', 'MLB', 'OLB', 'ILB'],
            'DB': ['CB', 'S', 'FS', 'SS', 'DB'],
            'K': ['K', 'PK'],
            'P': ['P'],
            'LS': ['LS', 'LP']
        }
        
        for group, positions in groups.items():
            if position in positions:
                return group
        
        return 'Other'
    
    def _convert_height_to_inches(self, height: str) -> Optional[float]:
        """Convert height from feet-inches format to total inches."""
        if pd.isna(height) or height == '':
            return None
        
        try:
            if "'" in str(height) and '"' in str(height):
                # Format: 6'2"
                parts = str(height).replace('"', '').split("'")
                feet = int(parts[0])
                inches = int(parts[1]) if parts[1] else 0
                return feet * 12 + inches
            elif '-' in str(height):
                # Format: 6-2
                parts = str(height).split('-')
                feet = int(parts[0])
                inches = int(parts[1]) if len(parts) > 1 else 0
                return feet * 12 + inches
            else:
                # Try to parse as total inches
                return float(height)
        except:
            return None
    
    def _calculate_bmi(self, weight_lbs: float, height_inches: float) -> Optional[float]:
        """Calculate BMI from weight (lbs) and height (inches)."""
        if pd.isna(weight_lbs) or pd.isna(height_inches) or height_inches <= 0:
            return None
        
        try:
            # Convert to metric (kg, meters) for BMI calculation
            weight_kg = weight_lbs * 0.453592
            height_m = height_inches * 0.0254
            return weight_kg / (height_m ** 2)
        except:
            return None
    
    def fetch_teams_data(self, seasons: List[int]) -> pd.DataFrame:
        """Fetch NFL team data for all seasons using the working API client."""
        logger.info(f"📊 Fetching NFL team data for seasons: {seasons}")
        
        all_teams = []
        
        for season in seasons:
            try:
                logger.info(f"Getting teams for {season}")
                teams = self.client.get_teams(season=season)
                
                if not teams.empty:
                    teams['season'] = season
                    teams['data_source'] = 'API-Sports'
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
    
    def clean_team_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean NFL team data for consistency."""
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
        if 'name' in df.columns:
            df['conference'] = df.apply(self._assign_nfl_conference, axis=1)
            df['division'] = df.apply(self._assign_nfl_division, axis=1)
        
        return df
    
    def _assign_nfl_conference(self, row) -> str:
        """Assign NFL conference based on team."""
        # Build team name from available columns
        team_name = ""
        if 'city' in row and pd.notna(row['city']):
            team_name += str(row['city']).lower()
        if 'name' in row and pd.notna(row['name']):
            team_name += " " + str(row['name']).lower()
        
        afc_teams = {
            'buffalo', 'miami', 'new england', 'new york jets', 'jets',
            'baltimore', 'cincinnati', 'cleveland', 'pittsburgh',
            'houston', 'indianapolis', 'jacksonville', 'tennessee',
            'denver', 'kansas city', 'las vegas', 'los angeles chargers', 'chargers'
        }
        
        if any(team in team_name for team in afc_teams):
            return 'AFC'
        else:
            return 'NFC'
    
    def _assign_nfl_division(self, row) -> str:
        """Assign NFL division based on team."""
        # Build team name from available columns
        team_name = ""
        if 'city' in row and pd.notna(row['city']):
            team_name += str(row['city']).lower()
        if 'name' in row and pd.notna(row['name']):
            team_name += " " + str(row['name']).lower()
        
        divisions = {
            'AFC East': ['buffalo', 'miami', 'new england', 'new york jets', 'jets'],
            'AFC North': ['baltimore', 'cincinnati', 'cleveland', 'pittsburgh'],
            'AFC South': ['houston', 'indianapolis', 'jacksonville', 'tennessee'],
            'AFC West': ['denver', 'kansas city', 'las vegas', 'los angeles chargers', 'chargers'],
            'NFC East': ['dallas', 'new york giants', 'giants', 'philadelphia', 'washington'],
            'NFC North': ['chicago', 'detroit', 'green bay', 'minnesota'],
            'NFC South': ['atlanta', 'carolina', 'new orleans', 'tampa bay'],
            'NFC West': ['arizona', 'los angeles rams', 'rams', 'san francisco', 'seattle']
        }
        
        for division, teams in divisions.items():
            if any(team in team_name for team in teams):
                return division
        
        return 'Unknown'
    
    def fetch_players_data(self, seasons: List[int]) -> pd.DataFrame:
        """Fetch comprehensive NFL player data using the working SportsAPIClient."""
        logger.info(f"🏈 Fetching NFL player data for seasons: {seasons}")

        all_players = []

        for season in seasons:
            try:
                logger.info(f"Getting players for {season}")
                
                # First, get teams for this season
                teams = self.client.get_teams(season=season)
                
                if teams.empty:
                    logger.warning(f"No teams found for {season}, skipping player fetch")
                    continue
                
                season_players = []
                
                # Try to get players from multiple approaches
                # Approach 1: Try to get all players for the season
                try:
                    players = self.client.get_players(season=season)
                    if not players.empty:
                        season_players.append(players)
                        logger.info(f"Got {len(players)} players from season query")
                except Exception as e:
                    logger.debug(f"Season-wide player query failed: {e}")
                
                # Approach 2: Get players by team
                for _, team in teams.iterrows():
                    team_id = team.get('team_id', team.get('id'))
                    if team_id:
                        try:
                            team_players = self.client.get_players(team_id=team_id, season=season)
                            if not team_players.empty:
                                season_players.append(team_players)
                                logger.debug(f"Got {len(team_players)} players from team {team_id}")
                            time.sleep(0.3)  # Rate limiting
                        except Exception as e:
                            logger.debug(f"Team {team_id} player query failed: {e}")
                            continue
                
                # Combine all player data for this season
                if season_players:
                    combined_season = pd.concat(season_players, ignore_index=True)
                    # Remove duplicates based on player_id
                    combined_season = combined_season.drop_duplicates(subset=['player_id'])
                    
                    combined_season['season'] = season
                    combined_season['data_source'] = 'API-Sports'
                    combined_season['ingestion_date'] = datetime.now()
                    all_players.append(combined_season)
                    logger.info(f"✅ Found {len(combined_season)} unique players for {season}")
                else:
                    logger.warning(f"❌ No players found for {season}")

                time.sleep(1.0)  # Pause between seasons

            except Exception as e:
                logger.error(f"❌ Error fetching players for {season}: {e}")
                continue

        if all_players:
            combined_players = pd.concat(all_players, ignore_index=True)
            # Final deduplication by player_id and season
            combined_players = combined_players.drop_duplicates(subset=['player_id', 'season'])
            logger.info(f"✅ Total unique player records collected: {len(combined_players)}")
            return self.clean_player_data(combined_players)
        else:
            logger.error("❌ No player data collected")
            return pd.DataFrame()
    
    def create_training_features(self, players_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for NFL ML training."""
        if players_df.empty:
            return players_df
        
        logger.info("🔧 Creating NFL training features")
        
        # Merge team information
        if not teams_df.empty:
            # Prepare team mapping with consistent column names
            team_cols = []
            if 'team_id' in teams_df.columns:
                team_cols.append('team_id')
            elif 'id' in teams_df.columns:
                teams_df['team_id'] = teams_df['id']
                team_cols.append('team_id')
            
            if team_cols and 'season' in teams_df.columns:
                team_mapping = teams_df[['team_id', 'season', 'name', 'conference', 'division']].drop_duplicates()
                team_mapping.columns = ['team_id', 'season', 'team_name', 'team_conference', 'team_division']
                
                # Merge with players
                if 'team_id' in players_df.columns:
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
            players_df['career_games'] = players_df.groupby('player_id')['season'].transform('count')
            
            # Experience vs career tracking
            if 'experience' in players_df.columns:
                players_df['experience_vs_seasons'] = players_df['experience'] - players_df['seasons_played']
                players_df['has_career_gap'] = players_df['experience_vs_seasons'] > 1
        
        # Position-specific features
        if 'position_group' in players_df.columns:
            # Create dummy variables for position groups
            position_dummies = pd.get_dummies(players_df['position_group'], prefix='pos')
            players_df = pd.concat([players_df, position_dummies], axis=1)
        
        # College-based features
        if 'college' in players_df.columns:
            # Top college programs (NFL production)
            top_colleges = ['Alabama', 'Ohio State', 'LSU', 'Georgia', 'Clemson', 'Notre Dame', 'USC', 'Miami']
            players_df['elite_college'] = players_df['college'].isin(top_colleges)
            
            # Power 5 conferences (major college programs)
            power5_keywords = ['State', 'University', 'Tech', 'College']
            players_df['power5_college'] = players_df['college'].str.contains('|'.join(power5_keywords), na=False)
        
        # Size-based features for different position groups
        if all(col in players_df.columns for col in ['height_inches', 'weight_lbs', 'position_group']):
            # Calculate position-relative size metrics
            position_avg_height = players_df.groupby('position_group')['height_inches'].transform('mean')
            position_avg_weight = players_df.groupby('position_group')['weight_lbs'].transform('mean')
            
            players_df['height_vs_position'] = players_df['height_inches'] - position_avg_height
            players_df['weight_vs_position'] = players_df['weight_lbs'] - position_avg_weight
            
            # Size categories relative to position
            players_df['size_vs_position'] = 'average'
            players_df.loc[
                (players_df['height_vs_position'] > 1) & (players_df['weight_vs_position'] > 10), 
                'size_vs_position'
            ] = 'large'
            players_df.loc[
                (players_df['height_vs_position'] < -1) & (players_df['weight_vs_position'] < -10), 
                'size_vs_position'
            ] = 'small'
        
        # Draft and development insights
        if 'experience' in players_df.columns and 'age' in players_df.columns:
            # Estimated draft age
            players_df['estimated_draft_age'] = players_df['age'] - players_df['experience'] - 1
            players_df['young_draftee'] = players_df['estimated_draft_age'] <= 21
            players_df['old_draftee'] = players_df['estimated_draft_age'] >= 24
        
        return players_df
    
    def validate_data_quality(self, df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """Validate NFL data quality for ML training."""
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
            'players': ['player_id', 'name', 'season', 'position'],
            'teams': ['team_id', 'name', 'season']
        }
        
        if data_type in key_columns:
            for col in key_columns[data_type]:
                if col in df.columns:
                    missing_pct = df[col].isna().mean() * 100
                    validation['missing_data'][col] = missing_pct
                    if missing_pct > 10:
                        validation['issues'].append(f"High missing data in {col}: {missing_pct:.1f}%")
        
        # NFL-specific validations
        if data_type == 'players':
            # Check for realistic height/weight for NFL players
            if 'height_inches' in df.columns:
                invalid_height = ((df['height_inches'] < 60) | (df['height_inches'] > 85)).sum()
                if invalid_height > 0:
                    validation['issues'].append(f"Found {invalid_height} players with unrealistic heights")
            
            if 'weight_lbs' in df.columns:
                invalid_weight = ((df['weight_lbs'] < 150) | (df['weight_lbs'] > 400)).sum()
                if invalid_weight > 0:
                    validation['issues'].append(f"Found {invalid_weight} players with unrealistic weights")
        
        # Check for duplicates
        if 'player_id' in df.columns and 'season' in df.columns:
            duplicates = df.duplicated(subset=['player_id', 'season']).sum()
            if duplicates > 0:
                validation['issues'].append(f"Found {duplicates} duplicate player-season records")
        
        # Calculate quality score
        quality_score = 100 - len(validation['issues']) * 10
        validation['data_quality_score'] = max(0, quality_score)
        
        return validation
    
    def save_data(self, teams_df: pd.DataFrame, players_df: pd.DataFrame) -> Dict[str, str]:
        """Save processed NFL data in multiple formats."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_files = {}
        
        logger.info("💾 Saving processed NFL data")
        
        try:
            # Save teams data
            if not teams_df.empty:
                # Parquet for efficient ML loading
                teams_parquet = self.latest_dir / f"nfl_teams_{timestamp}.parquet"
                teams_df.to_parquet(teams_parquet, index=False)
                saved_files['teams_parquet'] = str(teams_parquet)
                
                # CSV for human readability
                teams_csv = self.latest_dir / "nfl_teams_latest.csv"
                teams_df.to_csv(teams_csv, index=False)
                saved_files['teams_csv'] = str(teams_csv)
                
                logger.info(f"✅ Teams data saved: {len(teams_df)} records")
            
            # Save players data
            if not players_df.empty:
                # Parquet for efficient ML loading
                players_parquet = self.latest_dir / f"nfl_players_{timestamp}.parquet"
                players_df.to_parquet(players_parquet, index=False)
                saved_files['players_parquet'] = str(players_parquet)
                
                # CSV for human readability
                players_csv = self.latest_dir / "nfl_players_latest.csv"
                players_df.to_csv(players_csv, index=False)
                saved_files['players_csv'] = str(players_csv)
                
                logger.info(f"✅ Players data saved: {len(players_df)} records")
            
            # Save metadata
            metadata = {
                'ingestion_date': datetime.now().isoformat(),
                'seasons_processed': sorted(players_df['season'].unique().tolist()) if not players_df.empty else [],
                'total_teams': len(teams_df),
                'total_players': len(players_df),
                'data_source': 'API-Sports',
                'sport': 'NFL',
                'files_created': saved_files
            }
            
            metadata_file = self.latest_dir / f"nfl_metadata_{timestamp}.json"
            import json
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            saved_files['metadata'] = str(metadata_file)
            
        except Exception as e:
            logger.error(f"❌ Error saving data: {e}")
            raise
        
        return saved_files
    
    def run_full_ingestion(self) -> Dict[str, Any]:
        """Run complete NFL data ingestion process."""
        logger.info("🚀 Starting NFL data ingestion with working API client")
        start_time = datetime.now()
        
        results = {
            'status': 'unknown',
            'start_time': start_time.isoformat(),
            'seasons_processed': [],
            'teams_count': 0,
            'players_count': 0,
            'files_saved': {},
            'data_quality': {},
            'errors': []
        }
        
        try:
            # Get seasons to process
            seasons = self.get_seasons_to_process()
            results['seasons_processed'] = seasons
            logger.info(f"Processing seasons: {seasons}")
            
            # Fetch teams data
            teams_df = self.fetch_teams_data(seasons)
            results['teams_count'] = len(teams_df)
            
            # Fetch players data
            players_df = self.fetch_players_data(seasons)
            results['players_count'] = len(players_df)
            
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
        
        logger.info(f"🏁 NFL ingestion completed in {results['duration_minutes']:.1f} minutes")
        logger.info(f"Status: {results['status']}")
        logger.info(f"Teams: {results['teams_count']}, Players: {results['players_count']}")
        
        return results


def main():
    """Main execution function."""
    print("🏈 NFL Data Ingestion for Model Training - FIXED VERSION")
    print("=" * 50)
    
    # Initialize and run ingestion
    ingest = NFLDataIngest()
    results = ingest.run_full_ingestion()
    
    # Print summary
    print(f"\n📊 INGESTION SUMMARY")
    print(f"Status: {results['status']}")
    print(f"Seasons: {results['seasons_processed']}")
    print(f"Teams: {results['teams_count']}")
    print(f"Players: {results['players_count']}")
    print(f"Duration: {results['duration_minutes']:.1f} minutes")
    
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
