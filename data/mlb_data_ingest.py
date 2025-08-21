# mlb_data_ingest_fixed.py
"""
MLB Data Ingestion Script - FIXED VERSION
Uses comprehensive fetching strategies to get ALL players like the successful NFL script.
Implements pagination, multiple categories, and team-by-team fallbacks.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from loguru import logger
import time

# Import your ESPN API client
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from api_clients.espn_api import ESPNAPIClient


class MLBDataIngestComprehensive:
    """MLB data ingestion with comprehensive player fetching like the successful NFL script."""
    
    def __init__(self, output_dir: str = "data"):
        """Initialize MLB comprehensive data ingestion."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create latest data directory
        self.latest_dir = self.output_dir / "latest"
        self.latest_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ESPN client
        self.client = ESPNAPIClient(sport='mlb')
        
        # Setup comprehensive logging
        log_file = self.output_dir / f"mlb_comprehensive_ingest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger.add(log_file, rotation="10 MB", retention="30 days", level="DEBUG")
        
        # Track unique players to avoid duplicates
        self.all_players: Set[str] = set()
        self.player_data: List[Dict] = []
        
        logger.info("🏟️ MLB Comprehensive Data Ingestion initialized")
    
    def get_seasons_to_process(self) -> List[int]:
        """Get the last 5 seasons including current."""
        try:
            seasons = self.client.get_seasons()
            if len(seasons) >= 5:
                return seasons[-5:]
            else:
                return seasons
        except Exception as e:
            logger.warning(f"Could not get seasons from API: {e}")
            # Fallback to current year and 4 previous
            current_year = datetime.now().year
            return list(range(current_year - 4, current_year + 1))
    
    def fetch_players_comprehensive(self, seasons: List[int]) -> pd.DataFrame:
        """
        Fetch ALL MLB players using comprehensive strategies like the successful NFL script.
        Uses multiple approaches to ensure we get every single player.
        """
        logger.info(f"⚾ Starting comprehensive MLB player fetch for seasons: {seasons}")
        
        all_players = []
        
        for season in seasons:
            logger.info(f"🔍 Processing season {season} with multiple strategies...")
            season_players = []
            
            # STRATEGY 1: Get players by multiple categories (MLB has batting/pitching splits)
            categories = ['batting', 'pitching', 'fielding', 'general', 'all']
            
            for category in categories:
                try:
                    logger.info(f"  📊 Trying category: {category}")
                    
                    # Try paginated fetching for this category
                    page = 1
                    max_pages = 50  # Safety limit
                    
                    while page <= max_pages:
                        try:
                            # Use ESPN client with category and pagination
                            players = self.client.get_players(
                                season=season,
                                category=category,
                                page=page
                            )
                            
                            if not players.empty:
                                players['fetch_category'] = category
                                players['fetch_page'] = page
                                season_players.append(players)
                                logger.debug(f"    ✅ Page {page}: {len(players)} players")
                                page += 1
                                time.sleep(0.5)  # Rate limiting
                            else:
                                logger.debug(f"    🔚 No more players on page {page}")
                                break
                                
                        except Exception as e:
                            logger.debug(f"    ❌ Page {page} failed: {e}")
                            break
                    
                except Exception as e:
                    logger.debug(f"  ❌ Category {category} failed: {e}")
                    continue
            
            # STRATEGY 2: Team-by-team fetching as fallback
            try:
                logger.info(f"  🏟️ Trying team-by-team approach...")
                teams = self.client.get_teams(season=season)
                
                if not teams.empty:
                    for _, team in teams.iterrows():
                        team_id = team.get('team_id', team.get('id'))
                        team_name = team.get('name', 'Unknown')
                        
                        if team_id:
                            try:
                                # Get roster for this team
                                team_players = self.client.get_team_roster(
                                    team_id=team_id,
                                    season=season
                                )
                                
                                if not team_players.empty:
                                    team_players['fetch_method'] = 'team_roster'
                                    team_players['source_team_id'] = team_id
                                    season_players.append(team_players)
                                    logger.debug(f"    ✅ {team_name}: {len(team_players)} players")
                                
                                time.sleep(0.3)  # Rate limiting
                                
                            except Exception as e:
                                logger.debug(f"    ❌ {team_name} roster failed: {e}")
                                continue
                        
            except Exception as e:
                logger.debug(f"  ❌ Team-by-team approach failed: {e}")
            
            # STRATEGY 3: Direct season-wide fetch with different parameters
            try:
                logger.info(f"  🌐 Trying direct season fetch...")
                direct_players = self.client.get_players(season=season)
                
                if not direct_players.empty:
                    direct_players['fetch_method'] = 'direct_season'
                    season_players.append(direct_players)
                    logger.debug(f"    ✅ Direct fetch: {len(direct_players)} players")
                    
            except Exception as e:
                logger.debug(f"  ❌ Direct season fetch failed: {e}")
            
            # Combine and deduplicate season data
            if season_players:
                combined_season = pd.concat(season_players, ignore_index=True)
                
                # Deduplicate by player_id (keep first occurrence)
                combined_season = combined_season.drop_duplicates(subset=['player_id'])
                
                # Add season info
                combined_season['season'] = season
                combined_season['data_source'] = 'ESPN'
                combined_season['ingestion_date'] = datetime.now()
                
                all_players.append(combined_season)
                logger.info(f"✅ Season {season}: {len(combined_season)} unique players")
            else:
                logger.warning(f"❌ No players found for season {season}")
            
            time.sleep(1.0)  # Pause between seasons
        
        # Final combination and deduplication
        if all_players:
            final_players = pd.concat(all_players, ignore_index=True)
            # Final dedup by player_id and season
            final_players = final_players.drop_duplicates(subset=['player_id', 'season'])
            
            logger.info(f"🎯 COMPREHENSIVE FETCH COMPLETE!")
            logger.info(f"📊 Total unique player-season records: {len(final_players)}")
            logger.info(f"👥 Unique players across all seasons: {final_players['player_id'].nunique()}")
            
            return self.clean_player_data(final_players)
        else:
            logger.error("❌ No player data collected from any strategy")
            return pd.DataFrame()
    
    def clean_player_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize player data for ML training."""
        if df.empty:
            return df
        
        logger.info("🧹 Cleaning player data...")
        
        # Standardize column names
        df.columns = [col.lower().replace(' ', '_').replace('.', '_') for col in df.columns]
        
        # Convert numeric columns specific to MLB
        numeric_cols = [col for col in df.columns if any(stat in col.lower() for stat in 
                       ['avg', 'era', 'obp', 'slg', 'ops', 'hr', 'rbi', 'runs', 'hits', 
                        'doubles', 'triples', 'walks', 'strikeouts', 'stolen_bases',
                        'wins', 'losses', 'saves', 'innings', 'whip', 'k9', 'bb9', 'fip',
                        'war', 'wrc', 'babip', 'iso', 'woba', 'batting_average',
                        'earned_run_average', 'on_base_percentage', 'slugging_percentage'])]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clean text fields
        text_cols = ['name', 'firstname', 'lastname', 'position', 'team', 'country']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace('nan', '')
        
        # MLB-specific position standardization
        if 'position' in df.columns:
            df['position_standard'] = df['position'].apply(self._standardize_mlb_position)
            df['is_pitcher'] = df['position_standard'].isin(['SP', 'RP', 'CP', 'P'])
            df['is_catcher'] = df['position_standard'].isin(['C'])
            df['is_infielder'] = df['position_standard'].isin(['1B', '2B', '3B', 'SS'])
            df['is_outfielder'] = df['position_standard'].isin(['LF', 'CF', 'RF', 'OF'])
        
        # Add derived features
        if 'height' in df.columns:
            df['height_inches'] = df['height'].apply(self._convert_height_to_inches)
        
        # Determine player type based on available stats
        df['player_type'] = self._determine_player_type(df)
        
        # Add season-based features
        if 'season' in df.columns and 'player_id' in df.columns:
            df = df.sort_values(['player_id', 'season'])
            df['career_year'] = df['season'] - df.groupby('player_id')['season'].transform('min') + 1
            df['is_current_season'] = df['season'] == df['season'].max()
        
        logger.info(f"✅ Data cleaning complete: {len(df)} records")
        return df
    
    def _standardize_mlb_position(self, position: str) -> str:
        """Standardize MLB position abbreviations."""
        if pd.isna(position) or position == '':
            return 'Unknown'
        
        pos = str(position).upper().strip()
        
        position_map = {
            'PITCHER': 'P', 'P': 'P',
            'STARTING PITCHER': 'SP', 'SP': 'SP',
            'RELIEF PITCHER': 'RP', 'RP': 'RP',
            'CLOSER': 'CP', 'CP': 'CP', 'CL': 'CP',
            'CATCHER': 'C', 'C': 'C',
            'FIRST BASE': '1B', '1B': '1B', 'FIRST BASEMAN': '1B',
            'SECOND BASE': '2B', '2B': '2B', 'SECOND BASEMAN': '2B',
            'THIRD BASE': '3B', '3B': '3B', 'THIRD BASEMAN': '3B',
            'SHORTSTOP': 'SS', 'SS': 'SS',
            'LEFT FIELD': 'LF', 'LF': 'LF', 'LEFT FIELDER': 'LF',
            'CENTER FIELD': 'CF', 'CF': 'CF', 'CENTER FIELDER': 'CF',
            'RIGHT FIELD': 'RF', 'RF': 'RF', 'RIGHT FIELDER': 'RF',
            'OUTFIELD': 'OF', 'OF': 'OF', 'OUTFIELDER': 'OF',
            'DESIGNATED HITTER': 'DH', 'DH': 'DH'
        }
        
        return position_map.get(pos, pos)
    
    def _determine_player_type(self, df: pd.DataFrame) -> pd.Series:
        """Determine if player is primarily a batter or pitcher based on stats."""
        player_types = []
        
        for _, row in df.iterrows():
            # Check for pitching stats
            has_pitching = any(col for col in df.columns if any(stat in col.lower() for stat in 
                             ['era', 'wins', 'losses', 'saves', 'innings_pitched', 'whip', 
                              'strikeouts_pitched', 'walks_allowed', 'earned_run_average']))
            
            # Check for batting stats
            has_batting = any(col for col in df.columns if any(stat in col.lower() for stat in 
                            ['batting_avg', 'home_runs', 'rbi', 'hits', 'runs', 'obp', 'slg',
                             'on_base_percentage', 'slugging_percentage', 'stolen_bases']))
            
            if has_pitching and not has_batting:
                player_types.append('pitcher')
            elif has_batting and not has_pitching:
                player_types.append('batter')
            elif has_pitching and has_batting:
                player_types.append('two_way')
            else:
                player_types.append('unknown')
        
        return pd.Series(player_types)
    
    def _convert_height_to_inches(self, height: str) -> Optional[float]:
        """Convert height from feet-inches format to total inches."""
        if pd.isna(height) or height == '':
            return None
        
        try:
            if "'" in str(height) and '"' in str(height):
                parts = str(height).replace('"', '').split("'")
                feet = int(parts[0])
                inches = int(parts[1]) if parts[1] else 0
                return feet * 12 + inches
            elif '-' in str(height):
                parts = str(height).split('-')
                feet = int(parts[0])
                inches = int(parts[1]) if len(parts) > 1 else 0
                return feet * 12 + inches
            else:
                return float(height)
        except:
            return None
    
    def fetch_teams_data(self, seasons: List[int]) -> pd.DataFrame:
        """Fetch team data for all seasons."""
        logger.info(f"📊 Fetching team data for seasons: {seasons}")
        
        all_teams = []
        
        for season in seasons:
            try:
                logger.info(f"Getting teams for {season}")
                teams = self.client.get_teams(season=season)
                
                if not teams.empty:
                    teams['season'] = season
                    teams['data_source'] = 'ESPN'
                    teams['ingestion_date'] = datetime.now()
                    all_teams.append(teams)
                    logger.info(f"✅ Found {len(teams)} teams for {season}")
                else:
                    logger.warning(f"❌ No teams found for {season}")
                
                time.sleep(0.5)
                
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
        """Clean team data for consistency."""
        if df.empty:
            return df
        
        # Standardize column names
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        # Clean text fields
        text_cols = ['name', 'code', 'city', 'country']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Add league and division information
        if 'name' in df.columns:
            df['league'] = df.apply(self._assign_mlb_league, axis=1)
            df['division'] = df.apply(self._assign_mlb_division, axis=1)
        
        return df
    
    def _assign_mlb_league(self, row) -> str:
        """Assign MLB league (AL/NL) based on team."""
        al_teams = {
            'yankees', 'red sox', 'blue jays', 'orioles', 'rays',
            'guardians', 'tigers', 'royals', 'twins', 'white sox',
            'astros', 'angels', 'athletics', 'mariners', 'rangers'
        }
        
        team_name = str(row.get('name', '')).lower()
        if any(team in team_name for team in al_teams):
            return 'American League'
        else:
            return 'National League'
    
    def _assign_mlb_division(self, row) -> str:
        """Assign MLB division based on team."""
        divisions = {
            'AL East': ['yankees', 'red sox', 'blue jays', 'orioles', 'rays'],
            'AL Central': ['guardians', 'tigers', 'royals', 'twins', 'white sox'],
            'AL West': ['astros', 'angels', 'athletics', 'mariners', 'rangers'],
            'NL East': ['braves', 'marlins', 'mets', 'phillies', 'nationals'],
            'NL Central': ['cubs', 'reds', 'brewers', 'pirates', 'cardinals'],
            'NL West': ['diamondbacks', 'rockies', 'dodgers', 'padres', 'giants']
        }
        
        team_name = str(row.get('name', '')).lower()
        for division, teams in divisions.items():
            if any(team in team_name for team in teams):
                return division
        
        return 'Unknown'
    
    def create_training_features(self, players_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for ML training."""
        if players_df.empty:
            return players_df
        
        logger.info("🔧 Creating training features")
        
        # Merge team information
        if not teams_df.empty:
            team_mapping = teams_df[['team_id', 'season', 'name', 'city', 'league', 'division']].drop_duplicates()
            team_mapping.columns = ['team_id', 'season', 'team_name', 'team_city', 'team_league', 'team_division']
            
            players_df = players_df.merge(
                team_mapping,
                on=['team_id', 'season'],
                how='left'
            )
        
        # Career progression features
        if 'player_id' in players_df.columns and 'season' in players_df.columns:
            players_df = players_df.sort_values(['player_id', 'season'])
            
            players_df['seasons_played'] = players_df.groupby('player_id').cumcount() + 1
            players_df['career_games'] = players_df.groupby('player_id')['season'].transform('count')
            
            # Age-related features
            if 'age' in players_df.columns:
                players_df['age_numeric'] = pd.to_numeric(players_df['age'], errors='coerce')
                players_df['is_rookie'] = players_df['seasons_played'] == 1
                players_df['is_veteran'] = players_df['age_numeric'] >= 30
                players_df['prime_years'] = (players_df['age_numeric'] >= 25) & (players_df['age_numeric'] <= 29)
        
        return players_df
    
    def validate_data_quality(self, df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """Validate data quality for ML training."""
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
            'players': ['player_id', 'name', 'season'],
            'teams': ['team_id', 'name', 'season']
        }
        
        if data_type in key_columns:
            for col in key_columns[data_type]:
                if col in df.columns:
                    missing_pct = df[col].isna().mean() * 100
                    validation['missing_data'][col] = missing_pct
                    if missing_pct > 10:
                        validation['issues'].append(f"High missing data in {col}: {missing_pct:.1f}%")
        
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
        """Save processed data in multiple formats."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_files = {}
        
        logger.info("💾 Saving processed data")
        
        try:
            # Save teams data
            if not teams_df.empty:
                teams_parquet = self.latest_dir / f"mlb_teams_{timestamp}.parquet"
                teams_df.to_parquet(teams_parquet, index=False)
                saved_files['teams_parquet'] = str(teams_parquet)
                
                teams_csv = self.latest_dir / "mlb_teams_latest.csv"
                teams_df.to_csv(teams_csv, index=False)
                saved_files['teams_csv'] = str(teams_csv)
                
                logger.info(f"✅ Teams data saved: {len(teams_df)} records")
            
            # Save players data
            if not players_df.empty:
                players_parquet = self.latest_dir / f"mlb_players_{timestamp}.parquet"
                players_df.to_parquet(players_parquet, index=False)
                saved_files['players_parquet'] = str(players_parquet)
                
                players_csv = self.latest_dir / "mlb_players_latest.csv"
                players_df.to_csv(players_csv, index=False)
                saved_files['players_csv'] = str(players_csv)
                
                logger.info(f"✅ Players data saved: {len(players_df)} records")
            
            # Save metadata
            metadata = {
                'ingestion_date': datetime.now().isoformat(),
                'seasons_processed': sorted(players_df['season'].unique().tolist()) if not players_df.empty else [],
                'total_teams': len(teams_df),
                'total_players': len(players_df),
                'unique_players': players_df['player_id'].nunique() if not players_df.empty else 0,
                'data_source': 'ESPN',
                'sport': 'MLB',
                'files_created': saved_files,
                'fetch_strategies_used': ['category_pagination', 'team_by_team', 'direct_season']
            }
            
            metadata_file = self.latest_dir / f"mlb_metadata_{timestamp}.json"
            import json
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            saved_files['metadata'] = str(metadata_file)
            
        except Exception as e:
            logger.error(f"❌ Error saving data: {e}")
            raise
        
        return saved_files
    
    def run_full_ingestion(self) -> Dict[str, Any]:
        """Run complete comprehensive MLB data ingestion."""
        logger.info("🚀 Starting comprehensive MLB data ingestion")
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
            
            # Fetch teams data
            teams_df = self.fetch_teams_data(seasons)
            results['teams_count'] = len(teams_df)
            
            # Fetch players data using comprehensive approach
            players_df = self.fetch_players_comprehensive(seasons)
            results['players_count'] = len(players_df)
            results['unique_players'] = players_df['player_id'].nunique() if not players_df.empty else 0
            
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
        
        logger.info(f"🏁 MLB comprehensive ingestion completed in {results['duration_minutes']:.1f} minutes")
        logger.info(f"Status: {results['status']}")
        logger.info(f"Teams: {results['teams_count']}")
        logger.info(f"Total Player Records: {results['players_count']}")
        logger.info(f"Unique Players: {results['unique_players']}")
        
        return results


def main():
    """Main execution function."""
    print("⚾ MLB Comprehensive Data Ingestion - ALL PLAYERS")
    print("=" * 60)
    print("🎯 Using the same comprehensive approach that got 12,061 NFL players")
    
    # Initialize and run ingestion
    ingest = MLBDataIngestComprehensive()
    results = ingest.run_full_ingestion()
    
    # Print summary
    print(f"\n📊 COMPREHENSIVE INGESTION SUMMARY")
    print(f"Status: {results['status']}")
    print(f"Seasons: {results['seasons_processed']}")
    print(f"Teams: {results['teams_count']}")
    print(f"Total Player Records: {results['players_count']}")
    print(f"Unique Players: {results['unique_players']}")
    print(f"Duration: {results['duration_minutes']:.1f} minutes")
    
    if results['files_saved']:
        print(f"\n📁 Files saved:")
        for key, path in results['files_saved'].items():
            print(f"  {key}: {path}")
    
    if results['errors']:
        print(f"\n❌ Errors:")
        for error in results['errors']:
            print(f"  {error}")
    
    # Compare to NFL success
    if results['status'] == 'success':
        if results['players_count'] > 10000:
            print(f"\n🎉 SUCCESS! Got {results['players_count']} players - similar to NFL's 12,061!")
        elif results['players_count'] > 5000:
            print(f"\n✅ Good progress! Got {results['players_count']} players - working toward NFL's 12,061")
        else:
            print(f"\n⚠️ Only got {results['players_count']} players - still investigating why less than NFL's 12,061")
    
    return results


if __name__ == "__main__":
    main()
