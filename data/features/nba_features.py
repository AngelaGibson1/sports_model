import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import json

from config.settings import Settings

class NBADatabase:
    """Manages NBA data storage and retrieval."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize NBA database connection."""
        self.db_path = db_path or Settings.NBA_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database tables
        self._create_tables()
        logger.info(f"✅ NBA Database initialized: {self.db_path}")
    
    def _create_tables(self):
        """Create all necessary tables for NBA data."""
        with sqlite3.connect(self.db_path) as conn:
            # Games table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS games (
                    game_id INTEGER PRIMARY KEY,
                    date TEXT NOT NULL,
                    time TEXT,
                    season INTEGER,
                    home_team_id INTEGER,
                    away_team_id INTEGER,
                    home_team_name TEXT,
                    away_team_name TEXT,
                    home_score INTEGER,
                    away_score INTEGER,
                    status TEXT,
                    venue TEXT,
                    city TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Teams table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS teams (
                    team_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    code TEXT,
                    city TEXT,
                    conference TEXT,
                    division TEXT,
                    logo TEXT,
                    founded INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Players table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS players (
                    player_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    firstname TEXT,
                    lastname TEXT,
                    age INTEGER,
                    height TEXT,
                    weight TEXT,
                    position TEXT,
                    jersey_number INTEGER,
                    country TEXT,
                    team_id INTEGER,
                    active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (team_id) REFERENCES teams (team_id)
                )
            ''')
            
            # Team statistics table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS team_statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_id INTEGER,
                    season INTEGER,
                    games_played INTEGER,
                    wins INTEGER,
                    losses INTEGER,
                    win_percentage REAL,
                    points_per_game REAL,
                    points_allowed_per_game REAL,
                    field_goal_percentage REAL,
                    three_point_percentage REAL,
                    free_throw_percentage REAL,
                    rebounds_per_game REAL,
                    assists_per_game REAL,
                    steals_per_game REAL,
                    blocks_per_game REAL,
                    turnovers_per_game REAL,
                    offensive_rating REAL,
                    defensive_rating REAL,
                    pace REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (team_id) REFERENCES teams (team_id)
                )
            ''')
            
            # Player statistics table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS player_statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id INTEGER,
                    team_id INTEGER,
                    season INTEGER,
                    games_played INTEGER,
                    minutes_per_game REAL,
                    points_per_game REAL,
                    rebounds_per_game REAL,
                    assists_per_game REAL,
                    steals_per_game REAL,
                    blocks_per_game REAL,
                    field_goal_percentage REAL,
                    three_point_percentage REAL,
                    free_throw_percentage REAL,
                    turnovers_per_game REAL,
                    player_efficiency_rating REAL,
                    true_shooting_percentage REAL,
                    usage_rate REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (player_id) REFERENCES players (player_id),
                    FOREIGN KEY (team_id) REFERENCES teams (team_id)
                )
            ''')
            
            # Game odds table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS game_odds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id INTEGER,
                    bookmaker TEXT,
                    market TEXT,
                    home_odds REAL,
                    away_odds REAL,
                    spread REAL,
                    total REAL,
                    last_update TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (game_id) REFERENCES games (game_id)
                )
            ''')
            
            # Player props table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS player_props (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id INTEGER,
                    player_id INTEGER,
                    prop_type TEXT,
                    line REAL,
                    over_odds REAL,
                    under_odds REAL,
                    bookmaker TEXT,
                    last_update TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (game_id) REFERENCES games (game_id),
                    FOREIGN KEY (player_id) REFERENCES players (player_id)
                )
            ''')
            
            # Create indexes for better performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_games_date ON games (date)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_games_season ON games (season)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_games_teams ON games (home_team_id, away_team_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_team_stats_season ON team_statistics (team_id, season)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_player_stats_season ON player_statistics (player_id, season)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_odds_game ON game_odds (game_id)')
            
            conn.commit()
    
    def save_games(self, games_df: pd.DataFrame) -> int:
        """
        Save games data to database.
        
        Args:
            games_df: DataFrame with game data
            
        Returns:
            Number of games saved
        """
        if games_df.empty:
            return 0
        
        with sqlite3.connect(self.db_path) as conn:
            # Prepare data
            games_data = games_df.copy()
            
            # Map DataFrame columns to database columns
            column_mapping = {
                'game_id': 'game_id',
                'date': 'date',
                'time': 'time',
                'season': 'season',
                'home_team_id': 'home_team_id',
                'away_team_id': 'away_team_id',
                'home_team_name': 'home_team_name',
                'away_team_name': 'away_team_name',
                'home_score': 'home_score',
                'away_score': 'away_score',
                'status': 'status',
                'venue': 'venue',
                'city': 'city'
            }
            
            # Select and rename columns
            save_columns = [col for col in column_mapping.keys() if col in games_data.columns]
            games_data = games_data[save_columns].rename(columns=column_mapping)
            
            # Handle missing values
            games_data = games_data.fillna({
                'time': '',
                'home_score': 0,
                'away_score': 0,
                'status': 'Scheduled',
                'venue': '',
                'city': ''
            })
            
            # Use INSERT OR REPLACE to handle duplicates
            games_data.to_sql('games', conn, if_exists='append', index=False, method='multi')
            
            logger.info(f"✅ Saved {len(games_data)} NBA games to database")
            return len(games_data)
    
    def save_teams(self, teams_df: pd.DataFrame) -> int:
        """Save teams data to database."""
        if teams_df.empty:
            return 0
        
        with sqlite3.connect(self.db_path) as conn:
            teams_data = teams_df.copy()
            
            # Map columns
            column_mapping = {
                'team_id': 'team_id',
                'name': 'name',
                'code': 'code',
                'city': 'city',
                'logo': 'logo',
                'founded': 'founded'
            }
            
            save_columns = [col for col in column_mapping.keys() if col in teams_data.columns]
            teams_data = teams_data[save_columns].rename(columns=column_mapping)
            
            # Handle NBA-specific team data
            if 'conference' not in teams_data.columns:
                teams_data['conference'] = self._infer_conference(teams_data)
            if 'division' not in teams_data.columns:
                teams_data['division'] = self._infer_division(teams_data)
            
            teams_data.to_sql('teams', conn, if_exists='replace', index=False)
            
            logger.info(f"✅ Saved {len(teams_data)} NBA teams to database")
            return len(teams_data)
    
    def save_team_statistics(self, stats_df: pd.DataFrame) -> int:
        """Save team statistics to database."""
        if stats_df.empty:
            return 0
        
        with sqlite3.connect(self.db_path) as conn:
            stats_data = stats_df.copy()
            
            # NBA-specific column mapping
            nba_columns = {
                'team_id': 'team_id',
                'season': 'season',
                'games': 'games_played',
                'wins': 'wins',
                'losses': 'losses',
                'winPercentage': 'win_percentage',
                'points': 'points_per_game',
                'pointsAgainst': 'points_allowed_per_game',
                'fieldGoalsPercentage': 'field_goal_percentage',
                'threePointersPercentage': 'three_point_percentage',
                'freeThrowsPercentage': 'free_throw_percentage',
                'reboundsTotal': 'rebounds_per_game',
                'assists': 'assists_per_game',
                'steals': 'steals_per_game',
                'blocks': 'blocks_per_game',
                'turnovers': 'turnovers_per_game',
                'offensiveRating': 'offensive_rating',
                'defensiveRating': 'defensive_rating',
                'pace': 'pace'
            }
            
            # Map available columns
            available_cols = [col for col in nba_columns.keys() if col in stats_data.columns]
            stats_data = stats_data[available_cols].rename(columns=nba_columns)
            
            stats_data.to_sql('team_statistics', conn, if_exists='append', index=False)
            
            logger.info(f"✅ Saved {len(stats_data)} NBA team statistics records")
            return len(stats_data)
    
    def save_player_statistics(self, stats_df: pd.DataFrame) -> int:
        """Save player statistics to database."""
        if stats_df.empty:
            return 0
        
        with sqlite3.connect(self.db_path) as conn:
            stats_data = stats_df.copy()
            
            # NBA player stats mapping
            player_columns = {
                'player_id': 'player_id',
                'team_id': 'team_id',
                'season': 'season',
                'games': 'games_played',
                'minutes': 'minutes_per_game',
                'points': 'points_per_game',
                'totReb': 'rebounds_per_game',
                'assists': 'assists_per_game',
                'steals': 'steals_per_game',
                'blocks': 'blocks_per_game',
                'fgp': 'field_goal_percentage',
                'tpp': 'three_point_percentage',
                'ftp': 'free_throw_percentage',
                'turnovers': 'turnovers_per_game',
                'per': 'player_efficiency_rating',
                'ts': 'true_shooting_percentage',
                'usg': 'usage_rate'
            }
            
            available_cols = [col for col in player_columns.keys() if col in stats_data.columns]
            stats_data = stats_data[available_cols].rename(columns=player_columns)
            
            stats_data.to_sql('player_statistics', conn, if_exists='append', index=False)
            
            logger.info(f"✅ Saved {len(stats_data)} NBA player statistics records")
            return len(stats_data)
    
    def save_games_with_odds(self, games_df: pd.DataFrame) -> int:
        """Save games with embedded odds data."""
        if games_df.empty:
            return 0
        
        # First save the games
        games_saved = self.save_games(games_df)
        
        # Then extract and save odds if present
        odds_columns = [col for col in games_df.columns if 'odds' in col.lower() or 'spread' in col.lower()]
        
        if odds_columns:
            odds_data = []
            
            for _, row in games_df.iterrows():
                game_id = row.get('game_id')
                
                if pd.notna(game_id):
                    # Extract moneyline odds
                    if 'moneyline_home_odds' in row and 'moneyline_away_odds' in row:
                        odds_data.append({
                            'game_id': game_id,
                            'bookmaker': 'average',
                            'market': 'moneyline',
                            'home_odds': row.get('moneyline_home_odds'),
                            'away_odds': row.get('moneyline_away_odds'),
                            'spread': None,
                            'total': None,
                            'last_update': datetime.now()
                        })
                    
                    # Extract spread odds
                    if 'spread_home_odds' in row and 'spread_away_odds' in row:
                        odds_data.append({
                            'game_id': game_id,
                            'bookmaker': 'average',
                            'market': 'spread',
                            'home_odds': row.get('spread_home_odds'),
                            'away_odds': row.get('spread_away_odds'),
                            'spread': row.get('spread_line', 0),
                            'total': None,
                            'last_update': datetime.now()
                        })
                    
                    # Extract totals odds
                    if 'totals_over_odds' in row and 'totals_under_odds' in row:
                        odds_data.append({
                            'game_id': game_id,
                            'bookmaker': 'average',
                            'market': 'totals',
                            'home_odds': row.get('totals_over_odds'),
                            'away_odds': row.get('totals_under_odds'),
                            'spread': None,
                            'total': row.get('totals_line'),
                            'last_update': datetime.now()
                        })
            
            if odds_data:
                odds_df = pd.DataFrame(odds_data)
                with sqlite3.connect(self.db_path) as conn:
                    odds_df.to_sql('game_odds', conn, if_exists='append', index=False)
                logger.info(f"✅ Saved {len(odds_data)} NBA odds records")
        
        return games_saved
    
    def get_games(self, 
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None,
                  season: Optional[int] = None,
                  team_id: Optional[int] = None) -> pd.DataFrame:
        """Retrieve games from database."""
        
        query = """
        SELECT g.*, 
               ht.name as home_team_full_name,
               at.name as away_team_full_name
        FROM games g
        LEFT JOIN teams ht ON g.home_team_id = ht.team_id
        LEFT JOIN teams at ON g.away_team_id = at.team_id
        WHERE 1=1
        """
        
        params = []
        
        if start_date:
            query += " AND g.date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND g.date <= ?"
            params.append(end_date)
        if season:
            query += " AND g.season = ?"
            params.append(season)
        if team_id:
            query += " AND (g.home_team_id = ? OR g.away_team_id = ?)"
            params.extend([team_id, team_id])
        
        query += " ORDER BY g.date DESC, g.time DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def get_historical_data(self, seasons: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Get comprehensive historical data for model training.
        
        Args:
            seasons: List of seasons to include
            
        Returns:
            DataFrame with games, team stats, and outcomes
        """
        base_query = """
        SELECT 
            g.*,
            ht.name as home_team_name,
            at.name as away_team_name,
            hts.wins as home_wins,
            hts.losses as home_losses,
            hts.win_percentage as home_win_pct,
            hts.points_per_game as home_ppg,
            hts.points_allowed_per_game as home_papg,
            hts.offensive_rating as home_off_rating,
            hts.defensive_rating as home_def_rating,
            hts.pace as home_pace,
            ats.wins as away_wins,
            ats.losses as away_losses,
            ats.win_percentage as away_win_pct,
            ats.points_per_game as away_ppg,
            ats.points_allowed_per_game as away_papg,
            ats.offensive_rating as away_off_rating,
            ats.defensive_rating as away_def_rating,
            ats.pace as away_pace,
            CASE WHEN g.home_score > g.away_score THEN 1 ELSE 0 END as home_win
        FROM games g
        LEFT JOIN teams ht ON g.home_team_id = ht.team_id
        LEFT JOIN teams at ON g.away_team_id = at.team_id
        LEFT JOIN team_statistics hts ON g.home_team_id = hts.team_id AND g.season = hts.season
        LEFT JOIN team_statistics ats ON g.away_team_id = ats.team_id AND g.season = ats.season
        WHERE g.status = 'Finished'
        """
        
        params = []
        if seasons:
            placeholders = ','.join(['?' for _ in seasons])
            base_query += f" AND g.season IN ({placeholders})"
            params.extend(seasons)
        
        base_query += " ORDER BY g.date, g.time"
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(base_query, conn, params=params)
        
        logger.info(f"✅ Retrieved {len(df)} historical NBA games")
        return df
    
    def get_team_statistics(self, 
                           season: int,
                           team_id: Optional[int] = None) -> pd.DataFrame:
        """Get team statistics for a season."""
        
        query = """
        SELECT ts.*, t.name as team_name, t.code as team_code
        FROM team_statistics ts
        JOIN teams t ON ts.team_id = t.team_id
        WHERE ts.season = ?
        """
        
        params = [season]
        
        if team_id:
            query += " AND ts.team_id = ?"
            params.append(team_id)
        
        query += " ORDER BY ts.win_percentage DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def get_player_statistics(self, 
                             season: int,
                             player_id: Optional[int] = None,
                             team_id: Optional[int] = None) -> pd.DataFrame:
        """Get player statistics for a season."""
        
        query = """
        SELECT ps.*, p.name as player_name, p.position, t.name as team_name
        FROM player_statistics ps
        JOIN players p ON ps.player_id = p.player_id
        JOIN teams t ON ps.team_id = t.team_id
        WHERE ps.season = ?
        """
        
        params = [season]
        
        if player_id:
            query += " AND ps.player_id = ?"
            params.append(player_id)
        if team_id:
            query += " AND ps.team_id = ?"
            params.append(team_id)
        
        query += " ORDER BY ps.points_per_game DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of data in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            summary = {}
            
            # Games summary
            cursor.execute("SELECT COUNT(*) FROM games")
            summary['total_games'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT season) FROM games")
            summary['seasons_available'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT MIN(date), MAX(date) FROM games")
            date_range = cursor.fetchone()
            summary['date_range'] = {'start': date_range[0], 'end': date_range[1]}
            
            # Teams summary
            cursor.execute("SELECT COUNT(*) FROM teams")
            summary['total_teams'] = cursor.fetchone()[0]
            
            # Recent games
            cursor.execute("""
                SELECT COUNT(*) FROM games 
                WHERE date >= date('now', '-7 days')
            """)
            summary['recent_games_7d'] = cursor.fetchone()[0]
            
            # Statistics coverage
            cursor.execute("SELECT COUNT(DISTINCT season) FROM team_statistics")
            summary['team_stats_seasons'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT season) FROM player_statistics")
            summary['player_stats_seasons'] = cursor.fetchone()[0]
            
            # Odds coverage
            cursor.execute("SELECT COUNT(*) FROM game_odds")
            summary['total_odds_records'] = cursor.fetchone()[0]
        
        return summary
    
    def _infer_conference(self, teams_df: pd.DataFrame) -> pd.Series:
        """Infer NBA conference from team data."""
        # NBA conference mapping (simplified)
        eastern_teams = ['ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DET', 'IND', 
                        'MIA', 'MIL', 'NYK', 'ORL', 'PHI', 'TOR', 'WAS']
        
        def get_conference(row):
            if 'code' in row and row['code'] in eastern_teams:
                return 'Eastern'
            elif 'name' in row:
                for east_team in eastern_teams:
                    if east_team.lower() in row['name'].lower():
                        return 'Eastern'
            return 'Western'
        
        return teams_df.apply(get_conference, axis=1)
    
    def _infer_division(self, teams_df: pd.DataFrame) -> pd.Series:
        """Infer NBA division from team data."""
        # NBA division mapping (simplified)
        divisions = {
            'Atlantic': ['BOS', 'BKN', 'NYK', 'PHI', 'TOR'],
            'Central': ['CHI', 'CLE', 'DET', 'IND', 'MIL'],
            'Southeast': ['ATL', 'CHA', 'MIA', 'ORL', 'WAS'],
            'Northwest': ['DEN', 'MIN', 'OKC', 'POR', 'UTA'],
            'Pacific': ['GSW', 'LAC', 'LAL', 'PHX', 'SAC'],
            'Southwest': ['DAL', 'HOU', 'MEM', 'NOP', 'SAS']
        }
        
        def get_division(row):
            team_code = row.get('code', '')
            for division, teams in divisions.items():
                if team_code in teams:
                    return division
            return 'Unknown'
        
        return teams_df.apply(get_division, axis=1)
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        """Remove old data to keep database size manageable."""
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Delete old games and related data
            cursor.execute("DELETE FROM game_odds WHERE game_id IN (SELECT game_id FROM games WHERE date < ?)", (cutoff_date,))
            cursor.execute("DELETE FROM player_props WHERE game_id IN (SELECT game_id FROM games WHERE date < ?)", (cutoff_date,))
            cursor.execute("DELETE FROM games WHERE date < ?", (cutoff_date,))
            
            conn.commit()
            
            logger.info(f"✅ Cleaned up NBA data older than {cutoff_date}")
    
    def vacuum_database(self):
        """Optimize database storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("VACUUM")
            logger.info("✅ NBA database optimized")
