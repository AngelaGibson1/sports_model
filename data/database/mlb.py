# data/database/mlb.py

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import json

from config.settings import Settings

class MLBDatabase:
    """Manages MLB data storage and retrieval."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize MLB database connection."""
        self.db_path = db_path or Settings.MLB_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database tables
        self._create_tables()
        logger.info(f"⚾ MLB Database initialized: {self.db_path}")
    
    def _create_tables(self):
        """Create all necessary tables for MLB data."""
        with sqlite3.connect(self.db_path) as conn:
            # Games table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS games (
                    game_id INTEGER PRIMARY KEY,
                    date TEXT NOT NULL,
                    time TEXT,
                    season INTEGER,
                    inning INTEGER,
                    inning_half TEXT,
                    home_team_id INTEGER,
                    away_team_id INTEGER,
                    home_team_name TEXT,
                    away_team_name TEXT,
                    home_score INTEGER,
                    away_score INTEGER,
                    status TEXT,
                    venue TEXT,
                    city TEXT,
                    weather TEXT,
                    temperature REAL,
                    wind_speed REAL,
                    wind_direction TEXT,
                    humidity REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Teams table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS teams (
                    team_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    abbreviation TEXT,
                    city TEXT,
                    league TEXT,      -- 'AL', 'NL'
                    division TEXT,    -- 'East', 'Central', 'West'
                    logo TEXT,
                    founded INTEGER,
                    stadium TEXT,
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
                    weight INTEGER,
                    position TEXT,
                    jersey_number INTEGER,
                    bats TEXT,        -- 'L', 'R', 'S' (switch)
                    throws TEXT,      -- 'L', 'R'
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
                    runs_per_game REAL,
                    runs_allowed_per_game REAL,
                    batting_average REAL,
                    on_base_percentage REAL,
                    slugging_percentage REAL,
                    ops REAL,
                    home_runs_per_game REAL,
                    stolen_bases_per_game REAL,
                    earned_run_average REAL,
                    whip REAL,
                    strikeouts_per_nine REAL,
                    walks_per_nine REAL,
                    fielding_percentage REAL,
                    errors_per_game REAL,
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
                    -- Batting stats
                    at_bats INTEGER,
                    runs INTEGER,
                    hits INTEGER,
                    doubles INTEGER,
                    triples INTEGER,
                    home_runs INTEGER,
                    rbi INTEGER,
                    stolen_bases INTEGER,
                    caught_stealing INTEGER,
                    walks INTEGER,
                    strikeouts INTEGER,
                    batting_average REAL,
                    on_base_percentage REAL,
                    slugging_percentage REAL,
                    -- Pitching stats
                    wins INTEGER,
                    losses INTEGER,
                    saves INTEGER,
                    innings_pitched REAL,
                    hits_allowed INTEGER,
                    runs_allowed INTEGER,
                    earned_runs INTEGER,
                    walks_allowed INTEGER,
                    strikeouts_pitched INTEGER,
                    earned_run_average REAL,
                    whip REAL,
                    -- Fielding stats
                    putouts INTEGER,
                    assists INTEGER,
                    errors INTEGER,
                    fielding_percentage REAL,
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
            
            # Create indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_games_date ON games (date)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_games_season ON games (season)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_games_teams ON games (home_team_id, away_team_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_team_stats_season ON team_statistics (team_id, season)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_player_stats_season ON player_statistics (player_id, season)')
            
            conn.commit()
    
    def save_games(self, games_df: pd.DataFrame) -> int:
        """Save games data to database."""
        if games_df.empty:
            return 0
        
        with sqlite3.connect(self.db_path) as conn:
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
            
            save_columns = [col for col in column_mapping.keys() if col in games_data.columns]
            games_data = games_data[save_columns].rename(columns=column_mapping)
            
            games_data = games_data.fillna({
                'time': '',
                'home_score': 0,
                'away_score': 0,
                'status': 'Scheduled',
                'venue': '',
                'city': ''
            })
            
            games_data.to_sql('games', conn, if_exists='append', index=False, method='multi')
            
            logger.info(f"✅ Saved {len(games_data)} MLB games to database")
            return len(games_data)
    
    def get_historical_data(self, seasons: Optional[List[int]] = None) -> pd.DataFrame:
        """Get comprehensive historical data for model training."""
        base_query = """
        SELECT 
            g.*,
            ht.name as home_team_name,
            at.name as away_team_name,
            hts.runs_per_game as home_rpg,
            hts.runs_allowed_per_game as home_rapg,
            hts.batting_average as home_ba,
            hts.earned_run_average as home_era,
            hts.on_base_percentage as home_obp,
            hts.slugging_percentage as home_slg,
            ats.runs_per_game as away_rpg,
            ats.runs_allowed_per_game as away_rapg,
            ats.batting_average as away_ba,
            ats.earned_run_average as away_era,
            ats.on_base_percentage as away_obp,
            ats.slugging_percentage as away_slg,
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
        
        logger.info(f"✅ Retrieved {len(df)} historical MLB games")
        return df
    
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
            
            cursor.execute("SELECT COUNT(*) FROM teams")
            summary['total_teams'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM game_odds")
            summary['total_odds_records'] = cursor.fetchone()[0]
        
        return summary
    
    # Add other necessary methods similar to NBA database
    def save_teams(self, teams_df: pd.DataFrame) -> int:
        """Save teams data to database."""
        # Implementation similar to NBA
        pass
    
    def save_team_statistics(self, stats_df: pd.DataFrame) -> int:
        """Save team statistics to database."""
        # Implementation similar to NBA
        pass
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        """Remove old data to keep database size manageable."""
        # Implementation similar to NBA
        pass
