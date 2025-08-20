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
            
            # Starting pitchers table - NEW
            conn.execute('''
                CREATE TABLE IF NOT EXISTS game_starting_pitchers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id INTEGER,
                    home_starter_id INTEGER,
                    away_starter_id INTEGER,
                    home_starter_confirmed BOOLEAN DEFAULT FALSE,
                    away_starter_confirmed BOOLEAN DEFAULT FALSE,
                    estimation_method TEXT DEFAULT 'manual',
                    confidence_score REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (game_id) REFERENCES games (game_id),
                    FOREIGN KEY (home_starter_id) REFERENCES players (player_id),
                    FOREIGN KEY (away_starter_id) REFERENCES players (player_id)
                )
            ''')
            
            # Create indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_games_date ON games (date)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_games_season ON games (season)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_games_teams ON games (home_team_id, away_team_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_team_stats_season ON team_statistics (team_id, season)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_player_stats_season ON player_statistics (player_id, season)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_odds_game ON game_odds (game_id)')
            
            # Pitcher indexes - NEW
            conn.execute('CREATE INDEX IF NOT EXISTS idx_game_pitchers_game_id ON game_starting_pitchers (game_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_game_pitchers_home ON game_starting_pitchers (home_starter_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_game_pitchers_away ON game_starting_pitchers (away_starter_id)')
            
            conn.commit()

    # ============= SAVE METHODS =============

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
                'abbreviation': 'abbreviation',
                'city': 'city',
                'logo': 'logo',
                'founded': 'founded'
            }
            
            save_columns = [col for col in column_mapping.keys() if col in teams_data.columns]
            teams_data = teams_data[save_columns].rename(columns=column_mapping)
            
            # Add MLB-specific team data
            teams_data = self._add_mlb_team_info(teams_data)
            
            teams_data.to_sql('teams', conn, if_exists='replace', index=False)
            
            logger.info(f"✅ Saved {len(teams_data)} MLB teams to database")
            return len(teams_data)
    
    def save_team_statistics(self, stats_df: pd.DataFrame) -> int:
        """Save team statistics to database."""
        if stats_df.empty:
            return 0
        
        with sqlite3.connect(self.db_path) as conn:
            stats_data = stats_df.copy()
            
            # MLB-specific column mapping
            mlb_columns = {
                'team_id': 'team_id',
                'season': 'season',
                'games_played': 'games_played',
                'wins': 'wins',
                'losses': 'losses',
                'win_percentage': 'win_percentage',
                'runs_per_game': 'runs_per_game',
                'runs_allowed_per_game': 'runs_allowed_per_game',
                'batting_average': 'batting_average',
                'on_base_percentage': 'on_base_percentage',
                'slugging_percentage': 'slugging_percentage',
                'earned_run_average': 'earned_run_average',
                'whip': 'whip',
                'strikeouts_per_nine': 'strikeouts_per_nine',
                'walks_per_nine': 'walks_per_nine',
                'fielding_percentage': 'fielding_percentage',
                'home_runs_per_game': 'home_runs_per_game',
                'stolen_bases_per_game': 'stolen_bases_per_game',
                'errors_per_game': 'errors_per_game'
            }
            
            # Map available columns
            available_cols = [col for col in mlb_columns.keys() if col in stats_data.columns]
            stats_data = stats_data[available_cols].rename(columns=mlb_columns)
            
            # Calculate derived metrics
            if 'on_base_percentage' in stats_data.columns and 'slugging_percentage' in stats_data.columns:
                stats_data['ops'] = stats_data['on_base_percentage'] + stats_data['slugging_percentage']
            
            stats_data.to_sql('team_statistics', conn, if_exists='append', index=False)
            
            logger.info(f"✅ Saved {len(stats_data)} MLB team statistics records")
            return len(stats_data)
    
    def save_player_statistics(self, stats_df: pd.DataFrame) -> int:
        """Save player statistics to database."""
        if stats_df.empty:
            return 0
        
        with sqlite3.connect(self.db_path) as conn:
            stats_data = stats_df.copy()
            
            # MLB player stats mapping
            player_columns = {
                'player_id': 'player_id',
                'team_id': 'team_id',
                'season': 'season',
                'games_played': 'games_played',
                # Batting
                'at_bats': 'at_bats',
                'runs': 'runs',
                'hits': 'hits',
                'doubles': 'doubles',
                'triples': 'triples',
                'home_runs': 'home_runs',
                'rbi': 'rbi',
                'stolen_bases': 'stolen_bases',
                'caught_stealing': 'caught_stealing',
                'walks': 'walks',
                'strikeouts': 'strikeouts',
                'batting_average': 'batting_average',
                'on_base_percentage': 'on_base_percentage',
                'slugging_percentage': 'slugging_percentage',
                # Pitching
                'wins': 'wins',
                'losses': 'losses',
                'saves': 'saves',
                'innings_pitched': 'innings_pitched',
                'hits_allowed': 'hits_allowed',
                'runs_allowed': 'runs_allowed',
                'earned_runs': 'earned_runs',
                'walks_allowed': 'walks_allowed',
                'strikeouts_pitched': 'strikeouts_pitched',
                'earned_run_average': 'earned_run_average',
                'whip': 'whip',
                # Fielding
                'putouts': 'putouts',
                'assists': 'assists',
                'errors': 'errors',
                'fielding_percentage': 'fielding_percentage'
            }
            
            available_cols = [col for col in player_columns.keys() if col in stats_data.columns]
            stats_data = stats_data[available_cols].rename(columns=player_columns)
            
            stats_data.to_sql('player_statistics', conn, if_exists='append', index=False)
            
            logger.info(f"✅ Saved {len(stats_data)} MLB player statistics records")
            return len(stats_data)

    def save_starting_pitchers(self, pitchers_df: pd.DataFrame) -> int:
        """Save starting pitcher information for games."""
        if pitchers_df.empty:
            return 0
        
        with sqlite3.connect(self.db_path) as conn:
            pitcher_data = pitchers_df.copy()
            
            # Required columns: game_id, home_starter_id, away_starter_id
            required_cols = ['game_id', 'home_starter_id', 'away_starter_id']
            if not all(col in pitcher_data.columns for col in required_cols):
                logger.error(f"Missing required columns: {required_cols}")
                return 0
            
            pitcher_data.to_sql('game_starting_pitchers', conn, if_exists='append', index=False)
            logger.info(f"✅ Saved {len(pitcher_data)} starting pitcher records")
            return len(pitcher_data)

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
                    
                    # Extract run line odds
                    if 'runline_home_odds' in row and 'runline_away_odds' in row:
                        odds_data.append({
                            'game_id': game_id,
                            'bookmaker': 'average',
                            'market': 'runline',
                            'home_odds': row.get('runline_home_odds'),
                            'away_odds': row.get('runline_away_odds'),
                            'spread': row.get('runline_spread', -1.5),
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
                logger.info(f"✅ Saved {len(odds_data)} MLB odds records")
        
        return games_saved

    # ============= GET METHODS =============
    
    def get_games(self, 
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None,
                  season: Optional[int] = None,
                  team_id: Optional[int] = None) -> pd.DataFrame:
        """Retrieve games from database."""
        
        query = """
        SELECT g.*, 
               ht.name as home_team_full_name,
               ht.abbreviation as home_team_abbr,
               ht.league as home_league,
               ht.division as home_division,
               at.name as away_team_full_name,
               at.abbreviation as away_team_abbr,
               at.league as away_league,
               at.division as away_division
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
        """Get comprehensive historical data for model training."""
        base_query = """
        SELECT 
            g.*,
            ht.name as home_team_name,
            ht.abbreviation as home_team_abbr,
            ht.league as home_league,
            ht.division as home_division,
            at.name as away_team_name,
            at.abbreviation as away_team_abbr,
            at.league as away_league,
            at.division as away_division,
            hts.runs_per_game as home_rpg,
            hts.runs_allowed_per_game as home_rapg,
            hts.batting_average as home_ba,
            hts.earned_run_average as home_era,
            hts.on_base_percentage as home_obp,
            hts.slugging_percentage as home_slg,
            hts.ops as home_ops,
            hts.whip as home_whip,
            hts.fielding_percentage as home_fp,
            ats.runs_per_game as away_rpg,
            ats.runs_allowed_per_game as away_rapg,
            ats.batting_average as away_ba,
            ats.earned_run_average as away_era,
            ats.on_base_percentage as away_obp,
            ats.slugging_percentage as away_slg,
            ats.ops as away_ops,
            ats.whip as away_whip,
            ats.fielding_percentage as away_fp,
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
    
    def get_team_statistics(self, 
                           season: int,
                           team_id: Optional[int] = None) -> pd.DataFrame:
        """Get team statistics for a season."""
        
        query = """
        SELECT ts.*, t.name as team_name, t.abbreviation as team_abbr,
               t.league, t.division
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
                             team_id: Optional[int] = None,
                             position: Optional[str] = None) -> pd.DataFrame:
        """Get player statistics for a season."""
        
        query = """
        SELECT ps.*, p.name as player_name, p.position, p.bats, p.throws,
               t.name as team_name, t.abbreviation as team_abbr
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
        if position:
            query += " AND p.position = ?"
            params.append(position)
        
        query += " ORDER BY ps.batting_average DESC, ps.earned_run_average ASC"
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_pitcher_statistics_for_season(self, season: int, min_innings: float = 50.0) -> pd.DataFrame:
        """Get pitcher statistics for a season with minimum innings filter."""
        query = """
        SELECT 
            ps.*,
            p.name as pitcher_name,
            p.throws as pitcher_hand,
            p.age,
            t.name as team_name,
            t.abbreviation as team_abbr
        FROM player_statistics ps
        JOIN players p ON ps.player_id = p.player_id
        JOIN teams t ON ps.team_id = t.team_id
        WHERE ps.season = ? 
        AND ps.innings_pitched >= ?
        AND ps.innings_pitched IS NOT NULL
        ORDER BY ps.earned_run_average ASC
        """
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=[season, min_innings])

    def get_games_with_pitcher_data(self, 
                                   start_date: Optional[str] = None,
                                   end_date: Optional[str] = None,
                                   season: Optional[int] = None,
                                   include_unconfirmed: bool = False) -> pd.DataFrame:
        """Get games with starting pitcher information included."""
        
        base_query = """
        SELECT 
            g.*,
            ht.name as home_team_full_name,
            ht.abbreviation as home_team_abbr,
            at.name as away_team_full_name,
            at.abbreviation as away_team_abbr,
            gsp.home_starter_id,
            gsp.away_starter_id,
            hp.name as home_starter_name,
            hp.throws as home_starter_hand,
            ap.name as away_starter_name,
            ap.throws as away_starter_hand,
            -- Home pitcher current season stats
            hps.earned_run_average as home_starter_era,
            hps.whip as home_starter_whip,
            hps.innings_pitched as home_starter_ip,
            hps.strikeouts_pitched as home_starter_k,
            hps.walks_allowed as home_starter_bb,
            hps.wins as home_starter_w,
            hps.losses as home_starter_l,
            -- Away pitcher current season stats
            aps.earned_run_average as away_starter_era,
            aps.whip as away_starter_whip,
            aps.innings_pitched as away_starter_ip,
            aps.strikeouts_pitched as away_starter_k,
            aps.walks_allowed as away_starter_bb,
            aps.wins as away_starter_w,
            aps.losses as away_starter_l
        FROM games g
        LEFT JOIN teams ht ON g.home_team_id = ht.team_id
        LEFT JOIN teams at ON g.away_team_id = at.team_id
        LEFT JOIN game_starting_pitchers gsp ON g.game_id = gsp.game_id
        LEFT JOIN players hp ON gsp.home_starter_id = hp.player_id
        LEFT JOIN players ap ON gsp.away_starter_id = ap.player_id
        LEFT JOIN player_statistics hps ON hp.player_id = hps.player_id AND g.season = hps.season
        LEFT JOIN player_statistics aps ON ap.player_id = aps.player_id AND g.season = aps.season
        WHERE 1=1
        """
        
        params = []
        
        if not include_unconfirmed:
            base_query += " AND gsp.home_starter_id IS NOT NULL AND gsp.away_starter_id IS NOT NULL"
        
        if start_date:
            base_query += " AND g.date >= ?"
            params.append(start_date)
        if end_date:
            base_query += " AND g.date <= ?"
            params.append(end_date)
        if season:
            base_query += " AND g.season = ?"
            params.append(season)
        
        base_query += " ORDER BY g.date DESC, g.time DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(base_query, conn, params=params)
        
        # Calculate advanced pitcher metrics
        if not df.empty:
            df = self._add_calculated_pitcher_metrics(df)
        
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
            
            # Pitcher coverage - NEW
            cursor.execute("SELECT COUNT(*) FROM game_starting_pitchers")
            summary['total_pitcher_assignments'] = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(DISTINCT gsp.game_id) as games_with_pitchers,
                       COUNT(DISTINCT g.game_id) as total_finished_games
                FROM games g
                LEFT JOIN game_starting_pitchers gsp ON g.game_id = gsp.game_id
                WHERE g.status = 'Finished'
            """)
            pitcher_coverage = cursor.fetchone()
            if pitcher_coverage[1] > 0:
                summary['pitcher_coverage_pct'] = (pitcher_coverage[0] / pitcher_coverage[1]) * 100
            else:
                summary['pitcher_coverage_pct'] = 0
            
            # League breakdown
            cursor.execute("""
                SELECT league, COUNT(*) 
                FROM teams 
                WHERE league IS NOT NULL 
                GROUP BY league
            """)
            summary['teams_by_league'] = dict(cursor.fetchall())
        
        return summary

    # ============= PITCHER-SPECIFIC METHODS =============

    def get_enhanced_pitcher_matchup_data(self, game_id: int) -> Dict[str, Any]:
        """Get comprehensive pitcher matchup data for a game."""
        query = """
        SELECT 
            g.game_id,
            g.season,
            g.date,
            gsp.home_starter_id,
            gsp.away_starter_id,
            -- Home pitcher stats
            hp.name as home_pitcher_name,
            hp.throws as home_pitcher_hand,
            hp.age as home_pitcher_age,
            hps.earned_run_average as home_era,
            hps.whip as home_whip,
            hps.innings_pitched as home_ip,
            hps.strikeouts_pitched as home_k,
            hps.walks_allowed as home_bb,
            hps.hits_allowed as home_h,
            hps.wins as home_w,
            hps.losses as home_l,
            -- Away pitcher stats  
            ap.name as away_pitcher_name,
            ap.throws as away_pitcher_hand,
            ap.age as away_pitcher_age,
            aps.earned_run_average as away_era,
            aps.whip as away_whip,
            aps.innings_pitched as away_ip,
            aps.strikeouts_pitched as away_k,
            aps.walks_allowed as away_bb,
            aps.hits_allowed as away_h,
            aps.wins as away_w,
            aps.losses as away_l
        FROM games g
        JOIN game_starting_pitchers gsp ON g.game_id = gsp.game_id
        LEFT JOIN players hp ON gsp.home_starter_id = hp.player_id
        LEFT JOIN players ap ON gsp.away_starter_id = ap.player_id
        LEFT JOIN player_statistics hps ON hp.player_id = hps.player_id AND g.season = hps.season
        LEFT JOIN player_statistics aps ON ap.player_id = aps.player_id AND g.season = aps.season
        WHERE g.game_id = ?
        """
        
        with sqlite3.connect(self.db_path) as conn:
            result = pd.read_sql_query(query, conn, params=[game_id])
            
            if result.empty:
                # Fallback to placeholder data
                return {
                    'game_id': game_id,
                    'home_starter': {
                        'name': 'Unknown Pitcher',
                        'era': 4.00,
                        'whip': 1.30,
                        'k9': 8.0
                    },
                    'away_starter': {
                        'name': 'Unknown Pitcher',
                        'era': 4.00,
                        'whip': 1.30,
                        'k9': 8.0
                    }
                }
            
            row = result.iloc[0]
            
            # Calculate advanced metrics
            home_k9 = (row['home_k'] / row['home_ip']) * 9 if row['home_ip'] and row['home_ip'] > 0 else 8.0
            away_k9 = (row['away_k'] / row['away_ip']) * 9 if row['away_ip'] and row['away_ip'] > 0 else 8.0
            home_bb9 = (row['home_bb'] / row['home_ip']) * 9 if row['home_ip'] and row['home_ip'] > 0 else 3.0
            away_bb9 = (row['away_bb'] / row['away_ip']) * 9 if row['away_ip'] and row['away_ip'] > 0 else 3.0
            
            return {
                'game_id': game_id,
                'home_starter': {
                    'player_id': row['home_starter_id'],
                    'name': row['home_pitcher_name'] or 'Unknown',
                    'hand': row['home_pitcher_hand'] or 'R',
                    'age': row['home_pitcher_age'] or 28,
                    'era': row['home_era'] or 4.00,
                    'whip': row['home_whip'] or 1.30,
                    'k9': home_k9,
                    'bb9': home_bb9,
                    'innings_pitched': row['home_ip'] or 100.0,
                    'wins': row['home_w'] or 8,
                    'losses': row['home_l'] or 8
                },
                'away_starter': {
                    'player_id': row['away_starter_id'],
                    'name': row['away_pitcher_name'] or 'Unknown',
                    'hand': row['away_pitcher_hand'] or 'R',
                    'age': row['away_pitcher_age'] or 28,
                    'era': row['away_era'] or 4.00,
                    'whip': row['away_whip'] or 1.30,
                    'k9': away_k9,
                    'bb9': away_bb9,
                    'innings_pitched': row['away_ip'] or 100.0,
                    'wins': row['away_w'] or 8,
                    'losses': row['away_l'] or 8
                },
                'matchup_analysis': {
                    'era_advantage': 'home' if (row['home_era'] or 4.0) < (row['away_era'] or 4.0) else 'away',
                    'era_differential': abs((row['home_era'] or 4.0) - (row['away_era'] or 4.0)),
                    'whip_advantage': 'home' if (row['home_whip'] or 1.3) < (row['away_whip'] or 1.3) else 'away',
                    'k9_advantage': 'home' if home_k9 > away_k9 else 'away',
                    'handedness_matchup': f"{row['home_pitcher_hand'] or 'R'} vs {row['away_pitcher_hand'] or 'R'}"
                }
            }

    def get_pitcher_recent_form(self, pitcher_id: int, games: int = 5) -> Dict[str, Any]:
        """Get recent form statistics for a pitcher."""
        # This would require game-by-game pitcher logs
        # For now, return season stats
        query = """
        SELECT * FROM player_statistics ps
        JOIN players p ON ps.player_id = p.player_id
        WHERE ps.player_id = ?
        ORDER BY ps.season DESC
        LIMIT 1
        """
        
        with sqlite3.connect(self.db_path) as conn:
            result = pd.read_sql_query(query, conn, params=[pitcher_id])
        
        if result.empty:
            return {
                'games_started': 0,
                'era': 4.00,
                'whip': 1.30,
                'k9': 8.0,
                'form_trend': 'unknown'
            }
        
        row = result.iloc[0]
        innings = row.get('innings_pitched', 100.0)
        
        return {
            'games_started': int(innings / 6) if innings else 0,  # Rough estimate
            'era': row.get('earned_run_average', 4.00),
            'whip': row.get('whip', 1.30),
            'k9': (row.get('strikeouts_pitched', 0) / innings * 9) if innings > 0 else 8.0,
            'bb9': (row.get('walks_allowed', 0) / innings * 9) if innings > 0 else 3.0,
            'wins': row.get('wins', 0),
            'losses': row.get('losses', 0),
            'form_trend': 'stable'  # Would need game-by-game data for real trend
        }

    def analyze_pitcher_matchup_history(self, pitcher1_id: int, pitcher2_id: int) -> Dict[str, Any]:
        """Analyze historical matchups between two pitchers."""
        # This would require more detailed game logs
        # For now, return comparative stats
        
        pitcher1_stats = self.get_pitcher_recent_form(pitcher1_id)
        pitcher2_stats = self.get_pitcher_recent_form(pitcher2_id)
        
        return {
            'head_to_head_games': 0,  # Would need game logs
            'pitcher1_advantage': pitcher1_stats['era'] < pitcher2_stats['era'],
            'era_comparison': {
                'pitcher1': pitcher1_stats['era'],
                'pitcher2': pitcher2_stats['era'],
                'difference': abs(pitcher1_stats['era'] - pitcher2_stats['era'])
            },
            'strikeout_comparison': {
                'pitcher1': pitcher1_stats['k9'],
                'pitcher2': pitcher2_stats['k9'],
                'difference': abs(pitcher1_stats['k9'] - pitcher2_stats['k9'])
            },
            'recommendation': 'favor_pitcher1' if pitcher1_stats['era'] < pitcher2_stats['era'] else 'favor_pitcher2'
        }

    def estimate_starting_pitchers_for_games(self):
        """Estimate starting pitchers for games missing pitcher data."""
        logger.info("🔍 Estimating starting pitchers for games without pitcher data...")
        
        with sqlite3.connect(self.db_path) as conn:
            # Find games without pitcher assignments
            games_query = """
            SELECT g.game_id, g.home_team_id, g.away_team_id, g.date, g.season
            FROM games g
            LEFT JOIN game_starting_pitchers gsp ON g.game_id = gsp.game_id
            WHERE gsp.game_id IS NULL
            AND g.status = 'Finished'
            ORDER BY g.date
            """
            
            games_without_pitchers = pd.read_sql_query(games_query, conn)
            
            if games_without_pitchers.empty:
                logger.info("✅ All games already have pitcher assignments")
                return
            
            # Get available pitchers for each team/season
            pitchers_query = """
            SELECT ps.player_id, ps.team_id, ps.season, ps.wins, ps.losses, 
                   ps.innings_pitched, ps.earned_run_average,
                   p.name
            FROM player_statistics ps
            JOIN players p ON ps.player_id = p.player_id
            WHERE ps.innings_pitched > 30  -- Filter for actual starters
            ORDER BY ps.team_id, ps.season, ps.innings_pitched DESC
            """
            
            available_pitchers = pd.read_sql_query(pitchers_query, conn)
            
            estimated_assignments = []
            
            for _, game in games_without_pitchers.iterrows():
                # Find pitchers for home team
                home_pitchers = available_pitchers[
                    (available_pitchers['team_id'] == game['home_team_id']) &
                    (available_pitchers['season'] == game['season'])
                ].head(5)  # Top 5 starters
                
                # Find pitchers for away team  
                away_pitchers = available_pitchers[
                    (available_pitchers['team_id'] == game['away_team_id']) &
                    (available_pitchers['season'] == game['season'])
                ].head(5)  # Top 5 starters
                
                if not home_pitchers.empty and not away_pitchers.empty:
                    # Use simple rotation estimate (would be much more sophisticated in reality)
                    home_starter = home_pitchers.iloc[0]['player_id']  # Ace
                    away_starter = away_pitchers.iloc[0]['player_id']  # Ace
                    
                    estimated_assignments.append({
                        'game_id': game['game_id'],
                        'home_starter_id': home_starter,
                        'away_starter_id': away_starter,
                        'home_starter_confirmed': False,
                        'away_starter_confirmed': False,
                        'estimation_method': 'rotation_estimate',
                        'confidence_score': 0.6
                    })
            
            if estimated_assignments:
                assignments_df = pd.DataFrame(estimated_assignments)
                assignments_df.to_sql('game_starting_pitchers', conn, if_exists='append', index=False)
                logger.info(f"✅ Estimated starting pitchers for {len(estimated_assignments)} games")
            else:
                logger.warning("⚠️ Could not estimate starting pitchers - insufficient pitcher data")

    # ============= UTILITY METHODS =============
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        """Remove old data to keep database size manageable."""
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Delete old games and related data
            cursor.execute("DELETE FROM game_odds WHERE game_id IN (SELECT game_id FROM games WHERE date < ?)", (cutoff_date,))
            cursor.execute("DELETE FROM player_props WHERE game_id IN (SELECT game_id FROM games WHERE date < ?)", (cutoff_date,))
            cursor.execute("DELETE FROM game_starting_pitchers WHERE game_id IN (SELECT game_id FROM games WHERE date < ?)", (cutoff_date,))
            cursor.execute("DELETE FROM games WHERE date < ?", (cutoff_date,))
            
            conn.commit()
            
            logger.info(f"✅ Cleaned up MLB data older than {cutoff_date}")
    
    def vacuum_database(self):
        """Optimize database storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("VACUUM")
            logger.info("✅ MLB database optimized")
    
    def get_recent_form(self, team_id: int, games: int = 10) -> Dict[str, Any]:
        """Get recent form for a team."""
        query = """
        SELECT * FROM games 
        WHERE (home_team_id = ? OR away_team_id = ?) 
        AND status = 'Finished'
        ORDER BY date DESC 
        LIMIT ?
        """
        
        with sqlite3.connect(self.db_path) as conn:
            recent_games = pd.read_sql_query(query, conn, params=[team_id, team_id, games])
        
        if recent_games.empty:
            return {'wins': 0, 'losses': 0, 'win_percentage': 0.0, 'runs_for': 0, 'runs_against': 0}
        
        wins = 0
        total_runs_for = 0
        total_runs_against = 0
        
        for _, game in recent_games.iterrows():
            if game['home_team_id'] == team_id:
                # Team was home
                runs_for = game['home_score']
                runs_against = game['away_score']
                if runs_for > runs_against:
                    wins += 1
            else:
                # Team was away
                runs_for = game['away_score']
                runs_against = game['home_score']
                if runs_for > runs_against:
                    wins += 1
            
            total_runs_for += runs_for
            total_runs_against += runs_against
        
        games_played = len(recent_games)
        losses = games_played - wins
        
        return {
            'wins': wins,
            'losses': losses,
            'games_played': games_played,
            'win_percentage': wins / games_played if games_played > 0 else 0.0,
            'runs_per_game': total_runs_for / games_played if games_played > 0 else 0.0,
            'runs_allowed_per_game': total_runs_against / games_played if games_played > 0 else 0.0
        }
    
    def get_head_to_head(self, team1_id: int, team2_id: int, limit: int = 10) -> pd.DataFrame:
        """Get head-to-head matchup history between two teams."""
        query = """
        SELECT * FROM games 
        WHERE ((home_team_id = ? AND away_team_id = ?) 
               OR (home_team_id = ? AND away_team_id = ?))
        AND status = 'Finished'
        ORDER BY date DESC 
        LIMIT ?
        """
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=[team1_id, team2_id, team2_id, team1_id, limit])
    
    def get_pitcher_matchup_data(self, game_id: int) -> Dict[str, Any]:
        """Get pitcher matchup data for a game - now enhanced with real data!"""
        return self.get_enhanced_pitcher_matchup_data(game_id)

    # ============= HELPER METHODS =============
    
    def _add_mlb_team_info(self, teams_df: pd.DataFrame) -> pd.DataFrame:
        """Add MLB-specific team information."""
        teams_data = teams_df.copy()
        
        # MLB division/league mapping
        mlb_divisions = {
            'AL East': ['BOS', 'NYY', 'TB', 'TOR', 'BAL'],
            'AL Central': ['CLE', 'DET', 'KC', 'CWS', 'MIN'],
            'AL West': ['HOU', 'LAA', 'OAK', 'SEA', 'TEX'],
            'NL East': ['ATL', 'MIA', 'NYM', 'PHI', 'WAS'],
            'NL Central': ['CHC', 'CIN', 'MIL', 'PIT', 'STL'],
            'NL West': ['ARI', 'COL', 'LAD', 'SD', 'SF']
        }
        
        # Stadium information
        mlb_stadiums = {
            'BOS': 'Fenway Park',
            'NYY': 'Yankee Stadium',
            'TB': 'Tropicana Field',
            'TOR': 'Rogers Centre',
            'BAL': 'Oriole Park at Camden Yards',
            'CLE': 'Progressive Field',
            'DET': 'Comerica Park',
            'KC': 'Kauffman Stadium',
            'CWS': 'Guaranteed Rate Field',
            'MIN': 'Target Field',
            'HOU': 'Minute Maid Park',
            'LAA': 'Angel Stadium',
            'OAK': 'RingCentral Coliseum',
            'SEA': 'T-Mobile Park',
            'TEX': 'Globe Life Field',
            'ATL': 'Truist Park',
            'MIA': 'loanDepot park',
            'NYM': 'Citi Field',
            'PHI': 'Citizens Bank Park',
            'WAS': 'Nationals Park',
            'CHC': 'Wrigley Field',
            'CIN': 'Great American Ball Park',
            'MIL': 'American Family Field',
            'PIT': 'PNC Park',
            'STL': 'Busch Stadium',
            'ARI': 'Chase Field',
            'COL': 'Coors Field',
            'LAD': 'Dodger Stadium',
            'SD': 'Petco Park',
            'SF': 'Oracle Park'
        }
        
        def get_division_info(row):
            abbr = row.get('abbreviation', '')
            for division, teams in mlb_divisions.items():
                if abbr in teams:
                    league = division.split()[0]  # AL or NL
                    div_name = division.split()[1]    # East, Central, West
                    return pd.Series([league, div_name])
            return pd.Series(['Unknown', 'Unknown'])
        
        def get_stadium(row):
            abbr = row.get('abbreviation', '')
            return mlb_stadiums.get(abbr, 'Unknown Stadium')
        
        if 'abbreviation' in teams_data.columns:
            teams_data[['league', 'division']] = teams_data.apply(get_division_info, axis=1)
            teams_data['stadium'] = teams_data.apply(get_stadium, axis=1)
        else:
            teams_data['league'] = 'Unknown'
            teams_data['division'] = 'Unknown'
            teams_data['stadium'] = 'Unknown Stadium'
        
        return teams_data

    def _add_calculated_pitcher_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated pitcher metrics to games dataframe."""
        result_df = df.copy()
        
        # Calculate K/9 (strikeouts per 9 innings)
        for team in ['home', 'away']:
            ip_col = f'{team}_starter_ip'
            k_col = f'{team}_starter_k'
            k9_col = f'{team}_starter_k9'
            
            if ip_col in result_df.columns and k_col in result_df.columns:
                result_df[k9_col] = np.where(
                    (result_df[ip_col].notna()) & (result_df[ip_col] > 0),
                    (result_df[k_col] / result_df[ip_col]) * 9,
                    8.0  # League average
                )
            
            # Calculate BB/9 (walks per 9 innings)
            bb_col = f'{team}_starter_bb'
            bb9_col = f'{team}_starter_bb9'
            
            if ip_col in result_df.columns and bb_col in result_df.columns:
                result_df[bb9_col] = np.where(
                    (result_df[ip_col].notna()) & (result_df[ip_col] > 0),
                    (result_df[bb_col] / result_df[ip_col]) * 9,
                    3.0  # League average
                )
            
            # Calculate K/BB ratio
            kbb_col = f'{team}_starter_k_bb_ratio'
            if k_col in result_df.columns and bb_col in result_df.columns:
                result_df[kbb_col] = np.where(
                    (result_df[bb_col].notna()) & (result_df[bb_col] > 0),
                    result_df[k_col] / result_df[bb_col],
                    2.5  # League average
                )
        
        # Calculate pitcher vs pitcher metrics
        if all(col in result_df.columns for col in ['home_starter_era', 'away_starter_era']):
            result_df['era_differential'] = result_df['home_starter_era'] - result_df['away_starter_era']
            result_df['era_advantage_home'] = (result_df['home_starter_era'] < result_df['away_starter_era']).astype(int)
        
        if all(col in result_df.columns for col in ['home_starter_whip', 'away_starter_whip']):
            result_df['whip_differential'] = result_df['home_starter_whip'] - result_df['away_starter_whip']
            result_df['whip_advantage_home'] = (result_df['home_starter_whip'] < result_df['away_starter_whip']).astype(int)
        
        if all(col in result_df.columns for col in ['home_starter_k9', 'away_starter_k9']):
            result_df['k9_differential'] = result_df['home_starter_k9'] - result_df['away_starter_k9']
            result_df['k9_advantage_home'] = (result_df['home_starter_k9'] > result_df['away_starter_k9']).astype(int)
        
        return result_df

    def _calculate_pitcher_quality(self, pitcher):
        """Calculate a simple pitcher quality score."""
        era = pitcher.get('earned_run_average', 4.50)
        innings = pitcher.get('innings_pitched', 100)
        
        # Normalize ERA (lower is better, scale 0-1)
        era_score = max(0, (6.0 - era) / 6.0)
        
        # Normalize innings (more innings = more reliable, scale 0-1)  
        innings_score = min(1.0, innings / 200.0)
        
        # Combined score
        return (era_score * 0.7 + innings_score * 0.3)

    # ============= SETUP AND MIGRATION =============

    def setup_pitcher_integration(self):
        """One-time setup for pitcher integration. Run this once."""
        logger.info("🔧⚾ Setting up pitcher integration...")
        
        # Create the pitcher table if it doesn't exist
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS game_starting_pitchers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id INTEGER,
                    home_starter_id INTEGER,
                    away_starter_id INTEGER,
                    home_starter_confirmed BOOLEAN DEFAULT FALSE,
                    away_starter_confirmed BOOLEAN DEFAULT FALSE,
                    estimation_method TEXT DEFAULT 'manual',
                    confidence_score REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (game_id) REFERENCES games (game_id),
                    FOREIGN KEY (home_starter_id) REFERENCES players (player_id),
                    FOREIGN KEY (away_starter_id) REFERENCES players (player_id)
                )
            ''')
            
            # Create indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_game_pitchers_game_id ON game_starting_pitchers (game_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_game_pitchers_home ON game_starting_pitchers (home_starter_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_game_pitchers_away ON game_starting_pitchers (away_starter_id)')
            conn.commit()
        
        # Estimate pitchers for existing games
        self.estimate_starting_pitchers_for_games()
        
        # Get summary
        summary = self.get_data_summary()
        logger.info(f"✅ Pitcher integration setup complete!")
        logger.info(f"   Games with pitcher data: {summary.get('total_pitcher_assignments', 0)}")
        logger.info(f"   Pitcher coverage: {summary.get('pitcher_coverage_pct', 0):.1f}%")
