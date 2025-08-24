import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import json
from contextlib import contextmanager

from config.settings import Settings

class NFLDatabase:
    """Manages NFL data storage and retrieval."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize NFL database connection."""
        self.db_path = db_path or Settings.NFL_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database tables
        self._create_tables()
        logger.info(f"🏈 NFL Database initialized: {self.db_path}")
    
    def get_connection(self):
        """
        Get database connection for external use.
        Returns a SQLite connection object.
        
        Usage:
            conn = db.get_connection()
            # ... use connection ...
            conn.close()
        """
        return sqlite3.connect(self.db_path)
    
    @contextmanager
    def get_connection_context(self):
        """
        Get database connection as context manager.
        Automatically handles closing the connection.
        
        Usage:
            with db.get_connection_context() as conn:
                # ... use connection ...
        """
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def _create_tables(self):
        """Create all necessary tables for NFL data."""
        with sqlite3.connect(self.db_path) as conn:
            # Games table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS games (
                    game_id INTEGER PRIMARY KEY,
                    date TEXT NOT NULL,
                    time TEXT,
                    week INTEGER,
                    season INTEGER,
                    season_type TEXT,  -- 'preseason', 'regular', 'postseason'
                    home_team_id INTEGER,
                    away_team_id INTEGER,
                    home_team_name TEXT,
                    away_team_name TEXT,
                    home_score INTEGER,
                    away_score INTEGER,
                    status TEXT,
                    venue TEXT,
                    city TEXT,
                    temperature REAL,
                    weather_conditions TEXT,
                    wind_speed REAL,
                    dome BOOLEAN DEFAULT FALSE,
                    playoff_game BOOLEAN DEFAULT FALSE,
                    divisional_game BOOLEAN DEFAULT FALSE,
                    conference_game BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Teams table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS teams (
                    team_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    city TEXT,
                    abbreviation TEXT,
                    conference TEXT,  -- 'AFC', 'NFC'
                    division TEXT,    -- 'North', 'South', 'East', 'West'
                    logo TEXT,
                    founded INTEGER,
                    stadium TEXT,
                    dome BOOLEAN DEFAULT FALSE,
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
                    college TEXT,
                    experience INTEGER,
                    team_id INTEGER,
                    active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (team_id) REFERENCES teams (team_id)
                )
            ''')
            
            # Team statistics table (offensive and defensive stats)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS team_statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_id INTEGER,
                    season INTEGER,
                    week INTEGER,
                    games_played INTEGER,
                    wins INTEGER,
                    losses INTEGER,
                    ties INTEGER,
                    win_percentage REAL,
                    points_per_game REAL,
                    points_allowed_per_game REAL,
                    yards_per_game REAL,
                    yards_allowed_per_game REAL,
                    passing_yards_per_game REAL,
                    passing_yards_allowed_per_game REAL,
                    rushing_yards_per_game REAL,
                    rushing_yards_allowed_per_game REAL,
                    touchdowns_per_game REAL,
                    touchdowns_allowed_per_game REAL,
                    turnovers_per_game REAL,
                    turnovers_forced_per_game REAL,
                    turnover_differential REAL,
                    sacks_per_game REAL,
                    sacks_allowed_per_game REAL,
                    third_down_conversion_pct REAL,
                    third_down_defense_pct REAL,
                    red_zone_efficiency REAL,
                    red_zone_defense REAL,
                    time_of_possession REAL,
                    penalties_per_game REAL,
                    penalty_yards_per_game REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (team_id) REFERENCES teams (team_id)
                )
            ''')
            
            # Player statistics table (position-specific stats)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS player_statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id INTEGER,
                    team_id INTEGER,
                    season INTEGER,
                    week INTEGER,
                    games_played INTEGER,
                    -- Passing stats
                    passing_attempts INTEGER,
                    passing_completions INTEGER,
                    passing_yards INTEGER,
                    passing_touchdowns INTEGER,
                    interceptions INTEGER,
                    passing_rating REAL,
                    -- Rushing stats
                    rushing_attempts INTEGER,
                    rushing_yards INTEGER,
                    rushing_touchdowns INTEGER,
                    rushing_average REAL,
                    -- Receiving stats
                    receptions INTEGER,
                    receiving_yards INTEGER,
                    receiving_touchdowns INTEGER,
                    targets INTEGER,
                    -- Defensive stats
                    tackles INTEGER,
                    assists INTEGER,
                    sacks REAL,
                    interceptions_def INTEGER,
                    fumbles_forced INTEGER,
                    fumbles_recovered INTEGER,
                    passes_defended INTEGER,
                    -- Kicking stats
                    field_goals_made INTEGER,
                    field_goals_attempted INTEGER,
                    field_goal_percentage REAL,
                    extra_points_made INTEGER,
                    extra_points_attempted INTEGER,
                    -- Special teams
                    punt_returns INTEGER,
                    punt_return_yards INTEGER,
                    kick_returns INTEGER,
                    kick_return_yards INTEGER,
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
                    prop_type TEXT,  -- 'passing_yards', 'rushing_yards', 'receiving_yards', 'touchdowns', etc.
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
            
            # Injuries table (more important in NFL)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS injuries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id INTEGER,
                    team_id INTEGER,
                    injury_type TEXT,
                    body_part TEXT,
                    status TEXT,  -- 'Out', 'Doubtful', 'Questionable', 'Probable'
                    week INTEGER,
                    season INTEGER,
                    date_reported TEXT,
                    date_resolved TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (player_id) REFERENCES players (player_id),
                    FOREIGN KEY (team_id) REFERENCES teams (team_id)
                )
            ''')
            
            # Create indexes for better performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_games_date ON games (date)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_games_week_season ON games (week, season)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_games_teams ON games (home_team_id, away_team_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_team_stats_season ON team_statistics (team_id, season)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_player_stats_season ON player_statistics (player_id, season)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_odds_game ON game_odds (game_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_injuries_player_week ON injuries (player_id, week, season)')
            
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
                'week': 'week',
                'season': 'season',
                'season_type': 'season_type',
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
            
            # Add NFL-specific defaults
            if 'season_type' not in games_data.columns:
                games_data['season_type'] = 'regular'
            
            # Handle missing values
            games_data = games_data.fillna({
                'time': '',
                'home_score': 0,
                'away_score': 0,
                'status': 'Scheduled',
                'venue': '',
                'city': '',
                'week': 1
            })
            
            # Determine divisional and conference games
            games_data = self._add_game_context(games_data)
            
            games_data.to_sql('games', conn, if_exists='append', index=False, method='multi')
            
            logger.info(f"✅ Saved {len(games_data)} NFL games to database")
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
                'city': 'city',
                'abbreviation': 'abbreviation',
                'logo': 'logo',
                'founded': 'founded'
            }
            
            save_columns = [col for col in column_mapping.keys() if col in teams_data.columns]
            teams_data = teams_data[save_columns].rename(columns=column_mapping)
            
            # Add NFL-specific team data
            teams_data = self._add_nfl_team_info(teams_data)
            
            teams_data.to_sql('teams', conn, if_exists='replace', index=False)
            
            logger.info(f"✅ Saved {len(teams_data)} NFL teams to database")
            return len(teams_data)
    
    def save_team_statistics(self, stats_df: pd.DataFrame) -> int:
        """Save team statistics to database."""
        if stats_df.empty:
            return 0
        
        with sqlite3.connect(self.db_path) as conn:
            stats_data = stats_df.copy()
            
            # NFL-specific column mapping
            nfl_columns = {
                'team_id': 'team_id',
                'season': 'season',
                'week': 'week',
                'games_played': 'games_played',
                'wins': 'wins',
                'losses': 'losses',
                'ties': 'ties',
                'win_percentage': 'win_percentage',
                'points_for': 'points_per_game',
                'points_against': 'points_allowed_per_game',
                'total_yards': 'yards_per_game',
                'total_yards_allowed': 'yards_allowed_per_game',
                'passing_yards': 'passing_yards_per_game',
                'passing_yards_allowed': 'passing_yards_allowed_per_game',
                'rushing_yards': 'rushing_yards_per_game',
                'rushing_yards_allowed': 'rushing_yards_allowed_per_game',
                'touchdowns': 'touchdowns_per_game',
                'touchdowns_allowed': 'touchdowns_allowed_per_game',
                'turnovers': 'turnovers_per_game',
                'turnovers_forced': 'turnovers_forced_per_game',
                'sacks': 'sacks_per_game',
                'sacks_allowed': 'sacks_allowed_per_game'
            }
            
            # Map available columns
            available_cols = [col for col in nfl_columns.keys() if col in stats_data.columns]
            stats_data = stats_data[available_cols].rename(columns=nfl_columns)
            
            # Calculate derived metrics
            if 'turnovers_per_game' in stats_data.columns and 'turnovers_forced_per_game' in stats_data.columns:
                stats_data['turnover_differential'] = (
                    stats_data['turnovers_forced_per_game'] - stats_data['turnovers_per_game']
                )
            
            stats_data.to_sql('team_statistics', conn, if_exists='append', index=False)
            
            logger.info(f"✅ Saved {len(stats_data)} NFL team statistics records")
            return len(stats_data)
    
    def save_player_statistics(self, stats_df: pd.DataFrame) -> int:
        """Save player statistics to database."""
        if stats_df.empty:
            return 0
        
        with sqlite3.connect(self.db_path) as conn:
            stats_data = stats_df.copy()
            
            # NFL player stats mapping (comprehensive for all positions)
            player_columns = {
                'player_id': 'player_id',
                'team_id': 'team_id',
                'season': 'season',
                'week': 'week',
                'games_played': 'games_played',
                # Passing
                'pass_att': 'passing_attempts',
                'pass_comp': 'passing_completions',
                'pass_yds': 'passing_yards',
                'pass_td': 'passing_touchdowns',
                'interceptions': 'interceptions',
                'passer_rating': 'passing_rating',
                # Rushing
                'rush_att': 'rushing_attempts',
                'rush_yds': 'rushing_yards',
                'rush_td': 'rushing_touchdowns',
                # Receiving
                'receptions': 'receptions',
                'rec_yds': 'receiving_yards',
                'rec_td': 'receiving_touchdowns',
                'targets': 'targets',
                # Defense
                'tackles': 'tackles',
                'sacks': 'sacks',
                'int_def': 'interceptions_def',
                'fumbles_forced': 'fumbles_forced'
            }
            
            available_cols = [col for col in player_columns.keys() if col in stats_data.columns]
            stats_data = stats_data[available_cols].rename(columns=player_columns)
            
            # Calculate derived stats
            if 'rushing_yards' in stats_data.columns and 'rushing_attempts' in stats_data.columns:
                stats_data['rushing_average'] = (
                    stats_data['rushing_yards'] / stats_data['rushing_attempts'].replace(0, 1)
                )
            
            stats_data.to_sql('player_statistics', conn, if_exists='append', index=False)
            
            logger.info(f"✅ Saved {len(stats_data)} NFL player statistics records")
            return len(stats_data)
    
    def save_injuries(self, injuries_df: pd.DataFrame) -> int:
        """Save injury data to database."""
        if injuries_df.empty:
            return 0
        
        with sqlite3.connect(self.db_path) as conn:
            injuries_data = injuries_df.copy()
            
            # Map injury data columns
            injury_columns = {
                'player_id': 'player_id',
                'team_id': 'team_id',
                'injury_type': 'injury_type',
                'body_part': 'body_part',
                'status': 'status',
                'week': 'week',
                'season': 'season',
                'date_reported': 'date_reported'
            }
            
            available_cols = [col for col in injury_columns.keys() if col in injuries_data.columns]
            injuries_data = injuries_data[available_cols].rename(columns=injury_columns)
            
            injuries_data.to_sql('injuries', conn, if_exists='append', index=False)
            
            logger.info(f"✅ Saved {len(injuries_data)} NFL injury records")
            return len(injuries_data)
    
    def get_games(self, 
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None,
                  season: Optional[int] = None,
                  week: Optional[int] = None,
                  team_id: Optional[int] = None,
                  season_type: str = 'regular') -> pd.DataFrame:
        """Retrieve games from database."""
        
        query = """
        SELECT g.*, 
               ht.name as home_team_full_name,
               ht.abbreviation as home_team_abbr,
               ht.conference as home_conference,
               ht.division as home_division,
               at.name as away_team_full_name,
               at.abbreviation as away_team_abbr,
               at.conference as away_conference,
               at.division as away_division
        FROM games g
        LEFT JOIN teams ht ON g.home_team_id = ht.team_id
        LEFT JOIN teams at ON g.away_team_id = at.team_id
        WHERE g.season_type = ?
        """
        
        params = [season_type]
        
        if start_date:
            query += " AND g.date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND g.date <= ?"
            params.append(end_date)
        if season:
            query += " AND g.season = ?"
            params.append(season)
        if week:
            query += " AND g.week = ?"
            params.append(week)
        if team_id:
            query += " AND (g.home_team_id = ? OR g.away_team_id = ?)"
            params.extend([team_id, team_id])
        
        query += " ORDER BY g.week, g.date, g.time"
        
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
            ht.abbreviation as home_abbr,
            ht.conference as home_conference,
            ht.division as home_division,
            at.name as away_team_name,
            at.abbreviation as away_abbr,
            at.conference as away_conference,
            at.division as away_division,
            hts.points_per_game as home_ppg,
            hts.points_allowed_per_game as home_papg,
            hts.yards_per_game as home_ypg,
            hts.yards_allowed_per_game as home_yapg,
            hts.passing_yards_per_game as home_pass_ypg,
            hts.rushing_yards_per_game as home_rush_ypg,
            hts.turnovers_per_game as home_turnovers,
            hts.turnover_differential as home_turnover_diff,
            hts.sacks_per_game as home_sacks,
            ats.points_per_game as away_ppg,
            ats.points_allowed_per_game as away_papg,
            ats.yards_per_game as away_ypg,
            ats.yards_allowed_per_game as away_yapg,
            ats.passing_yards_per_game as away_pass_ypg,
            ats.rushing_yards_per_game as away_rush_ypg,
            ats.turnovers_per_game as away_turnovers,
            ats.turnover_differential as away_turnover_diff,
            ats.sacks_per_game as away_sacks,
            CASE WHEN g.home_score > g.away_score THEN 1 ELSE 0 END as home_win
        FROM games g
        LEFT JOIN teams ht ON g.home_team_id = ht.team_id
        LEFT JOIN teams at ON g.away_team_id = at.team_id
        LEFT JOIN team_statistics hts ON g.home_team_id = hts.team_id AND g.season = hts.season
        LEFT JOIN team_statistics ats ON g.away_team_id = ats.team_id AND g.season = ats.season
        WHERE g.status = 'Finished' AND g.season_type = 'regular'
        """
        
        params = []
        if seasons:
            placeholders = ','.join(['?' for _ in seasons])
            base_query += f" AND g.season IN ({placeholders})"
            params.extend(seasons)
        
        base_query += " ORDER BY g.season, g.week, g.date"
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(base_query, conn, params=params)
        
        logger.info(f"✅ Retrieved {len(df)} historical NFL games")
        return df
    
    def get_team_statistics(self, 
                           season: int,
                           team_id: Optional[int] = None,
                           week: Optional[int] = None) -> pd.DataFrame:
        """Get team statistics for a season/week."""
        
        query = """
        SELECT ts.*, t.name as team_name, t.abbreviation as team_abbr,
               t.conference, t.division
        FROM team_statistics ts
        JOIN teams t ON ts.team_id = t.team_id
        WHERE ts.season = ?
        """
        
        params = [season]
        
        if team_id:
            query += " AND ts.team_id = ?"
            params.append(team_id)
        if week:
            query += " AND ts.week = ?"
            params.append(week)
        
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
        if position:
            query += " AND p.position = ?"
            params.append(position)
        
        query += " ORDER BY ps.passing_yards DESC, ps.rushing_yards DESC, ps.receiving_yards DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def get_injuries_by_week(self, season: int, week: int) -> pd.DataFrame:
        """Get injury report for a specific week."""
        
        query = """
        SELECT i.*, p.name as player_name, p.position, t.name as team_name, t.abbreviation
        FROM injuries i
        JOIN players p ON i.player_id = p.player_id
        JOIN teams t ON i.team_id = t.team_id
        WHERE i.season = ? AND i.week = ?
        ORDER BY t.name, i.status, p.position
        """
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=[season, week])
    
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
            
            # Current week games
            cursor.execute("""
                SELECT COUNT(*) FROM games 
                WHERE date >= date('now') AND date <= date('now', '+7 days')
            """)
            summary['upcoming_games_7d'] = cursor.fetchone()[0]
            
            # Statistics coverage
            cursor.execute("SELECT COUNT(DISTINCT season) FROM team_statistics")
            summary['team_stats_seasons'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT season) FROM player_statistics")
            summary['player_stats_seasons'] = cursor.fetchone()[0]
            
            # Injury data
            cursor.execute("SELECT COUNT(*) FROM injuries")
            summary['total_injury_records'] = cursor.fetchone()[0]
            
            # Season type breakdown
            cursor.execute("""
                SELECT season_type, COUNT(*) 
                FROM games 
                GROUP BY season_type
            """)
            summary['games_by_type'] = dict(cursor.fetchall())
        
        return summary
    
    def _add_game_context(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Add contextual information about games (divisional, conference, etc.)."""
        games_data = games_df.copy()
        
        # Get team division/conference info
        with sqlite3.connect(self.db_path) as conn:
            teams_info = pd.read_sql_query(
                "SELECT team_id, conference, division FROM teams",
                conn
            )
        
        if not teams_info.empty:
            # Merge team info
            games_data = games_data.merge(
                teams_info.rename(columns={'team_id': 'home_team_id', 'conference': 'home_conf', 'division': 'home_div'}),
                on='home_team_id', how='left'
            )
            games_data = games_data.merge(
                teams_info.rename(columns={'team_id': 'away_team_id', 'conference': 'away_conf', 'division': 'away_div'}),
                on='away_team_id', how='left'
            )
            
            # Determine game types
            games_data['divisional_game'] = (
                games_data['home_div'] == games_data['away_div']
            ).astype(int)
            
            games_data['conference_game'] = (
                games_data['home_conf'] == games_data['away_conf']
            ).astype(int)
            
            # Clean up temporary columns
            games_data = games_data.drop(['home_conf', 'home_div', 'away_conf', 'away_div'], axis=1)
        else:
            games_data['divisional_game'] = 0
            games_data['conference_game'] = 0
        
        return games_data
    
    def _add_nfl_team_info(self, teams_df: pd.DataFrame) -> pd.DataFrame:
        """Add NFL-specific team information."""
        teams_data = teams_df.copy()
        
        # NFL division/conference mapping
        nfl_divisions = {
            'AFC East': ['BUF', 'MIA', 'NE', 'NYJ'],
            'AFC North': ['BAL', 'CIN', 'CLE', 'PIT'],
            'AFC South': ['HOU', 'IND', 'JAX', 'TEN'],
            'AFC West': ['DEN', 'KC', 'LV', 'LAC'],
            'NFC East': ['DAL', 'NYG', 'PHI', 'WAS'],
            'NFC North': ['CHI', 'DET', 'GB', 'MIN'],
            'NFC South': ['ATL', 'CAR', 'NO', 'TB'],
            'NFC West': ['ARI', 'LAR', 'SF', 'SEA']
        }
        
        # Stadium info (dome teams)
        dome_teams = ['ATL', 'DAL', 'DET', 'HOU', 'IND', 'LV', 'LAR', 'MIN', 'NO', 'ARI']
        
        def get_division_info(row):
            abbr = row.get('abbreviation', '')
            for division, teams in nfl_divisions.items():
                if abbr in teams:
                    conference = division.split()[0]  # AFC or NFC
                    div_name = division.split()[1]    # East, North, etc.
                    return pd.Series([conference, div_name])
            return pd.Series(['Unknown', 'Unknown'])
        
        if 'abbreviation' in teams_data.columns:
            teams_data[['conference', 'division']] = teams_data.apply(get_division_info, axis=1)
            teams_data['dome'] = teams_data['abbreviation'].isin(dome_teams).astype(int)
        else:
            teams_data['conference'] = 'Unknown'
            teams_data['division'] = 'Unknown'
            teams_data['dome'] = 0
        
        return teams_data
    
    def cleanup_old_data(self, days_to_keep: int = 1095):  # 3 years for NFL
        """Remove old data to keep database size manageable."""
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Delete old games and related data
            cursor.execute("DELETE FROM game_odds WHERE game_id IN (SELECT game_id FROM games WHERE date < ?)", (cutoff_date,))
            cursor.execute("DELETE FROM player_props WHERE game_id IN (SELECT game_id FROM games WHERE date < ?)", (cutoff_date,))
            cursor.execute("DELETE FROM games WHERE date < ?", (cutoff_date,))
            
            # Clean old injury data
            cursor.execute("DELETE FROM injuries WHERE date_reported < ?", (cutoff_date,))
            
            conn.commit()
            
            logger.info(f"✅ Cleaned up NFL data older than {cutoff_date}")
    
    def vacuum_database(self):
        """Optimize database storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("VACUUM")
            logger.info("✅ NFL database optimized")
