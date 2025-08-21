# api_clients/unified_api_client.py - UPDATED VERSION
"""
Enhanced unified API client that includes ESPN as a primary data source.
"""

import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from loguru import logger

# Import existing clients
from .sports_api import SportsAPIClient
from .odds_api import OddsAPIClient

# Import new ESPN client
try:
    from .espn_api import ESPNAPIClient
    ESPN_AVAILABLE = True
except ImportError:
    logger.warning("ESPN API client not found. Install ESPN client for enhanced reliability.")
    ESPN_AVAILABLE = False


class DataSourceClient:
    """
    Enhanced unified client that prioritizes ESPN for reliable data.
    Falls back to existing API clients if needed.
    
    This maintains full compatibility with your existing code while adding ESPN reliability.
    """
    
    def __init__(self, sport: str, use_espn_primary: bool = True):
        """
        Initialize with ESPN as primary source.
        
        Args:
            sport: Sport key ('nba', 'mlb', 'nfl')
            use_espn_primary: Whether to use ESPN as primary data source
        """
        self.sport = sport.lower()
        self.use_espn_primary = use_espn_primary and ESPN_AVAILABLE
        
        # Initialize ESPN client as primary if available
        self.espn_client = None
        if self.use_espn_primary:
            try:
                self.espn_client = ESPNAPIClient(sport)
                logger.info(f"✅ ESPN client enabled for {sport.upper()}")
            except Exception as e:
                logger.warning(f"ESPN client failed to initialize: {e}")
                self.use_espn_primary = False
        
        # Initialize existing clients as fallback
        self.sports_client = None
        self.odds_client = None
        
        try:
            self.sports_client = SportsAPIClient(sport)
            logger.info(f"✅ Sports API client available for {sport.upper()}")
        except Exception as e:
            logger.warning(f"Sports API client failed: {e}")
        
        try:
            self.odds_client = OddsAPIClient()
            logger.info(f"✅ Odds API client available")
        except Exception as e:
            logger.warning(f"Odds API client failed: {e}")
        
        # Track which data source was used for debugging
        self.last_data_source = {
            'teams': 'unknown',
            'players': 'unknown',
            'stats': 'unknown'
        }
    
    def get_teams(self, **kwargs) -> pd.DataFrame:
        """Get teams - ESPN first, fallback to sports API."""
        # Try ESPN first if enabled
        if self.use_espn_primary and self.espn_client:
            try:
                teams = self.espn_client.get_teams(**kwargs)
                if not teams.empty:
                    self.last_data_source['teams'] = 'ESPN'
                    logger.debug("✅ Teams from ESPN")
                    return teams
            except Exception as e:
                logger.debug(f"ESPN teams failed: {e}")
        
        # Fallback to sports API
        if self.sports_client:
            try:
                teams = self.sports_client.get_teams(**kwargs)
                if not teams.empty:
                    self.last_data_source['teams'] = 'Sports API'
                    logger.debug("✅ Teams from Sports API (fallback)")
                    return teams
            except Exception as e:
                logger.debug(f"Sports API teams failed: {e}")
        
        self.last_data_source['teams'] = 'none'
        return pd.DataFrame()
    
    def get_players(self, **kwargs) -> pd.DataFrame:
        """Get players - ESPN first, fallback to sports API."""
        # Try ESPN first if enabled
        if self.use_espn_primary and self.espn_client:
            try:
                players = self.espn_client.get_players(**kwargs)
                if not players.empty:
                    self.last_data_source['players'] = 'ESPN'
                    logger.debug("✅ Players from ESPN")
                    return players
            except Exception as e:
                logger.debug(f"ESPN players failed: {e}")
        
        # Fallback to sports API
        if self.sports_client:
            try:
                players = self.sports_client.get_players(**kwargs)
                if not players.empty:
                    self.last_data_source['players'] = 'Sports API'
                    logger.debug("✅ Players from Sports API (fallback)")
                    return players
            except Exception as e:
                logger.debug(f"Sports API players failed: {e}")
        
        self.last_data_source['players'] = 'none'
        return pd.DataFrame()
    
    def get_player_statistics(self, **kwargs) -> pd.DataFrame:
        """Get player stats - try both sources."""
        # Try ESPN first if enabled
        if self.use_espn_primary and self.espn_client:
            try:
                stats = self.espn_client.get_player_statistics(**kwargs)
                if not stats.empty:
                    self.last_data_source['stats'] = 'ESPN'
                    return stats
            except Exception as e:
                logger.debug(f"ESPN player stats failed: {e}")
        
        # Fallback to sports API
        if self.sports_client:
            try:
                stats = self.sports_client.get_player_statistics(**kwargs)
                if not stats.empty:
                    self.last_data_source['stats'] = 'Sports API'
                    return stats
            except Exception as e:
                logger.debug(f"Sports API player stats failed: {e}")
        
        self.last_data_source['stats'] = 'none'
        return pd.DataFrame()
    
    def get_team_statistics(self, **kwargs) -> pd.DataFrame:
        """Get team stats - try both sources."""
        # Try ESPN first if enabled
        if self.use_espn_primary and self.espn_client:
            try:
                stats = self.espn_client.get_team_statistics(**kwargs)
                if not stats.empty:
                    return stats
            except Exception as e:
                logger.debug(f"ESPN team stats failed: {e}")
        
        # Fallback to sports API
        if self.sports_client:
            try:
                return self.sports_client.get_team_statistics(**kwargs)
            except Exception as e:
                logger.debug(f"Sports API team stats failed: {e}")
        
        return pd.DataFrame()
    
    def get_standings(self, **kwargs) -> pd.DataFrame:
        """Get league standings - Sports API only (ESPN doesn't have standings endpoint)."""
        if self.sports_client:
            try:
                return self.sports_client.get_standings(**kwargs)
            except Exception as e:
                logger.debug(f"Standings failed: {e}")
        
        return pd.DataFrame()
    
    def get_games(self, **kwargs) -> pd.DataFrame:
        """Get games data - Sports API only (ESPN doesn't have games endpoint)."""
        if self.sports_client:
            try:
                return self.sports_client.get_games(**kwargs)
            except Exception as e:
                logger.debug(f"Games failed: {e}")
        
        return pd.DataFrame()
    
    def get_odds(self, **kwargs) -> pd.DataFrame:
        """Get odds - only from odds client."""
        if self.odds_client:
            try:
                return self.odds_client.get_odds(sport=self.sport, **kwargs)
            except Exception as e:
                logger.debug(f"Odds API failed: {e}")
        
        return pd.DataFrame()
    
    def get_player_props(self, **kwargs) -> pd.DataFrame:
        """Get player props - only from odds client."""
        if self.odds_client:
            try:
                return self.odds_client.get_player_props(sport=self.sport, **kwargs)
            except Exception as e:
                logger.debug(f"Player props failed: {e}")
        
        return pd.DataFrame()
    
    def get_seasons(self) -> List:
        """Get available seasons."""
        # Try ESPN first
        if self.use_espn_primary and self.espn_client:
            try:
                return self.espn_client.get_seasons()
            except Exception as e:
                logger.debug(f"ESPN seasons failed: {e}")
        
        # Fallback to sports API
        if self.sports_client:
            try:
                return self.sports_client.get_seasons()
            except Exception as e:
                logger.debug(f"Sports API seasons failed: {e}")
        
        return []
    
    def test_connections(self) -> Dict[str, Any]:
        """Test all connections."""
        result = {
            'sport': self.sport,
            'espn_api': {'status': 'not_available'},
            'sports_api': {'status': 'not_available'},
            'odds_api': {'status': 'not_available'}
        }
        
        # Test ESPN if available
        if self.espn_client:
            try:
                result['espn_api'] = self.espn_client.test_connection()
            except Exception as e:
                result['espn_api'] = {'status': 'error', 'error': str(e)}
        
        # Test Sports API
        if self.sports_client:
            try:
                result['sports_api'] = self.sports_client.test_connection()
            except Exception as e:
                result['sports_api'] = {'status': 'error', 'error': str(e)}
        
        # Test Odds API
        if self.odds_client:
            try:
                valid = self.odds_client.validate_api_key()
                result['odds_api'] = {
                    'status': 'success' if valid else 'error',
                    'api_key_valid': valid
                }
            except Exception as e:
                result['odds_api'] = {'status': 'error', 'error': str(e)}
        
        return result
    
    def get_data_source_usage(self) -> Dict[str, str]:
        """Get which data sources were used for the last operations."""
        return self.last_data_source.copy()


# Backwards compatibility - maintain existing class name
class UnifiedAPIClient(DataSourceClient):
    """Alias for backwards compatibility."""
    pass


# Factory function for easy creation
def create_unified_client(sport: str, use_espn_primary: bool = True) -> DataSourceClient:
    """
    Create a unified API client with ESPN integration.
    
    Args:
        sport: Sport key ('nba', 'mlb', 'nfl')
        use_espn_primary: Whether to use ESPN as primary data source
        
    Returns:
        DataSourceClient instance
    """
    return DataSourceClient(sport, use_espn_primary)


# Test function
def test_unified_client(sport: str = 'mlb') -> Dict[str, Any]:
    """Test the unified client functionality."""
    print(f"🧪 Testing Unified Client for {sport.upper()}")
    print("=" * 50)
    
    results = {
        'sport': sport,
        'tests': {},
        'summary': 'unknown'
    }
    
    try:
        # Create client
        print("1️⃣ Creating unified client...")
        client = DataSourceClient(sport, use_espn_primary=True)
        results['tests']['creation'] = 'success'
        
        # Test connections
        print("2️⃣ Testing connections...")
        connections = client.test_connections()
        results['tests']['connections'] = connections
        
        # Test teams
        print("3️⃣ Testing teams retrieval...")
        teams = client.get_teams()
        results['tests']['teams'] = len(teams)
        print(f"   Found: {len(teams)} teams")
        
        # Test players
        print("4️⃣ Testing players retrieval...")
        players = client.get_players()
        results['tests']['players'] = len(players)
        print(f"   Found: {len(players)} players")
        
        # Show data sources used
        print("5️⃣ Data source usage...")
        usage = client.get_data_source_usage()
        print(f"   Teams from: {usage['teams']}")
        print(f"   Players from: {usage['players']}")
        
        # Overall result
        if len(teams) > 0 and len(players) > 0:
            results['summary'] = 'success'
            print("✅ All tests passed!")
        elif len(teams) > 0:
            results['summary'] = 'partial'
            print("⚠️ Teams only")
        else:
            results['summary'] = 'failed'
            print("❌ No data retrieved")
            
    except Exception as e:
        results['tests']['error'] = str(e)
        results['summary'] = 'error'
        print(f"❌ Test failed: {e}")
    
    return results


if __name__ == "__main__":
    # Test all sports
    for sport in ['mlb', 'nba', 'nfl']:
        test_results = test_unified_client(sport)
        print(f"\n{sport.upper()} Test Results: {test_results['summary']}")
        print()
