#!/usr/bin/env python3
"""
Test script for the fixed sports API
This will help find what data is actually available
"""

def test_data_availability():
    """Test what data is actually available for each sport."""
    print("🔍 Testing Data Availability")
    print("=" * 40)
    
    from api_clients.sports_api import SportsAPIClient
    
    sports = ['nba', 'mlb', 'nfl']
    
    for sport in sports:
        print(f"\n{sport.upper()} Data Availability:")
        
        try:
            client = SportsAPIClient(sport)
            
            # Get seasons
            seasons = client.get_seasons()
            print(f"  📅 Available seasons: {len(seasons)}")
            if seasons:
                recent_seasons = seasons[-3:] if len(seasons) >= 3 else seasons
                print(f"      Recent: {recent_seasons}")
            
            # Test each recent season for teams and games
            for season in recent_seasons:
                print(f"\n  Testing season {season}:")
                
                # Test teams
                teams = client.get_teams(season=season)
                print(f"    👥 Teams: {len(teams)}")
                
                # Test games for this season
                games = client.get_games(season=season)
                print(f"    🎮 Games: {len(games)}")
                
                # If we have teams, test stats for first team
                if not teams.empty:
                    first_team = teams.iloc[0]
                    team_id = first_team.get('team_id', first_team.get('id'))
                    team_name = first_team.get('name', 'Unknown')
                    
                    if team_id:
                        stats = client.get_team_statistics(team_id, season)
                        print(f"    📊 Stats for {team_name}: {len(stats)} records")
                        
                        # Test players
                        players = client.get_players(team_id=team_id, season=season)
                        print(f"    ⚾ Players for {team_name}: {len(players)}")
                
                # If any data found, this is a working season
                if not teams.empty or not games.empty:
                    print(f"    ✅ Season {season} has data!")
                    break
            
        except Exception as e:
            print(f"  ❌ Error: {e}")

def find_current_nfl_games():
    """Focus on NFL since it's currently active."""
    print("\n\n🏈 NFL Current Games Deep Dive")
    print("=" * 40)
    
    from api_clients.sports_api import SportsAPIClient
    from datetime import datetime, timedelta
    
    try:
        client = SportsAPIClient('nfl')
        
        # Check seasons available
        seasons = client.get_seasons()
        print(f"📅 NFL Seasons available: {seasons}")
        
        # Try current year and recent years
        current_year = datetime.now().year
        test_years = [current_year, current_year - 1, current_year - 2]
        
        for year in test_years:
            print(f"\n🔍 Testing NFL {year}:")
            
            # Check teams
            teams = client.get_teams(season=year)
            print(f"  👥 Teams: {len(teams)}")
            
            # Check games
            games = client.get_games(season=year)
            print(f"  🎮 Total games: {len(games)}")
            
            # Check recent games (last 30 days)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            recent_games = client.get_games(
                from_date=start_date.strftime('%Y-%m-%d'),
                to_date=end_date.strftime('%Y-%m-%d')
            )
            print(f"  🕒 Recent games (30 days): {len(recent_games)}")
            
            if not recent_games.empty:
                print("  📋 Recent NFL games found:")
                for _, game in recent_games.head(3).iterrows():
                    home = game.get('home_team_name', 'Home')
                    away = game.get('away_team_name', 'Away')
                    date = game.get('date', 'Unknown')
                    status = game.get('status', 'Unknown')
                    print(f"      • {date}: {away} @ {home} ({status})")
            
            # If we found data, this is our working season
            if not teams.empty and not games.empty:
                print(f"  ✅ NFL {year} is active!")
                
                # Try to get team stats
                if not teams.empty:
                    first_team = teams.iloc[0]
                    team_id = first_team.get('team_id', first_team.get('id'))
                    team_name = first_team.get('name', 'Unknown')
                    
                    stats = client.get_team_statistics(team_id, year)
                    print(f"  📊 Stats for {team_name}: {len(stats)} records")
                
                break
    
    except Exception as e:
        print(f"❌ NFL deep dive error: {e}")

def test_alternative_endpoints():
    """Test alternative ways to get data."""
    print("\n\n🔄 Testing Alternative Endpoints")
    print("=" * 40)
    
    from api_clients.sports_api import SportsAPIClient
    
    # Focus on NFL since we know it has current games
    try:
        client = SportsAPIClient('nfl')
        
        print("🏈 NFL Alternative Approaches:")
        
        # 1. Try live games
        live_games = client.get_games(live=True)
        print(f"  🔴 Live games: {len(live_games)}")
        
        # 2. Try today's games
        today = datetime.now().strftime('%Y-%m-%d')
        today_games = client.get_games(date=today)
        print(f"  📅 Today's games: {len(today_games)}")
        
        # 3. Try this week's games
        dates_this_week = []
        for i in range(7):
            date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            dates_this_week.append(date)
        
        week_games = []
        for date in dates_this_week:
            daily_games = client.get_games(date=date)
            if not daily_games.empty:
                week_games.extend(daily_games.to_dict('records'))
                print(f"  📅 Games on {date}: {len(daily_games)}")
        
        print(f"  📊 Total games this week: {len(week_games)}")
        
        # 4. Try getting standings (we know this works)
        standings = client.get_standings()
        print(f"  🏆 Standings: {len(standings)} teams")
        
        if not standings.empty:
            # Get a few teams and try individual team data
            sample_teams = standings.head(3)
            
            for _, team in sample_teams.iterrows():
                team_id = team.get('team_id')
                team_name = team.get('team_name', 'Unknown')
                
                if team_id:
                    # Try team games
                    team_games = client.get_games(team_id=team_id)
                    print(f"  🎮 {team_name} games: {len(team_games)}")
                    
                    # Try H2H if we have another team
                    if len(sample_teams) > 1:
                        other_team = sample_teams.iloc[1]
                        other_team_id = other_team.get('team_id')
                        
                        if other_team_id and team_id != other_team_id:
                            h2h = client.get_head_to_head(team_id, other_team_id)
                            print(f"  ⚔️ H2H {team_name} vs {other_team.get('team_name')}: {len(h2h)}")
    
    except Exception as e:
        print(f"❌ Alternative endpoints error: {e}")

def create_working_data_summary():
    """Create a summary of what actually works."""
    print("\n\n📋 Working Data Summary")
    print("=" * 40)
    
    from api_clients.sports_api import SportsAPIClient
    
    working_data = {
        'nba': {'teams': False, 'games': False, 'stats': False, 'standings': False},
        'mlb': {'teams': False, 'games': False, 'stats': False, 'standings': False},
        'nfl': {'teams': False, 'games': False, 'stats': False, 'standings': False}
    }
    
    for sport in ['nba', 'mlb', 'nfl']:
        try:
            client = SportsAPIClient(sport)
            
            # Test teams
            teams = client.get_teams()
            working_data[sport]['teams'] = not teams.empty
            
            # Test standings
            standings = client.get_standings()
            working_data[sport]['standings'] = not standings.empty
            
            # Test recent games
            today = datetime.now().strftime('%Y-%m-%d')
            games = client.get_games(date=today)
            working_data[sport]['games'] = not games.empty
            
            # Test stats if we have teams
            if not teams.empty:
                first_team_id = teams.iloc[0].get('team_id', teams.iloc[0].get('id'))
                if first_team_id:
                    stats = client.get_team_statistics(first_team_id, 2024)
                    working_data[sport]['stats'] = not stats.empty
        
        except Exception as e:
            print(f"❌ Error testing {sport}: {e}")
    
    # Print summary
    print("\n📊 What's Working Right Now:")
    for sport, data in working_data.items():
        print(f"\n{sport.upper()}:")
        for endpoint, working in data.items():
            status = "✅" if working else "❌"
            print(f"  {endpoint}: {status}")
    
    return working_data

def main():
    """Run all tests to find working data."""
    print("🧪 Fixed API Testing Suite")
    print("🎯 Finding what data is actually available")
    print("=" * 50)
    
    test_data_availability()
    find_current_nfl_games()
    test_alternative_endpoints()
    working_summary = create_working_data_summary()
    
    print("\n" + "=" * 50)
    print("🎯 ACTIONABLE RESULTS")
    print("=" * 50)
    
    # Count what's working
    total_working = sum(sum(sport_data.values()) for sport_data in working_summary.values())
    total_endpoints = len(working_summary) * 4  # 3 sports × 4 endpoints
    
    print(f"📊 Working endpoints: {total_working}/{total_endpoints}")
    
    if total_working > 0:
        print("\n✅ SUCCESS! You have working data endpoints.")
        print("\n🚀 IMMEDIATE NEXT STEPS:")
        
        # Suggest specific actions based on what's working
        for sport, data in working_summary.items():
            if any(data.values()):
                working_endpoints = [k for k, v in data.items() if v]
                print(f"   {sport.upper()}: Use {', '.join(working_endpoints)}")
        
        print("\n💡 RECOMMENDED ACTIONS:")
        print("1. 🏈 Focus on NFL (preseason is active)")
        print("2. 🏆 Use standings data (most reliable)")
        print("3. 👥 Use team data for basic info")
        print("4. 🔄 Set up daily data collection")
        
    else:
        print("⚠️ Limited data available - may need different approaches")
        print("💡 Consider using historical data or different seasons")

if __name__ == "__main__":
    from datetime import datetime
    main()
