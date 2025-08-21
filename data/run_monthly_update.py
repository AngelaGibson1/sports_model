# run_monthly_update.py
"""
Monthly Update Orchestrator for Sports Data Ingestion
Runs MLB, NBA, and NFL data ingestion scripts and generates comprehensive reports.
Designed for automated monthly execution to keep ML training data current.
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
from loguru import logger
import time
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.base import MimeBase
from email import encoders

# Add parent directory to path for api_clients access
sys.path.append(str(Path(__file__).parent.parent))

# Import the individual ingestion classes (they're in the same directory now)
from mlb_data_ingest import MLBDataIngest
from nba_data_ingest import NBADataIngest  
from nfl_data_ingest import NFLDataIngest


class MonthlyUpdateOrchestrator:
    """
    Orchestrates monthly data updates for all sports.
    Handles execution, monitoring, reporting, and notifications.
    """
    
    def __init__(self, base_output_dir: str = "data", enable_notifications: bool = False):
        """
        Initialize the monthly update orchestrator.
        
        Args:
            base_output_dir: Base directory for logs and data files
            enable_notifications: Whether to send email notifications
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create latest directory for ML training files
        self.latest_dir = self.base_output_dir / "latest"
        self.latest_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_notifications = enable_notifications
        
        # Setup master logging
        self.setup_logging()
        
        # Sports configuration
        self.sports_config = {
            'mlb': {
                'name': 'Major League Baseball',
                'class': MLBDataIngest,
                'emoji': '⚾',
                'priority': 1
            },
            'nba': {
                'name': 'National Basketball Association',
                'class': NBADataIngest,
                'emoji': '🏀',
                'priority': 2
            },
            'nfl': {
                'name': 'National Football League',
                'class': NFLDataIngest,
                'emoji': '🏈',
                'priority': 3
            }
        }
        
        # Email configuration (set these in environment variables)
        self.email_config = {
            'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(os.getenv('SMTP_PORT', '587')),
            'from_email': os.getenv('FROM_EMAIL', ''),
            'from_password': os.getenv('EMAIL_PASSWORD', ''),
            'to_emails': os.getenv('TO_EMAILS', '').split(',') if os.getenv('TO_EMAILS') else []
        }
        
        logger.info("🔄 Monthly Update Orchestrator initialized")
    
    def setup_logging(self):
        """Setup comprehensive logging for the orchestrator."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Master log file in latest directory
        master_log = self.latest_dir / f"monthly_update_{timestamp}.log"
        logger.add(master_log, rotation="50 MB", retention="90 days", level="INFO")
        
        # Error log file in latest directory
        error_log = self.latest_dir / f"monthly_update_errors_{timestamp}.log"
        logger.add(error_log, rotation="10 MB", retention="90 days", level="ERROR")
        
        logger.info(f"Logging setup complete - Master: {master_log}")
    
    def run_sport_ingestion(self, sport: str) -> Dict[str, Any]:
        """
        Run data ingestion for a specific sport.
        
        Args:
            sport: Sport key ('mlb', 'nba', 'nfl')
            
        Returns:
            Results dictionary with execution details
        """
        if sport not in self.sports_config:
            raise ValueError(f"Unknown sport: {sport}")
        
        config = self.sports_config[sport]
        
        logger.info(f"{config['emoji']} Starting {config['name']} data ingestion")
        start_time = datetime.now()
        
        try:
            # Initialize sport-specific ingestion class
            ingest_class = config['class'](output_dir=str(self.base_output_dir))
            
            # Run the ingestion
            results = ingest_class.run_full_ingestion()
            
            # Add orchestrator metadata
            results['sport'] = sport
            results['sport_name'] = config['name']
            results['orchestrator_start'] = start_time.isoformat()
            results['orchestrator_end'] = datetime.now().isoformat()
            
            logger.info(f"✅ {config['name']} ingestion completed: {results['status']}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ {config['name']} ingestion failed: {e}")
            
            return {
                'sport': sport,
                'sport_name': config['name'],
                'status': 'error',
                'orchestrator_start': start_time.isoformat(),
                'orchestrator_end': datetime.now().isoformat(),
                'errors': [str(e)],
                'teams_count': 0,
                'players_count': 0,
                'files_saved': {},
                'data_quality': {}
            }
    
    def run_all_sports(self) -> Dict[str, Any]:
        """
        Run data ingestion for all sports in priority order.
        
        Returns:
            Comprehensive results for all sports
        """
        logger.info("🚀 Starting monthly update for all sports")
        overall_start = datetime.now()
        
        overall_results = {
            'update_date': overall_start.isoformat(),
            'sports_results': {},
            'summary': {
                'total_teams': 0,
                'total_players': 0,
                'successful_sports': 0,
                'failed_sports': 0,
                'partial_sports': 0
            },
            'duration_minutes': 0,
            'files_created': [],
            'data_quality_summary': {},
            'notifications_sent': False
        }
        
        # Sort sports by priority
        sports_list = sorted(self.sports_config.keys(), 
                           key=lambda x: self.sports_config[x]['priority'])
        
        for sport in sports_list:
            try:
                # Add delay between sports to be respectful to APIs
                if sport != sports_list[0]:  # Don't delay before first sport
                    logger.info("⏳ Waiting 30 seconds between sports...")
                    time.sleep(30)
                
                results = self.run_sport_ingestion(sport)
                overall_results['sports_results'][sport] = results
                
                # Update summary statistics
                overall_results['summary']['total_teams'] += results.get('teams_count', 0)
                overall_results['summary']['total_players'] += results.get('players_count', 0)
                
                if results['status'] == 'success':
                    overall_results['summary']['successful_sports'] += 1
                elif results['status'] == 'no_data':
                    overall_results['summary']['partial_sports'] += 1
                else:
                    overall_results['summary']['failed_sports'] += 1
                
                # Collect file paths
                files_saved = results.get('files_saved', {})
                for file_type, file_path in files_saved.items():
                    overall_results['files_created'].append(f"{sport}_{file_type}: {file_path}")
                
                # Collect data quality info
                data_quality = results.get('data_quality', {})
                if data_quality:
                    overall_results['data_quality_summary'][sport] = data_quality
                
            except Exception as e:
                logger.error(f"❌ Failed to process {sport}: {e}")
                overall_results['sports_results'][sport] = {
                    'sport': sport,
                    'status': 'error',
                    'errors': [str(e)]
                }
                overall_results['summary']['failed_sports'] += 1
        
        # Calculate total duration
        overall_end = datetime.now()
        overall_results['duration_minutes'] = (overall_end - overall_start).total_seconds() / 60
        overall_results['end_date'] = overall_end.isoformat()
        
        # Save overall results
        self.save_overall_results(overall_results)
        
        # Send notifications if enabled
        if self.enable_notifications:
            try:
                self.send_update_notification(overall_results)
                overall_results['notifications_sent'] = True
            except Exception as e:
                logger.error(f"❌ Failed to send notifications: {e}")
        
        logger.info(f"🏁 Monthly update completed in {overall_results['duration_minutes']:.1f} minutes")
        
        return overall_results
    
    def save_overall_results(self, results: Dict[str, Any]):
        """Save comprehensive results to latest directory."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed JSON results to latest directory
        results_file = self.latest_dir / f"monthly_update_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save human-readable summary to latest directory
        summary_file = self.latest_dir / f"monthly_update_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(self.generate_text_report(results))
        
        # Save latest results to latest directory (overwrite each time)
        latest_file = self.latest_dir / "latest_monthly_update.json"
        with open(latest_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📊 Results saved to {results_file}")
    
    def generate_text_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable text report."""
        report = []
        report.append("=" * 60)
        report.append("🏆 SPORTS DATA MONTHLY UPDATE REPORT")
        report.append("=" * 60)
        report.append(f"Update Date: {results['update_date']}")
        report.append(f"Duration: {results['duration_minutes']:.1f} minutes")
        report.append("")
        
        # Overall summary
        summary = results['summary']
        report.append("📊 OVERALL SUMMARY")
        report.append("-" * 20)
        report.append(f"Total Teams: {summary['total_teams']:,}")
        report.append(f"Total Players: {summary['total_players']:,}")
        report.append(f"Successful Sports: {summary['successful_sports']}")
        report.append(f"Failed Sports: {summary['failed_sports']}")
        report.append(f"Partial Sports: {summary['partial_sports']}")
        report.append("")
        
        # Sport-by-sport breakdown
        report.append("🏟️ SPORT-BY-SPORT RESULTS")
        report.append("-" * 30)
        
        for sport, sport_results in results['sports_results'].items():
            config = self.sports_config.get(sport, {})
            emoji = config.get('emoji', '🏆')
            name = config.get('name', sport.upper())
            
            report.append(f"\n{emoji} {name}")
            report.append(f"   Status: {sport_results['status'].upper()}")
            report.append(f"   Teams: {sport_results.get('teams_count', 0)}")
            report.append(f"   Players: {sport_results.get('players_count', 0)}")
            
            if sport_results.get('seasons_processed'):
                seasons = sport_results['seasons_processed']
                report.append(f"   Seasons: {min(seasons)} - {max(seasons)}")
            
            if sport_results.get('errors'):
                report.append(f"   Errors: {len(sport_results['errors'])}")
                for error in sport_results['errors']:
                    report.append(f"     - {error}")
        
        # Data quality summary
        if results['data_quality_summary']:
            report.append("\n🔍 DATA QUALITY SUMMARY")
            report.append("-" * 25)
            
            for sport, quality_data in results['data_quality_summary'].items():
                report.append(f"\n{sport.upper()}:")
                
                if 'players' in quality_data:
                    player_quality = quality_data['players']
                    score = player_quality.get('data_quality_score', 0)
                    report.append(f"   Player Data Quality: {score}/100")
                    
                    if player_quality.get('issues'):
                        report.append(f"   Player Issues:")
                        for issue in player_quality['issues']:
                            report.append(f"     - {issue}")
                
                if 'teams' in quality_data:
                    team_quality = quality_data['teams']
                    score = team_quality.get('data_quality_score', 0)
                    report.append(f"   Team Data Quality: {score}/100")
                    
                    if team_quality.get('issues'):
                        report.append(f"   Team Issues:")
                        for issue in team_quality['issues']:
                            report.append(f"     - {issue}")
        
        # Files created
        if results['files_created']:
            report.append("\n📁 FILES CREATED")
            report.append("-" * 15)
            for file_info in results['files_created']:
                report.append(f"   {file_info}")
        
        report.append("\n" + "=" * 60)
        report.append("Report generated on: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def send_update_notification(self, results: Dict[str, Any]):
        """Send email notification with update results."""
        if not self.email_config['from_email'] or not self.email_config['to_emails']:
            logger.warning("Email configuration incomplete - skipping notifications")
            return
        
        logger.info("📧 Sending update notification email")
        
        # Create email
        msg = MimeMultipart()
        msg['From'] = self.email_config['from_email']
        msg['To'] = ', '.join(self.email_config['to_emails'])
        
        # Determine subject based on results
        summary = results['summary']
        if summary['failed_sports'] == 0:
            status = "✅ SUCCESS"
        elif summary['successful_sports'] > 0:
            status = "⚠️ PARTIAL"
        else:
            status = "❌ FAILED"
        
        msg['Subject'] = f"Sports Data Update {status} - {summary['total_players']:,} Players Updated"
        
        # Create email body
        body = self.generate_text_report(results)
        msg.attach(MimeText(body, 'plain'))
        
        # Send email
        try:
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['from_email'], self.email_config['from_password'])
            
            text = msg.as_string()
            server.sendmail(self.email_config['from_email'], self.email_config['to_emails'], text)
            server.quit()
            
            logger.info("✅ Notification email sent successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to send email: {e}")
            raise
    
    def cleanup_old_files(self, days_to_keep: int = 90):
        """Clean up old log and result files in latest directory."""
        logger.info(f"🧹 Cleaning up files older than {days_to_keep} days")
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        files_deleted = 0
        
        # Cleanup patterns for all timestamped files in latest directory
        # Keep the "latest" files (no timestamp in name) but clean old timestamped ones
        patterns = ['*_202*.log', '*_202*.json', '*_202*.txt', '*_202*.parquet']
        
        for pattern in patterns:
            for file_path in self.latest_dir.glob(pattern):
                try:
                    if file_path.stat().st_mtime < cutoff_date.timestamp():
                        file_path.unlink()
                        files_deleted += 1
                except Exception as e:
                    logger.warning(f"Could not delete {file_path}: {e}")
        
        logger.info(f"🗑️ Deleted {files_deleted} old files")
    
    def validate_environment(self) -> Dict[str, bool]:
        """Validate that the environment is ready for data ingestion."""
        logger.info("🔍 Validating environment")
        
        validation = {
            'directories_created': False,
            'api_clients_available': False,
            'dependencies_available': False,
            'disk_space_sufficient': False
        }
        
        try:
            # Check directories
            self.base_output_dir.mkdir(parents=True, exist_ok=True)
            self.latest_dir.mkdir(parents=True, exist_ok=True)
            validation['directories_created'] = True
            
            # Check API clients
            try:
                from api_clients.espn_api import ESPNAPIClient
                validation['api_clients_available'] = True
            except ImportError:
                logger.error("ESPN API client not found")
            
            # Check dependencies
            try:
                import pandas
                import requests
                import loguru
                validation['dependencies_available'] = True
            except ImportError as e:
                logger.error(f"Missing dependency: {e}")
            
            # Check disk space (require at least 1GB free)
            try:
                import shutil
                free_bytes = shutil.disk_usage(self.base_output_dir).free
                free_gb = free_bytes / (1024**3)
                validation['disk_space_sufficient'] = free_gb >= 1.0
                
                if not validation['disk_space_sufficient']:
                    logger.warning(f"Low disk space: {free_gb:.1f}GB available")
                
            except Exception as e:
                logger.warning(f"Could not check disk space: {e}")
                validation['disk_space_sufficient'] = True  # Assume OK if can't check
            
        except Exception as e:
            logger.error(f"Environment validation failed: {e}")
        
        # Log validation results
        for check, passed in validation.items():
            status = "✅" if passed else "❌"
            logger.info(f"{status} {check}: {passed}")
        
        return validation


def main():
    """Main execution function for monthly updates."""
    print("🔄 SPORTS DATA MONTHLY UPDATE")
    print("=" * 50)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run monthly sports data update')
    parser.add_argument('--sports', nargs='+', choices=['mlb', 'nba', 'nfl'], 
                       help='Specific sports to update (default: all)')
    parser.add_argument('--output-dir', default='data', 
                       help='Output directory for data files')
    parser.add_argument('--enable-email', action='store_true',
                       help='Enable email notifications')
    parser.add_argument('--cleanup-days', type=int, default=90,
                       help='Days of files to keep (default: 90)')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate environment, don\'t run ingestion')
    
    args = parser.parse_args()
    
    try:
        # Initialize orchestrator
        orchestrator = MonthlyUpdateOrchestrator(
            base_output_dir=args.output_dir,
            enable_notifications=args.enable_email
        )
        
        # Validate environment
        validation = orchestrator.validate_environment()
        
        if not all(validation.values()):
            print("❌ Environment validation failed")
            for check, passed in validation.items():
                if not passed:
                    print(f"   Failed: {check}")
            
            if args.validate_only:
                return 1
            else:
                print("⚠️ Continuing despite validation issues...")
        
        if args.validate_only:
            print("✅ Environment validation passed")
            return 0
        
        # Clean up old files
        orchestrator.cleanup_old_files(args.cleanup_days)
        
        # Run updates
        if args.sports:
            # Run specific sports only
            print(f"Running updates for: {', '.join(args.sports)}")
            results = {'sports_results': {}}
            
            for sport in args.sports:
                results['sports_results'][sport] = orchestrator.run_sport_ingestion(sport)
        else:
            # Run all sports
            results = orchestrator.run_all_sports()
        
        # Print final summary
        print("\n" + "=" * 50)
        print("📊 FINAL SUMMARY")
        print("=" * 50)
        
        if 'summary' in results:
            summary = results['summary']
            print(f"Total Teams: {summary['total_teams']:,}")
            print(f"Total Players: {summary['total_players']:,}")
            print(f"Successful Sports: {summary['successful_sports']}")
            print(f"Failed Sports: {summary['failed_sports']}")
            print(f"Duration: {results.get('duration_minutes', 0):.1f} minutes")
        
        # Determine exit code
        if 'summary' in results and results['summary']['failed_sports'] == 0:
            print("✅ All updates completed successfully")
            return 0
        else:
            print("⚠️ Some updates had issues")
            return 1
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logger.error(f"Fatal error in main: {e}")
        return 1


def setup_cron_job():
    """Helper function to set up a monthly cron job."""
    print("🕒 CRON JOB SETUP INSTRUCTIONS")
    print("=" * 40)
    print("To run this script monthly, add the following to your crontab:")
    print("(Run 'crontab -e' to edit)")
    print()
    
    script_path = Path(__file__).absolute()
    python_path = sys.executable
    data_dir = script_path.parent  # Since script is in data folder
    
    # First day of month at 2 AM - run from data directory
    cron_line = f"0 2 1 * * cd {data_dir} && {python_path} run_monthly_update.py --enable-email"
    
    print(f"# Monthly sports data update (1st of month at 2 AM)")
    print(cron_line)
    print()
    print("Environment variables to set:")
    print("export SMTP_SERVER='smtp.gmail.com'")
    print("export SMTP_PORT='587'")
    print("export FROM_EMAIL='your-email@gmail.com'")
    print("export EMAIL_PASSWORD='your-app-password'")
    print("export TO_EMAILS='recipient1@email.com,recipient2@email.com'")
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--setup-cron':
        setup_cron_job()
    else:
        exit_code = main()
        sys.exit(exit_code)
