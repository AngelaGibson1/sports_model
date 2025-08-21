# /utils/betting_calculator.py
# Kelly Criterion & Portfolio Management for MLB Betting

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from loguru import logger
import math

class KellyCriterionCalculator:
    """
    Advanced Kelly Criterion calculator for sports betting with risk management.
    Handles probability estimation, edge calculation, and optimal bet sizing.
    """
    
    def __init__(self, 
                 max_kelly_fraction: float = 0.25,
                 min_edge_threshold: float = 0.02,
                 max_bet_percentage: float = 0.05):
        """
        Initialize Kelly calculator with risk parameters.
        
        Args:
            max_kelly_fraction: Maximum Kelly fraction (risk management)
            min_edge_threshold: Minimum edge required to bet (2% minimum)
            max_bet_percentage: Maximum percentage of bankroll per bet
        """
        self.max_kelly_fraction = max_kelly_fraction
        self.min_edge_threshold = min_edge_threshold
        self.max_bet_percentage = max_bet_percentage
        
        logger.info(f"💰 Kelly Calculator initialized (max: {max_kelly_fraction:.1%}, min edge: {min_edge_threshold:.1%})")
    
    def calculate_kelly_bet(self, 
                           predicted_probability: float,
                           bookmaker_odds: float,
                           confidence_level: float = 1.0) -> Dict[str, float]:
        """
        Calculate optimal Kelly bet size with risk adjustments.
        
        Args:
            predicted_probability: Model's win probability (0-1)
            bookmaker_odds: American odds (-110, +150, etc.)
            confidence_level: Confidence in prediction (0-1)
        
        Returns:
            Dictionary with bet sizing information
        """
        # Convert American odds to decimal probability
        if bookmaker_odds > 0:
            decimal_odds = (bookmaker_odds / 100) + 1
            implied_prob = 100 / (bookmaker_odds + 100)
        else:
            decimal_odds = (100 / abs(bookmaker_odds)) + 1
            implied_prob = abs(bookmaker_odds) / (abs(bookmaker_odds) + 100)
        
        # Calculate edge
        edge = predicted_probability - implied_prob
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds-1, p = win probability, q = lose probability
        b = decimal_odds - 1
        p = predicted_probability
        q = 1 - predicted_probability
        
        kelly_fraction = (b * p - q) / b if b > 0 else 0
        
        # Apply confidence adjustment
        kelly_fraction *= confidence_level
        
        # Risk management: cap Kelly fraction
        kelly_fraction = min(kelly_fraction, self.max_kelly_fraction)
        kelly_fraction = max(kelly_fraction, 0)  # Never negative
        
        # Additional risk management
        final_bet_size = min(kelly_fraction, self.max_bet_percentage)
        
        # Only bet if edge exceeds minimum threshold
        if edge < self.min_edge_threshold:
            final_bet_size = 0
            recommendation = "NO BET - Insufficient edge"
        elif final_bet_size > 0:
            recommendation = "BET"
        else:
            recommendation = "NO BET - Risk limits"
        
        return {
            'kelly_fraction': kelly_fraction,
            'final_bet_size': final_bet_size,
            'edge': edge,
            'edge_percentage': edge * 100,
            'implied_probability': implied_prob,
            'predicted_probability': predicted_probability,
            'decimal_odds': decimal_odds,
            'expected_value': (predicted_probability * b) - q,
            'recommendation': recommendation,
            'confidence_adjusted': confidence_level < 1.0
        }
    
    def calculate_portfolio_bets(self, 
                                games_data: List[Dict],
                                bankroll: float,
                                max_total_exposure: float = 0.20) -> pd.DataFrame:
        """
        Calculate optimal bet sizes for a portfolio of games.
        
        Args:
            games_data: List of game dictionaries with predictions and odds
            bankroll: Total available bankroll
            max_total_exposure: Maximum total percentage of bankroll to risk
        
        Returns:
            DataFrame with recommended bet sizes
        """
        portfolio_results = []
        
        for game in games_data:
            kelly_result = self.calculate_kelly_bet(
                predicted_probability=game['predicted_probability'],
                bookmaker_odds=game['odds'],
                confidence_level=game.get('confidence', 1.0)
            )
            
            # Calculate dollar amounts
            recommended_bet = kelly_result['final_bet_size'] * bankroll
            
            portfolio_results.append({
                'game_id': game.get('game_id', 'Unknown'),
                'matchup': game.get('matchup', 'Unknown'),
                'predicted_probability': kelly_result['predicted_probability'],
                'bookmaker_odds': game['odds'],
                'implied_probability': kelly_result['implied_probability'],
                'edge': kelly_result['edge'],
                'edge_percentage': kelly_result['edge_percentage'],
                'kelly_fraction': kelly_result['kelly_fraction'],
                'bet_percentage': kelly_result['final_bet_size'],
                'bet_amount': recommended_bet,
                'expected_value': kelly_result['expected_value'],
                'recommendation': kelly_result['recommendation']
            })
        
        df = pd.DataFrame(portfolio_results)
        
        # Portfolio risk management: limit total exposure
        if not df.empty and len(df[df['bet_amount'] > 0]) > 0:
            total_exposure = df['bet_percentage'].sum()
            
            if total_exposure > max_total_exposure:
                # Scale down all bets proportionally
                scale_factor = max_total_exposure / total_exposure
                df['bet_percentage'] *= scale_factor
                df['bet_amount'] *= scale_factor
                
                logger.warning(f"⚠️ Portfolio exposure reduced by {(1-scale_factor)*100:.1f}% for risk management")
        
        return df
    
    def backtest_kelly_performance(self, 
                                  historical_bets: pd.DataFrame,
                                  initial_bankroll: float = 10000) -> Dict[str, float]:
        """
        Backtest Kelly strategy performance on historical data.
        
        Args:
            historical_bets: DataFrame with columns: bet_size, odds, outcome (1/0)
            initial_bankroll: Starting bankroll amount
        
        Returns:
            Performance metrics dictionary
        """
        if historical_bets.empty:
            return {}
        
        bankroll = initial_bankroll
        bankroll_history = [bankroll]
        
        for _, bet in historical_bets.iterrows():
            bet_amount = bet['bet_amount']
            outcome = bet['outcome']  # 1 for win, 0 for loss
            
            if outcome == 1:
                # Win: add profit
                if bet['odds'] > 0:
                    profit = bet_amount * (bet['odds'] / 100)
                else:
                    profit = bet_amount * (100 / abs(bet['odds']))
                bankroll += profit
            else:
                # Loss: lose bet amount
                bankroll -= bet_amount
            
            bankroll_history.append(bankroll)
        
        # Calculate performance metrics
        total_return = (bankroll - initial_bankroll) / initial_bankroll
        max_bankroll = max(bankroll_history)
        max_drawdown = (max_bankroll - min(bankroll_history)) / max_bankroll
        
        win_rate = historical_bets['outcome'].mean()
        avg_bet_size = historical_bets['bet_amount'].mean() / initial_bankroll
        
        # Sharpe-like ratio (return per unit of max drawdown)
        sharpe_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
        
        return {
            'total_return': total_return,
            'final_bankroll': bankroll,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_bets': len(historical_bets),
            'avg_bet_size_pct': avg_bet_size,
            'sharpe_ratio': sharpe_ratio,
            'profitable': bankroll > initial_bankroll
        }

class MLBBettingAnalyzer:
    """MLB-specific betting analysis and recommendations."""
    
    def __init__(self):
        self.kelly_calc = KellyCriterionCalculator()
        
    def analyze_mlb_game(self, 
                        game_prediction: Dict,
                        odds_data: Dict) -> Dict[str, any]:
        """
        Comprehensive MLB game betting analysis.
        
        Args:
            game_prediction: Model prediction with probabilities
            odds_data: Bookmaker odds for various markets
        
        Returns:
            Complete betting analysis
        """
        analysis = {
            'game_info': {
                'matchup': f"{game_prediction.get('away_team')} @ {game_prediction.get('home_team')}",
                'date': game_prediction.get('date'),
                'model_confidence': game_prediction.get('confidence', 0.6)
            },
            'moneyline_analysis': {},
            'total_runs_analysis': {},
            'first_5_innings_analysis': {},
            'recommendations': []
        }
        
        # Moneyline analysis
        if 'home_win_probability' in game_prediction and 'moneyline' in odds_data:
            home_ml = self.kelly_calc.calculate_kelly_bet(
                predicted_probability=game_prediction['home_win_probability'],
                bookmaker_odds=odds_data['moneyline']['home'],
                confidence_level=game_prediction.get('confidence', 1.0)
            )
            
            away_ml = self.kelly_calc.calculate_kelly_bet(
                predicted_probability=1 - game_prediction['home_win_probability'],
                bookmaker_odds=odds_data['moneyline']['away'],
                confidence_level=game_prediction.get('confidence', 1.0)
            )
            
            analysis['moneyline_analysis'] = {
                'home_bet': home_ml,
                'away_bet': away_ml,
                'best_bet': 'home' if home_ml['final_bet_size'] > away_ml['final_bet_size'] else 'away'
            }
        
        # Total runs analysis
        if 'total_runs_prediction' in game_prediction and 'totals' in odds_data:
            over_prob = game_prediction.get('over_probability', 0.5)
            
            over_bet = self.kelly_calc.calculate_kelly_bet(
                predicted_probability=over_prob,
                bookmaker_odds=odds_data['totals']['over'],
                confidence_level=game_prediction.get('confidence', 1.0)
            )
            
            under_bet = self.kelly_calc.calculate_kelly_bet(
                predicted_probability=1 - over_prob,
                bookmaker_odds=odds_data['totals']['under'],
                confidence_level=game_prediction.get('confidence', 1.0)
            )
            
            analysis['total_runs_analysis'] = {
                'predicted_total': game_prediction.get('total_runs_prediction'),
                'bookmaker_line': odds_data['totals'].get('line'),
                'over_bet': over_bet,
                'under_bet': under_bet
            }
        
        # Generate recommendations
        recommendations = []
        
        for market, market_analysis in analysis.items():
            if isinstance(market_analysis, dict) and 'bet' in str(market_analysis):
                for bet_type, bet_data in market_analysis.items():
                    if isinstance(bet_data, dict) and bet_data.get('recommendation') == 'BET':
                        recommendations.append({
                            'market': market.replace('_analysis', ''),
                            'bet_type': bet_type.replace('_bet', ''),
                            'edge': bet_data['edge_percentage'],
                            'bet_size': bet_data['final_bet_size'],
                            'confidence': 'High' if bet_data['edge_percentage'] > 5 else 'Medium'
                        })
        
        analysis['recommendations'] = recommendations
        analysis['total_recommendations'] = len(recommendations)
        
        return analysis

# Usage example and testing
if __name__ == "__main__":
    # Test Kelly calculator
    kelly = KellyCriterionCalculator()
    
    # Example: Model predicts 60% chance, bookmaker offers +150
    test_bet = kelly.calculate_kelly_bet(
        predicted_probability=0.60,
        bookmaker_odds=150,
        confidence_level=0.8
    )
    
    print("🧪 Kelly Calculator Test:")
    print(f"Edge: {test_bet['edge_percentage']:.1f}%")
    print(f"Kelly fraction: {test_bet['kelly_fraction']:.1%}")
    print(f"Recommended bet: {test_bet['final_bet_size']:.1%}")
    print(f"Recommendation: {test_bet['recommendation']}")
