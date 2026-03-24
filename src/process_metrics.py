"""
Process-based evaluation metrics for TCG-Bench.
Implements intermediate step validation and reasoning metrics.
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from game import GameState, Player, Card, resolve_card_effect
import json


class ProcessMetricsCollector:
    """Collects process-based metrics during game execution."""
    
    def __init__(self):
        """Initialize the metrics collector."""
        self.reset()
    
    def reset(self):
        """Reset all metrics for a new game."""
        self.metrics = {
            "move_validity": [],
            "decision_times": [],
            "token_usage": [],
            "resource_efficiency": [],
            "strategic_consistency": [],
            "card_diversity": defaultdict(int),
            "turn_advantages": [],
            "critical_decisions": [],
            "error_types": defaultdict(int),
            "adaptation_scores": []
        }
        self.game_start_time = time.time()
        self.previous_states = []
        self.move_history = []
    
    def record_move_attempt(self, move: str, is_valid: bool, game_state: GameState):
        """
        Record a move attempt and its validity.

        Args:
            move: The attempted move
            is_valid: Whether the move was valid
            game_state: Current game state
        """
        self.metrics["move_validity"].append({
            "turn": game_state.turn_number,
            "move": move,
            "valid": is_valid,
            "timestamp": time.time() - self.game_start_time
        })

        if not is_valid:
            self.metrics["error_types"]["invalid_move"] += 1

    def record_parsing_method(self, parsing_result: Dict):
        """
        Record parsing method information.

        Args:
            parsing_result: Dictionary containing parsing information
        """
        if "parsing_methods" not in self.metrics:
            self.metrics["parsing_methods"] = []

        self.metrics["parsing_methods"].append({
            "timestamp": time.time() - self.game_start_time,
            "method": parsing_result.get("method", "unknown"),
            "success": parsing_result.get("success", False),
            "details": parsing_result
        })
    
    def record_decision_time(self, decision_time: float, turn: int):
        """Record time taken to make a decision."""
        self.metrics["decision_times"].append({
            "turn": turn,
            "time": decision_time,
            "category": self._categorize_decision_time(decision_time)
        })
    
    def _categorize_decision_time(self, time_seconds: float) -> str:
        """Categorize decision time."""
        if time_seconds < 1.0:
            return "instant"
        elif time_seconds < 3.0:
            return "fast"
        elif time_seconds < 10.0:
            return "normal"
        elif time_seconds < 30.0:
            return "slow"
        else:
            return "very_slow"
    
    def record_token_usage(self, input_tokens: int, output_tokens: int, turn: int):
        """Record token usage for a turn."""
        self.metrics["token_usage"].append({
            "turn": turn,
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens
        })
    
    def calculate_resource_efficiency(self, game_state: GameState, player: Player) -> float:
        """
        Calculate resource efficiency score.
        
        Args:
            game_state: Current game state
            player: Player to evaluate
            
        Returns:
            Efficiency score (0-1)
        """
        # Cards played effectively vs cards available
        cards_played = len(self.move_history)
        cards_available = len(player.hand) + cards_played
        
        # Damage dealt per card
        damage_efficiency = 0.0
        if cards_played > 0:
            opponent = game_state.p2 if player == game_state.p1 else game_state.p1
            damage_dealt = 10 - opponent.lp  # Assuming starting LP is 10
            damage_efficiency = damage_dealt / cards_played
        
        # Mana/resource efficiency (simplified)
        efficiency = min(1.0, damage_efficiency / 3.0)  # Normalize to 0-1
        
        self.metrics["resource_efficiency"].append({
            "turn": game_state.turn_number,
            "score": efficiency,
            "cards_played": cards_played,
            "damage_efficiency": damage_efficiency
        })
        
        return efficiency
    
    def evaluate_strategic_consistency(self, current_move: str, 
                                      game_state: GameState) -> float:
        """
        Evaluate consistency of strategy across turns.
        
        Args:
            current_move: Current move being made
            game_state: Current game state
            
        Returns:
            Consistency score (0-1)
        """
        if len(self.move_history) < 2:
            return 1.0  # Not enough history to evaluate
        
        # Check if following a consistent strategy pattern
        # (e.g., aggressive = mostly champions, control = mostly spells/tricks)
        move_types = [self._categorize_move(m) for m in self.move_history[-3:]]
        current_type = self._categorize_move(current_move)
        
        # Calculate consistency
        if move_types:
            most_common = max(set(move_types), key=move_types.count)
            consistency = (move_types.count(most_common) + 
                         (1 if current_type == most_common else 0)) / (len(move_types) + 1)
        else:
            consistency = 1.0
        
        self.metrics["strategic_consistency"].append({
            "turn": game_state.turn_number,
            "score": consistency,
            "strategy_type": current_type
        })
        
        return consistency
    
    def _categorize_move(self, move: str) -> str:
        """Categorize a move by strategy type."""
        move_lower = move.lower()
        if "champion" in move_lower or "warrior" in move_lower:
            return "aggressive"
        elif "spell" in move_lower or "fireball" in move_lower:
            return "control"
        elif "trick" in move_lower or "counter" in move_lower:
            return "defensive"
        else:
            return "balanced"
    
    def track_card_diversity(self, card_name: str, turn: int):
        """Track diversity of cards played."""
        self.metrics["card_diversity"][card_name] += 1
    
    def calculate_turn_advantage(self, game_state: GameState) -> float:
        """
        Calculate current advantage score.
        
        Args:
            game_state: Current game state
            
        Returns:
            Advantage score (-1 to 1, positive = P1 advantage)
        """
        # Life point differential
        lp_diff = game_state.players[0].lp - game_state.players[1].lp
        
        # Board presence differential
        board_diff = len(game_state.players[0].board) - len(game_state.players[1].board)
        
        # Hand size differential
        hand_diff = len(game_state.players[0].hand) - len(game_state.players[1].hand)
        
        # Weighted advantage score
        advantage = (lp_diff * 0.5 + board_diff * 0.3 + hand_diff * 0.2) / 10.0
        advantage = max(-1.0, min(1.0, advantage))  # Clamp to [-1, 1]
        
        self.metrics["turn_advantages"].append({
            "turn": game_state.turn_number,
            "advantage": advantage,
            "lp_diff": lp_diff,
            "board_diff": board_diff,
            "hand_diff": hand_diff
        })
        
        return advantage
    
    def identify_critical_decision(self, game_state: GameState, 
                                  move: str, outcome_change: float):
        """
        Identify and record critical decisions.
        
        Args:
            game_state: Game state when decision was made
            move: The move made
            outcome_change: Change in win probability
        """
        if abs(outcome_change) > 0.2:  # Threshold for critical
            self.metrics["critical_decisions"].append({
                "turn": game_state.turn_number,
                "move": move,
                "impact": outcome_change,
                "category": "game_changing" if abs(outcome_change) > 0.4 else "significant"
            })
    
    def calculate_adaptation_score(self, current_performance: float,
                                  historical_performance: List[float]) -> float:
        """
        Calculate adaptation/learning score over multiple games.

        Args:
            current_performance: Current game performance
            historical_performance: Previous game performances

        Returns:
            Adaptation score (positive = improving)
        """
        if not historical_performance:
            return 0.0

        # Calculate trend
        recent_avg = np.mean(historical_performance[-3:]) if len(historical_performance) >= 3 else np.mean(historical_performance)
        adaptation = current_performance - recent_avg

        self.metrics["adaptation_scores"].append(adaptation)
        return adaptation

    def get_parsing_stats(self) -> Dict:
        """
        Get parsing method statistics.

        Returns:
            Dictionary of parsing statistics
        """
        if "parsing_methods" not in self.metrics or not self.metrics["parsing_methods"]:
            return {
                "total_parses": 0,
                "successful_parses": 0,
                "success_rate": 0.0,
                "methods_used": {}
            }

        parsing_methods = self.metrics["parsing_methods"]
        total = len(parsing_methods)
        successful = sum(1 for p in parsing_methods if p.get("success", False))

        # Count methods used
        method_counts = defaultdict(int)
        for parse in parsing_methods:
            method = parse.get("method", "unknown")
            method_counts[method] += 1

        return {
            "total_parses": total,
            "successful_parses": successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "methods_used": dict(method_counts),
            "parsing_details": parsing_methods
        }
    
    def get_summary_metrics(self) -> Dict:
        """
        Get summary of all collected metrics.
        
        Returns:
            Dictionary of summarized metrics
        """
        # Move validity rate
        valid_moves = sum(1 for m in self.metrics["move_validity"] if m["valid"])
        total_moves = len(self.metrics["move_validity"])
        validity_rate = valid_moves / total_moves if total_moves > 0 else 0.0
        
        # Average decision time
        avg_decision_time = np.mean([d["time"] for d in self.metrics["decision_times"]]) if self.metrics["decision_times"] else 0.0
        
        # Average token usage
        avg_tokens = np.mean([t["total"] for t in self.metrics["token_usage"]]) if self.metrics["token_usage"] else 0.0
        
        # Resource efficiency
        avg_efficiency = np.mean([e["score"] for e in self.metrics["resource_efficiency"]]) if self.metrics["resource_efficiency"] else 0.0
        
        # Strategic consistency
        avg_consistency = np.mean([c["score"] for c in self.metrics["strategic_consistency"]]) if self.metrics["strategic_consistency"] else 0.0
        
        # Card diversity (entropy)
        if self.metrics["card_diversity"]:
            total_cards = sum(self.metrics["card_diversity"].values())
            probabilities = [count/total_cards for count in self.metrics["card_diversity"].values()]
            entropy = -sum(p * np.log(p) for p in probabilities if p > 0)
            diversity_score = entropy / np.log(len(self.metrics["card_diversity"])) if len(self.metrics["card_diversity"]) > 1 else 0.0
        else:
            diversity_score = 0.0
        
        # Critical decisions
        num_critical = len(self.metrics["critical_decisions"])
        
        # Final advantage
        final_advantage = self.metrics["turn_advantages"][-1]["advantage"] if self.metrics["turn_advantages"] else 0.0
        
        return {
            "move_validity_rate": validity_rate,
            "avg_decision_time": avg_decision_time,
            "avg_token_usage": avg_tokens,
            "resource_efficiency": avg_efficiency,
            "strategic_consistency": avg_consistency,
            "card_diversity": diversity_score,
            "num_critical_decisions": num_critical,
            "final_advantage": final_advantage,
            "error_counts": dict(self.metrics["error_types"]),
            "total_turns": len(self.metrics["turn_advantages"])
        }
    
    def save_detailed_metrics(self, filepath: str):
        """Save detailed metrics to file."""
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
    
    def load_metrics(self, filepath: str):
        """Load metrics from file."""
        with open(filepath, 'r') as f:
            self.metrics = json.load(f)


class ProcessBasedEvaluator:
    """Evaluates models using process-based metrics."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.collectors = {}
        self.game_results = []
    
    def evaluate_game_with_metrics(self, game_state: GameState, 
                                  llm_agent, mcts_agent) -> Dict:
        """
        Run a game while collecting process metrics.
        
        Args:
            game_state: Initial game state
            llm_agent: LLM agent
            mcts_agent: Opponent agent
            
        Returns:
            Game result with process metrics
        """
        collector = ProcessMetricsCollector()
        
        # Game loop
        while not game_state.is_game_over():
            current_player = game_state.current_player()
            
            # Time decision
            start_time = time.time()
            
            if current_player == game_state.p1:  # LLM player
                move = llm_agent.get_move(game_state)
                decision_time = time.time() - start_time
                
                # Record metrics
                collector.record_decision_time(decision_time, game_state.turn_number)
                collector.record_move_attempt(move, True, game_state)  # Assume valid
                
                # Calculate process metrics
                efficiency = collector.calculate_resource_efficiency(game_state, current_player)
                consistency = collector.evaluate_strategic_consistency(move, game_state)
                advantage = collector.calculate_turn_advantage(game_state)
                
            else:  # MCTS player
                move = mcts_agent.get_move(game_state)
            
            # Execute move
            resolve_card_effect(game_state, current_player, move)
            collector.move_history.append(move)
            
            game_state.next_turn()
        
        # Game over - collect final metrics
        winner = "llm" if game_state.p1.lp > 0 else "mcts"
        
        result = {
            "winner": winner,
            "final_scores": {
                "llm": game_state.p1.lp,
                "mcts": game_state.p2.lp
            },
            "process_metrics": collector.get_summary_metrics(),
            "detailed_metrics": collector.metrics
        }
        
        self.game_results.append(result)
        return result
    
    def run_adaptation_study(self, llm_agent, num_games: int = 10) -> Dict:
        """
        Run adaptation study over consecutive games.
        
        Args:
            llm_agent: LLM agent to test
            num_games: Number of consecutive games
            
        Returns:
            Adaptation analysis results
        """
        performances = []
        all_metrics = []
        
        for i in range(num_games):
            # Setup new game
            game_state = self._setup_new_game()
            
            # Run game with metrics
            result = self.evaluate_game_with_metrics(
                game_state, llm_agent, self._create_mcts_opponent()
            )
            
            # Track performance
            win = 1.0 if result["winner"] == "llm" else 0.0
            performances.append(win)
            all_metrics.append(result["process_metrics"])
            
            # Calculate adaptation
            if i > 0:
                collector = ProcessMetricsCollector()
                adaptation = collector.calculate_adaptation_score(win, performances[:-1])
                result["adaptation_score"] = adaptation
        
        # Analyze adaptation trend
        early_performance = np.mean(performances[:3]) if len(performances) >= 3 else np.mean(performances)
        late_performance = np.mean(performances[-3:]) if len(performances) >= 3 else np.mean(performances)
        improvement = late_performance - early_performance
        
        return {
            "num_games": num_games,
            "performances": performances,
            "early_performance": early_performance,
            "late_performance": late_performance,
            "improvement": improvement,
            "shows_adaptation": improvement > 0.1,
            "all_metrics": all_metrics
        }
    
    def _setup_new_game(self) -> GameState:
        """Setup a new game state."""
        from game import Player, get_card_pool
        
        p1 = Player("LLM")
        p2 = Player("MCTS")
        
        # Setup decks
        cards = get_card_pool("en")
        p1.deck = [Card(c) for c in cards[:len(cards)//2]]
        p2.deck = [Card(c) for c in cards[len(cards)//2:]]
        
        # Shuffle
        import random
        random.shuffle(p1.deck)
        random.shuffle(p2.deck)
        
        # Draw initial hands
        for _ in range(3):
            p1.draw()
            p2.draw()
        
        return GameState(p1, p2)
    
    def _create_mcts_opponent(self):
        """Create MCTS opponent."""
        from agents import MCTSAgent
        return MCTSAgent("MCTS", rollout_count=10)
    
    def generate_process_metrics_report(self) -> Dict:
        """Generate comprehensive process metrics report."""
        if not self.game_results:
            return {"error": "No games evaluated"}
        
        # Aggregate metrics across all games
        aggregated = defaultdict(list)
        
        for result in self.game_results:
            metrics = result["process_metrics"]
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    aggregated[key].append(value)
        
        # Calculate statistics
        report = {}
        for key, values in aggregated.items():
            report[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "median": np.median(values)
            }
        
        return {
            "num_games": len(self.game_results),
            "win_rate": sum(1 for r in self.game_results if r["winner"] == "llm") / len(self.game_results),
            "aggregated_metrics": report
        }


def test_process_metrics():
    """Test the process metrics module."""
    print("Testing Process Metrics Module")
    print("=" * 60)
    
    collector = ProcessMetricsCollector()
    
    # Simulate some metrics
    from game import GameState, Player
    p1 = Player("TestPlayer1")
    p2 = Player("TestPlayer2")
    game_state = GameState(p1, p2)
    
    # Record some test metrics
    collector.record_move_attempt("Play Mighty Warrior", True, game_state)
    collector.record_decision_time(2.5, 1)
    collector.record_token_usage(150, 50, 1)
    
    efficiency = collector.calculate_resource_efficiency(game_state, p1)
    print(f"Resource efficiency: {efficiency:.2f}")
    
    consistency = collector.evaluate_strategic_consistency("Play Fireball", game_state)
    print(f"Strategic consistency: {consistency:.2f}")
    
    advantage = collector.calculate_turn_advantage(game_state)
    print(f"Turn advantage: {advantage:.2f}")
    
    # Get summary
    summary = collector.get_summary_metrics()
    print("\nSummary Metrics:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_process_metrics()