"""
Counterfactual Regret Minimization Agent for Sacra Battle.

This implementation follows the vanilla CFR algorithm with:
- Information set abstraction for tractability
- Regret matching for strategy computation
- Average strategy tracking for convergence to Nash equilibrium
"""

import time
import random
import logging
import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import Agent
from game import GameState, Player, Card, resolve_card_effect, combat_phase
try:
    from .information_set import InfoSet, StateAbstractor, abstract_action
except:
    from information_set import InfoSet, StateAbstractor, abstract_action
try:
    from .abstraction_levels import AbstractionLevel, create_abstractor
except:
    from abstraction_levels import AbstractionLevel, create_abstractor

logger = logging.getLogger("SacraBattle")


class CFRAgent(Agent):
    """
    CFR-based agent implementing counterfactual regret minimization.

    The agent maintains:
    - Cumulative regret for each information set and action
    - Strategy sum for computing average strategy (Nash equilibrium)
    - Current strategy computed via regret matching
    """

    def __init__(
        self,
        name: str = "CFRAgent",
        exploration_epsilon: float = 0.0,
        model_path: Optional[str] = None,
        abstraction_level: AbstractionLevel = AbstractionLevel.HIGH
    ):
        """
        Initialize CFR agent.

        Args:
            name: Agent name
            exploration_epsilon: Epsilon for epsilon-greedy exploration during play
            abstraction_level: Level of state abstraction (HIGH/MEDIUM/LOW)
            model_path: Path to load pre-trained model
        """
        super().__init__(name)

        # CFR data structures
        self.regret_sum = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))

        # Store abstraction level
        self.abstraction_level = abstraction_level

        # Create appropriate abstractor based on level
        self.abstractor = create_abstractor(abstraction_level)

        # Exploration parameter (for playing, not training)
        self.exploration_epsilon = exploration_epsilon

        # Training statistics
        self.iteration_count = 0

        if model_path:
            self.load_model(model_path)

    def choose_move(self, game_state: GameState, player: Player) -> Optional[Card]:
        """
        Choose a move using the learned CFR strategy.

        With probability epsilon: random move (exploration)
        Otherwise: sample from computed strategy distribution
        """
        start_time = time.time()

        if not player.hand:
            player.decision_times.append(time.time() - start_time)
            return None

        # Get information set
        info_set = self.abstractor.abstract_state(game_state, player)

        # Epsilon-greedy exploration
        if random.random() < self.exploration_epsilon:
            chosen = random.choice(player.hand)
        else:
            # Get strategy for this information set
            strategy = self.get_average_strategy(info_set, player.hand)

            if not strategy:
                # No data for this info set, play randomly
                chosen = random.choice(player.hand)
            else:
                # Sample action from strategy distribution
                chosen = self._sample_from_strategy(strategy, player.hand)

        decision_time = time.time() - start_time
        player.decision_times.append(decision_time)

        logger.debug(
            f"{player.name} (CFR) chooses {chosen.name if chosen else 'None'} "
            f"in {decision_time:.4f}s (info_set: {info_set})"
        )

        return chosen

    def get_current_strategy(self, info_set: InfoSet, actions: List[Card]) -> Dict[str, float]:
        """
        Get current strategy using regret matching.

        Strategy is proportional to positive regrets.
        If all regrets are negative, use uniform distribution.
        """
        if not actions:
            return {}

        regrets = self.regret_sum[info_set]
        strategy = {}
        normalizing_sum = 0.0

        # Compute positive regrets
        for card in actions:
            action_key = abstract_action(card)
            positive_regret = max(0.0, regrets.get(action_key, 0.0))
            strategy[action_key] = positive_regret
            normalizing_sum += positive_regret

        # Normalize or use uniform
        if normalizing_sum > 0:
            for action_key in strategy:
                strategy[action_key] /= normalizing_sum
        else:
            # Uniform distribution
            uniform_prob = 1.0 / len(actions)
            for card in actions:
                strategy[abstract_action(card)] = uniform_prob

        return strategy

    def get_average_strategy(self, info_set: InfoSet, actions: List[Card]) -> Dict[str, float]:
        """
        Get average strategy (converges to Nash equilibrium).

        This is the strategy to use for actual play.
        """
        if info_set not in self.strategy_sum:
            return {}

        strategy_sum = self.strategy_sum[info_set]
        if not strategy_sum:
            return {}

        avg_strategy = {}
        normalizing_sum = sum(strategy_sum.values())

        if normalizing_sum > 0:
            for card in actions:
                action_key = abstract_action(card)
                if action_key in strategy_sum:
                    avg_strategy[action_key] = strategy_sum[action_key] / normalizing_sum
                else:
                    avg_strategy[action_key] = 0.0
        else:
            # Uniform if no data
            uniform_prob = 1.0 / len(actions)
            for card in actions:
                avg_strategy[abstract_action(card)] = uniform_prob

        return avg_strategy

    def _sample_from_strategy(
        self, strategy: Dict[str, float], actions: List[Card]
    ) -> Card:
        """Sample an action from strategy distribution."""
        # Build probability distribution
        action_probs = []
        action_cards = []

        for card in actions:
            action_key = abstract_action(card)
            prob = strategy.get(action_key, 0.0)
            action_probs.append(prob)
            action_cards.append(card)

        # Handle edge case of all zero probabilities
        if sum(action_probs) == 0:
            return random.choice(actions)

        # Weighted random selection
        return random.choices(action_cards, weights=action_probs, k=1)[0]

    def update_regrets(
        self,
        info_set: InfoSet,
        actions: List[Card],
        action_values: Dict[str, float],
        current_value: float
    ):
        """
        Update cumulative regrets for an information set.

        Regret for action a = value(a) - current_value
        """
        for card in actions:
            action_key = abstract_action(card)
            regret = action_values.get(action_key, 0.0) - current_value
            self.regret_sum[info_set][action_key] += regret

    def update_strategy_sum(
        self,
        info_set: InfoSet,
        strategy: Dict[str, float],
        reach_probability: float
    ):
        """
        Update strategy sum weighted by reach probability.

        Used to compute average strategy over all iterations.
        """
        for action_key, prob in strategy.items():
            self.strategy_sum[info_set][action_key] += reach_probability * prob

    def save_model(self, path: str):
        """Save trained CFR model to disk."""
        model_data = {
            "regret_sum": dict(self.regret_sum),
            "strategy_sum": dict(self.strategy_sum),
            "iteration_count": self.iteration_count,
            "abstraction_level": self.abstraction_level.value  # Save abstraction level
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"CFR model saved to {path} (iterations: {self.iteration_count}, abstraction: {self.abstraction_level.value})")

    def load_model(self, path: str):
        """Load trained CFR model from disk."""
        with open(path, "rb") as f:
            model_data = pickle.load(f)

        self.regret_sum = defaultdict(lambda: defaultdict(float), model_data["regret_sum"])
        self.strategy_sum = defaultdict(lambda: defaultdict(float), model_data["strategy_sum"])
        self.iteration_count = model_data.get("iteration_count", 0)

        # Load abstraction level and recreate abstractor
        if "abstraction_level" in model_data:
            saved_level = AbstractionLevel(model_data["abstraction_level"])
            if saved_level != self.abstraction_level:
                logger.warning(f"Model was trained with {saved_level.value} abstraction, but agent uses {self.abstraction_level.value}")
            self.abstraction_level = saved_level
            self.abstractor = create_abstractor(saved_level)

        logger.info(f"CFR model loaded from {path} (iterations: {self.iteration_count}, abstraction: {self.abstraction_level.value})")

    def get_info_set_count(self) -> int:
        """Return number of unique information sets encountered."""
        return len(self.regret_sum)

    def get_exploitability(self) -> float:
        """
        Calculate exploitability bound based on average overall regret.

        According to Zinkevich et al. (2007), if average regret ≤ ε,
        then the average strategy is a 2ε-Nash equilibrium.

        This returns the average regret per iteration (NOT normalized by info sets).
        The actual exploitability is bounded by 2 * this value.

        True exploitability would require computing best response strategies,
        which is computationally expensive.
        """
        if self.iteration_count == 0:
            return float('inf')

        # Sum of positive counterfactual regrets across all info sets
        total_regret = 0.0
        for info_set_regrets in self.regret_sum.values():
            if info_set_regrets:
                # Take the maximum regret for this info set (worst-case action regret)
                max_regret = max(info_set_regrets.values())
                # Only positive regrets contribute to exploitability
                total_regret += max(0, max_regret)

        # Average regret per iteration (Theorem 4, Zinkevich et al. 2007)
        # This gives us ε where the strategy is a 2ε-Nash equilibrium
        return total_regret / self.iteration_count

    def get_convergence_metrics(self) -> Dict[str, float]:
        """
        Get additional metrics for tracking convergence.

        Returns:
            Dictionary with various convergence metrics:
            - exploitability: Average regret bound
            - info_sets_total: Total unique information sets
            - info_sets_active: Info sets with non-zero regret
            - avg_actions_per_infoset: Average number of actions per info set
            - max_single_regret: Maximum regret for any single action
        """
        if self.iteration_count == 0:
            return {
                'exploitability': float('inf'),
                'info_sets_total': 0,
                'info_sets_active': 0,
                'avg_actions_per_infoset': 0,
                'max_single_regret': 0
            }

        info_sets_total = len(self.regret_sum)
        info_sets_active = sum(1 for regrets in self.regret_sum.values() if any(r != 0 for r in regrets.values()))

        total_actions = sum(len(regrets) for regrets in self.regret_sum.values())
        avg_actions = total_actions / max(1, info_sets_total)

        max_regret = 0.0
        for regrets in self.regret_sum.values():
            if regrets:
                max_regret = max(max_regret, max(abs(r) for r in regrets.values()))

        return {
            'exploitability': self.get_exploitability(),
            'info_sets_total': info_sets_total,
            'info_sets_active': info_sets_active,
            'avg_actions_per_infoset': avg_actions,
            'max_single_regret': max_regret
        }