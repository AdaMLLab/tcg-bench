"""
CFR training module for Sacra Battle.

Implements the training loop with:
- Vanilla CFR traversal
- Monte Carlo CFR (MCCFR) for sampling
- Self-play game generation
"""

import random
import logging
import copy
from typing import Optional, Dict, Tuple
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import GameState, Player, Card, resolve_card_effect, combat_phase, get_card_pool
try:
    from .cfr_agent import CFRAgent
except:
    from cfr_agent import CFRAgent
try:
    from .information_set import abstract_action
except:
    from information_set import abstract_action

logger = logging.getLogger("SacraBattle.CFR")


class CFRTrainer:
    """
    Trains a CFR agent through self-play.

    Implements vanilla CFR with recursive game tree traversal.
    """

    def __init__(
        self,
        cfr_agent: CFRAgent,
        language: str = "en",
        max_game_length: int = 50,
        **kwargs
    ):
        """
        Initialize CFR trainer.

        Args:
            cfr_agent: The CFR agent to train
            language: "en" or "ar"
            max_game_length: Maximum turns before declaring draw
        """
        self.agent = cfr_agent
        self.language = language
        self.max_game_length = max_game_length
        self.card_pool = get_card_pool(language)

    def train(self, iterations: int):
        """
        Run CFR training for specified iterations.

        Each iteration:
        1. Generate random starting game state
        2. Run CFR traversal to compute regrets
        3. Update strategy based on regrets
        """
        logger.info(f"Starting CFR training for {iterations} iterations")

        for i in range(iterations):
            # More frequent logging for debugging
            log_interval = min(100, max(1, iterations // 10))
            if i % log_interval == 0:
                logger.info(
                    f"Iteration {i}/{iterations} - "
                    f"Info sets: {self.agent.get_info_set_count()}, "
                    f"Exploitability: {self.agent.get_exploitability():.4f}"
                )

            # Generate random starting state
            game_state = self._create_random_game_state()

            # Run CFR iteration
            self._cfr(game_state, 1.0, 1.0)

            # Update iteration counter
            self.agent.iteration_count += 1

        logger.info(
            f"Training complete. Final stats - "
            f"Info sets: {self.agent.get_info_set_count()}, "
            f"Exploitability: {self.agent.get_exploitability():.4f}"
        )

    def _create_random_game_state(self) -> GameState:
        """Create a random starting game state."""
        # Create players
        player1 = Player("Player1")
        player2 = Player("Player2")

        # Create random decks
        cards = [Card(card_data) for card_data in self.card_pool]
        random.shuffle(cards)

        # Split cards between players (or give full deck to each)
        deck_size = len(cards) // 2
        player1.deck = [c.clone() for c in cards[:deck_size]]
        player2.deck = [c.clone() for c in cards[deck_size:]]

        # Shuffle decks
        random.shuffle(player1.deck)
        random.shuffle(player2.deck)

        # Draw initial hands (3 cards each)
        for _ in range(3):
            player1.draw()
            player2.draw()

        # Create game state
        game_state = GameState(player1, player2)

        # Randomly advance game to various states for better coverage
        if random.random() < 0.3:  # 30% chance to start mid-game
            turns_to_advance = random.randint(1, 10)
            for _ in range(turns_to_advance):
                if game_state.players[0].lp <= 0 or game_state.players[1].lp <= 0:
                    break
                self._play_random_turn(game_state)

        return game_state

    def _play_random_turn(self, game_state: GameState):
        """Play a random turn to advance game state."""
        current = game_state.current_player()
        current.draw()

        if current.hand:
            card = random.choice(current.hand)
            current.play_card(card)
            resolve_card_effect(game_state, current, card)

        combat_phase(game_state)
        game_state.next_turn()

    def _cfr(
        self,
        game_state: GameState,
        p0_reach: float,
        p1_reach: float
    ) -> float:
        """
        Recursive CFR traversal.

        Args:
            game_state: Current game state
            p0_reach: Player 0's reach probability
            p1_reach: Player 1's reach probability

        Returns:
            Expected value for the current player
        """
        # Terminal node check
        value = self._get_terminal_value(game_state)
        if value is not None:
            return value

        # Get current player
        current_idx = game_state.current_player_idx
        current_player = game_state.current_player()

        # Get available actions
        actions = current_player.hand if current_player.hand else []

        if not actions:
            # No actions available, pass turn
            game_state_copy = game_state.clone()
            game_state_copy.next_turn()
            return -self._cfr(game_state_copy, p1_reach, p0_reach)

        # Get information set
        info_set = self.agent.abstractor.abstract_state(game_state, current_player)

        # Get current strategy
        strategy = self.agent.get_current_strategy(info_set, actions)

        # Calculate action values
        action_values = {}
        node_value = 0.0

        for card in actions:
            action_key = abstract_action(card)
            action_prob = strategy.get(action_key, 0.0)

            # Calculate new reach probabilities
            if current_idx == 0:
                new_p0 = p0_reach * action_prob
                new_p1 = p1_reach
            else:
                new_p0 = p0_reach
                new_p1 = p1_reach * action_prob

            # Recurse with action applied
            game_state_copy = game_state.clone()
            current_copy = game_state_copy.current_player()

            # Find and play the matching card
            matching_card = None
            for c in current_copy.hand:
                if c.name == card.name and c.id == card.id:
                    matching_card = c
                    break

            if matching_card:
                current_copy.play_card(matching_card)
                resolve_card_effect(game_state_copy, current_copy, matching_card)
                combat_phase(game_state_copy)
                game_state_copy.next_turn()

                # Recursive call (opponent's perspective, so negate)
                action_value = -self._cfr(game_state_copy, new_p0, new_p1)
                action_values[action_key] = action_value
                node_value += action_prob * action_value

        # Update regrets
        if current_idx == 0:
            cfr_reach = p1_reach  # Opponent's reach probability
        else:
            cfr_reach = p0_reach

        # Calculate and accumulate regrets
        for card in actions:
            action_key = abstract_action(card)
            regret = action_values.get(action_key, 0.0) - node_value
            self.agent.regret_sum[info_set][action_key] += cfr_reach * regret

        # Update strategy sum
        if current_idx == 0:
            reach = p0_reach
        else:
            reach = p1_reach

        self.agent.update_strategy_sum(info_set, strategy, reach)

        return node_value

    def _get_terminal_value(self, game_state: GameState) -> Optional[float]:
        """
        Check if state is terminal and return value.

        Returns:
            None if not terminal
            Value from current player's perspective if terminal
        """
        # Check for player death
        if game_state.players[0].lp <= 0:
            # Player 0 lost
            return -1.0 if game_state.current_player_idx == 0 else 1.0
        elif game_state.players[1].lp <= 0:
            # Player 1 lost
            return 1.0 if game_state.current_player_idx == 0 else -1.0

        # Check for max game length
        if game_state.turn_number >= self.max_game_length:
            # Draw - return small value based on life differential
            life_diff = game_state.players[0].lp - game_state.players[1].lp
            value = 0.01 * life_diff  # Small value to break ties
            return value if game_state.current_player_idx == 0 else -value

        return None


class MCCFRTrainer(CFRTrainer):
    """
    Monte Carlo CFR trainer using sampling instead of full traversal.

    More efficient for larger games but requires more iterations.
    """

    def _cfr(
        self,
        game_state: GameState,
        p0_reach: float,
        p1_reach: float
    ) -> float:
        """
        Monte Carlo CFR with outcome sampling.

        Only samples one action per decision point.
        """
        # Terminal node check
        value = self._get_terminal_value(game_state)
        if value is not None:
            return value

        # Get current player
        current_idx = game_state.current_player_idx
        current_player = game_state.current_player()

        # Get available actions
        actions = current_player.hand if current_player.hand else []

        if not actions:
            # No actions, pass turn
            game_state_copy = game_state.clone()
            game_state_copy.next_turn()
            return -self._cfr(game_state_copy, p1_reach, p0_reach)

        # Get information set
        info_set = self.agent.abstractor.abstract_state(game_state, current_player)

        # Get strategy
        strategy = self.agent.get_current_strategy(info_set, actions)

        # Sample action according to strategy
        action_probs = []
        for card in actions:
            action_key = abstract_action(card)
            action_probs.append(strategy.get(action_key, 0.0))

        # Handle zero probabilities
        if sum(action_probs) == 0:
            sampled_idx = random.randint(0, len(actions) - 1)
            sampled_prob = 1.0 / len(actions)
        else:
            sampled_idx = random.choices(
                range(len(actions)),
                weights=action_probs,
                k=1
            )[0]
            sampled_prob = action_probs[sampled_idx]

        sampled_card = actions[sampled_idx]
        sampled_key = abstract_action(sampled_card)

        # Update reach probabilities
        if current_idx == 0:
            new_p0 = p0_reach * sampled_prob
            new_p1 = p1_reach
        else:
            new_p0 = p0_reach
            new_p1 = p1_reach * sampled_prob

        # Apply sampled action
        game_state_copy = game_state.clone()
        current_copy = game_state_copy.current_player()

        # Find and play matching card
        for c in current_copy.hand:
            if c.name == sampled_card.name and c.id == sampled_card.id:
                current_copy.play_card(c)
                resolve_card_effect(game_state_copy, current_copy, c)
                break

        combat_phase(game_state_copy)
        game_state_copy.next_turn()

        # Recursive call
        sampled_value = -self._cfr(game_state_copy, new_p0, new_p1)

        # Update regrets for sampled action only (MCCFR)
        if current_idx == 0:
            cfr_reach = p1_reach
        else:
            cfr_reach = p0_reach

        # Compute counterfactual values for all actions
        for i, card in enumerate(actions):
            action_key = abstract_action(card)

            if i == sampled_idx:
                # This was the sampled action
                action_value = sampled_value
            else:
                # Counterfactual value is 0 for unsampled actions in MCCFR
                action_value = 0

            # Update regret
            regret = action_value - sampled_value * sampled_prob
            self.agent.regret_sum[info_set][action_key] += cfr_reach * regret

        # Update strategy sum for sampled action
        if current_idx == 0:
            reach = p0_reach
        else:
            reach = p1_reach

        self.agent.strategy_sum[info_set][sampled_key] += reach * sampled_prob

        return sampled_value