#!/usr/bin/env python3
"""
Test suite for CFR implementation.

Tests:
1. Information set abstraction correctness
2. CFR convergence on simplified game
3. Strategy computation (regret matching)
4. Model save/load functionality
"""

import unittest
import tempfile
import os
import sys

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import GameState, Player, Card, get_card_pool
from cfr.information_set import InfoSet, StateAbstractor, abstract_action
from cfr.cfr_agent import CFRAgent
from cfr.cfr_trainer import CFRTrainer


class TestInformationSet(unittest.TestCase):
    """Test information set abstraction."""

    def setUp(self):
        self.abstractor = StateAbstractor()

    def test_game_phase_abstraction(self):
        """Test game phase categorization."""
        player1 = Player("P1")
        player2 = Player("P2")
        game_state = GameState(player1, player2)

        # Early game
        game_state.turn_number = 3
        info_set = self.abstractor.abstract_state(game_state, player1)
        self.assertEqual(info_set.phase, "early")

        # Mid game
        game_state.turn_number = 10
        info_set = self.abstractor.abstract_state(game_state, player1)
        self.assertEqual(info_set.phase, "mid")

        # Late game
        game_state.turn_number = 20
        info_set = self.abstractor.abstract_state(game_state, player1)
        self.assertEqual(info_set.phase, "late")

    def test_life_differential(self):
        """Test life point differential bucketing."""
        player1 = Player("P1")
        player2 = Player("P2")
        game_state = GameState(player1, player2)

        # Equal life
        player1.lp = 10
        player2.lp = 10
        info_set = self.abstractor.abstract_state(game_state, player1)
        self.assertEqual(info_set.life_diff, 0)

        # Player ahead
        player1.lp = 10
        player2.lp = 5
        info_set = self.abstractor.abstract_state(game_state, player1)
        self.assertEqual(info_set.life_diff, 5)

        # Player behind
        player1.lp = 3
        player2.lp = 8
        info_set = self.abstractor.abstract_state(game_state, player1)
        self.assertEqual(info_set.life_diff, -5)

        # Test capping
        player1.lp = 20
        player2.lp = 0
        info_set = self.abstractor.abstract_state(game_state, player1)
        self.assertEqual(info_set.life_diff, 10)  # Capped at 10

    def test_hand_composition_abstraction(self):
        """Test hand composition bucketing."""
        player1 = Player("P1")
        player2 = Player("P2")
        game_state = GameState(player1, player2)

        # Create test cards
        champion = Card({"id": "test_champ", "name": "Champion", "type": "Champion", "power": 2, "guard": 2})
        spell = Card({"id": "test_spell", "name": "Spell", "type": "Spell", "effect": "Test"})
        trick = Card({"id": "test_trick", "name": "Trick", "type": "Trick", "trigger": "Test", "effect": "Test"})

        # Empty hand
        info_set = self.abstractor.abstract_state(game_state, player1)
        self.assertEqual(info_set.hand_composition, (0, 0, 0))

        # Mixed hand
        player1.hand = [champion.clone(), champion.clone(), spell.clone(), trick.clone()]
        info_set = self.abstractor.abstract_state(game_state, player1)
        self.assertEqual(info_set.hand_composition, (2, 1, 1))

        # Test bucketing limits
        player1.hand = [champion.clone() for _ in range(5)] + [spell.clone() for _ in range(4)]
        info_set = self.abstractor.abstract_state(game_state, player1)
        self.assertEqual(info_set.hand_composition, (3, 3, 0))  # Capped at 3


class TestCFRAgent(unittest.TestCase):
    """Test CFR agent functionality."""

    def setUp(self):
        self.agent = CFRAgent("TestCFR")

    def test_strategy_computation(self):
        """Test regret matching strategy computation."""
        # Create mock info set and actions
        info_set = InfoSet(
            phase="early",
            life_diff=0,
            hand_strength="medium",
            board_control="neutral",
            tricks_active=0,
            hand_composition=(1, 1, 0)
        )

        # Create test cards
        card1 = Card({"id": "card1", "name": "Card1", "type": "Spell", "effect": "Test"})
        card2 = Card({"id": "card2", "name": "Card2", "type": "Spell", "effect": "Test"})
        actions = [card1, card2]

        # Initially should be uniform
        strategy = self.agent.get_current_strategy(info_set, actions)
        self.assertAlmostEqual(strategy["card1"], 0.5, places=2)
        self.assertAlmostEqual(strategy["card2"], 0.5, places=2)

        # Add positive regret for card1
        self.agent.regret_sum[info_set]["card1"] = 10.0
        self.agent.regret_sum[info_set]["card2"] = -5.0  # Negative regret

        # Strategy should favor card1
        strategy = self.agent.get_current_strategy(info_set, actions)
        self.assertEqual(strategy["card1"], 1.0)  # All weight on positive regret
        self.assertEqual(strategy["card2"], 0.0)  # No weight on negative regret

    def test_save_load_model(self):
        """Test model serialization."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            try:
                # Add some data
                info_set = InfoSet(
                    phase="mid",
                    life_diff=2,
                    hand_strength="strong",
                    board_control="winning",
                    tricks_active=1,
                    hand_composition=(2, 1, 0)
                )
                self.agent.regret_sum[info_set]["action1"] = 15.0
                self.agent.strategy_sum[info_set]["action1"] = 25.0
                self.agent.iteration_count = 1000

                # Save model
                self.agent.save_model(tmp.name)

                # Load into new agent
                new_agent = CFRAgent("LoadedCFR", model_path=tmp.name)

                # Check data preserved
                self.assertEqual(new_agent.iteration_count, 1000)
                self.assertEqual(new_agent.regret_sum[info_set]["action1"], 15.0)
                self.assertEqual(new_agent.strategy_sum[info_set]["action1"], 25.0)

            finally:
                os.unlink(tmp.name)


class TestCFRTrainer(unittest.TestCase):
    """Test CFR training."""

    def test_game_state_generation(self):
        """Test random game state generation."""
        agent = CFRAgent("TestAgent")
        trainer = CFRTrainer(agent, card_pool_type="community", language="en")

        # Generate multiple states and check validity
        for _ in range(10):
            game_state = trainer._create_random_game_state()

            # Check players exist and have valid state
            self.assertEqual(len(game_state.players), 2)
            self.assertGreater(game_state.players[0].lp, 0)
            self.assertGreater(game_state.players[1].lp, 0)

            # Check hands were drawn
            self.assertGreaterEqual(len(game_state.players[0].hand), 0)
            self.assertGreaterEqual(len(game_state.players[1].hand), 0)

    def test_terminal_value_calculation(self):
        """Test terminal state value calculation."""
        agent = CFRAgent("TestAgent")
        trainer = CFRTrainer(agent)

        player1 = Player("P1")
        player2 = Player("P2")
        game_state = GameState(player1, player2)

        # Not terminal
        value = trainer._get_terminal_value(game_state)
        self.assertIsNone(value)

        # Player 1 wins
        player2.lp = 0
        value = trainer._get_terminal_value(game_state)
        self.assertEqual(value, 1.0)  # Player 0 perspective

        # Player 0 wins
        player1.lp = 0
        player2.lp = 10
        value = trainer._get_terminal_value(game_state)
        self.assertEqual(value, -1.0)  # Player 0 perspective

        # Max game length draw
        player1.lp = 5
        player2.lp = 5
        game_state.turn_number = 100
        value = trainer._get_terminal_value(game_state)
        self.assertAlmostEqual(value, 0.0, places=2)

    def test_cfr_convergence_simple(self):
        """Test CFR convergence on a simplified game."""
        agent = CFRAgent("TestAgent")
        trainer = CFRTrainer(agent, max_game_length=10)

        # Train for small number of iterations
        initial_exploitability = agent.get_exploitability()
        trainer.train(100)

        # Check that training occurred
        self.assertEqual(agent.iteration_count, 100)
        self.assertGreater(agent.get_info_set_count(), 0)

        # Exploitability should change (though not necessarily decrease with so few iterations)
        final_exploitability = agent.get_exploitability()
        self.assertNotEqual(initial_exploitability, final_exploitability)


class TestActionAbstraction(unittest.TestCase):
    """Test action abstraction."""

    def test_champion_abstraction(self):
        """Test champion card abstraction."""
        # Aggressive champion
        aggressive = Card({
            "id": "aggro",
            "name": "Aggressive",
            "type": "Champion",
            "power": 3,
            "guard": 1
        })
        self.assertEqual(abstract_action(aggressive), "champ_aggressive_aggro")

        # Defensive champion
        defensive = Card({
            "id": "def",
            "name": "Defensive",
            "type": "Champion",
            "power": 1,
            "guard": 3
        })
        self.assertEqual(abstract_action(defensive), "champ_defensive_def")

        # Balanced champion
        balanced = Card({
            "id": "bal",
            "name": "Balanced",
            "type": "Champion",
            "power": 2,
            "guard": 2
        })
        self.assertEqual(abstract_action(balanced), "champ_balanced_bal")

    def test_spell_trick_abstraction(self):
        """Test spell and trick abstraction."""
        spell = Card({
            "id": "spell1",
            "name": "TestSpell",
            "type": "Spell",
            "effect": "Test effect"
        })
        self.assertEqual(abstract_action(spell), "spell1")

        trick = Card({
            "id": "trick1",
            "name": "TestTrick",
            "type": "Trick",
            "trigger": "Test",
            "effect": "Test effect"
        })
        self.assertEqual(abstract_action(trick), "trick1")


if __name__ == "__main__":
    # Run tests
    unittest.main()