#!/usr/bin/env python3
"""
Test different abstraction levels for CFR.
"""

import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import GameState, Player, Card, get_card_pool
from abstraction_levels import (
    AbstractionLevel,
    HighAbstractor,
    MediumAbstractor,
    LowAbstractor,
    create_abstractor
)
from cfr_agent import CFRAgent
from cfr_trainer import CFRTrainer


def test_abstraction_levels():
    """Test that different abstraction levels produce different info sets."""

    # Create test game state
    player1 = Player("P1")
    player2 = Player("P2")

    # Setup test state
    cards = get_card_pool("en", "community")
    test_cards = [Card(card_data) for card_data in cards[:5]]

    player1.lp = 8
    player2.lp = 6
    player1.hand = test_cards[:3]
    player2.hand = test_cards[3:5]
    player1.board = [test_cards[0].clone()]

    game_state = GameState(player1, player2)
    game_state.turn_number = 7

    # Test each abstractor
    abstractors = {
        "High": HighAbstractor(),
        "Medium": MediumAbstractor(),
        "Low": LowAbstractor()
    }

    print("Testing abstraction levels on same game state:")
    print(f"Turn: {game_state.turn_number}, P1 LP: {player1.lp}, P2 LP: {player2.lp}")
    print(f"P1 Hand size: {len(player1.hand)}, P1 Board size: {len(player1.board)}")
    print()

    for name, abstractor in abstractors.items():
        info_set = abstractor.abstract_state(game_state, player1)
        print(f"{name} Abstraction:")
        print(f"  Type: {type(info_set).__name__}")
        print(f"  Hash: {hash(info_set)}")
        print(f"  String: {str(info_set)[:100]}...")
        print()


def test_cfr_with_abstraction():
    """Test CFR training with different abstraction levels."""

    print("Testing CFR with different abstraction levels:")
    print()

    for level in [AbstractionLevel.HIGH, AbstractionLevel.MEDIUM]:
        print(f"Testing {level.value} abstraction:")

        # Create agent with abstraction level
        agent = CFRAgent(f"CFR_{level.value}", abstraction_level=level)

        # Create trainer
        trainer = CFRTrainer(agent, max_game_length=10)

        # Train for small number of iterations
        trainer.train(20)

        print(f"  Iterations: {agent.iteration_count}")
        print(f"  Info sets discovered: {agent.get_info_set_count()}")
        print(f"  Exploitability: {agent.get_exploitability():.6f}")

        # Show sample info set
        if agent.regret_sum:
            sample_info_set = list(agent.regret_sum.keys())[0]
            print(f"  Sample info set type: {type(sample_info_set).__name__}")

        print()


def test_info_set_counts():
    """Compare info set counts across abstraction levels."""

    print("Comparing info set discovery rates:")
    print()

    iterations = 50

    results = {}

    for level in [AbstractionLevel.HIGH, AbstractionLevel.MEDIUM]:
        # Create agent and trainer
        agent = CFRAgent(f"Test_{level.value}", abstraction_level=level)
        trainer = CFRTrainer(agent, max_game_length=15)

        # Train
        trainer.train(iterations)

        results[level.value] = {
            'info_sets': agent.get_info_set_count(),
            'exploitability': agent.get_exploitability()
        }

    # Display comparison
    print(f"After {iterations} iterations:")
    for level, data in results.items():
        print(f"  {level.capitalize():8} - Info sets: {data['info_sets']:5}, Exploitability: {data['exploitability']:.6f}")

    # Check that medium has more info sets than high
    if results['medium']['info_sets'] > results['high']['info_sets']:
        print("\n✓ Medium abstraction correctly discovers more info sets than high")
    else:
        print("\n✗ Warning: Medium should discover more info sets than high")


if __name__ == "__main__":
    print("=" * 60)
    print("CFR Abstraction Level Tests")
    print("=" * 60)
    print()

    test_abstraction_levels()
    print("=" * 60)
    test_cfr_with_abstraction()
    print("=" * 60)
    test_info_set_counts()
    print("=" * 60)

    print("\nAll abstraction level tests completed!")