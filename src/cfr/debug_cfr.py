#!/usr/bin/env python3
"""Debug script to find where CFR is hanging."""

import sys
import traceback
from cfr_agent import CFRAgent
from cfr_trainer import CFRTrainer
from abstraction_levels import AbstractionLevel

print("Creating CFR agent...")
agent = CFRAgent(name="TestCFR", abstraction_level=AbstractionLevel.HIGH)
print(f"Agent created with abstractor: {agent.abstractor}")

print("Creating trainer...")
trainer = CFRTrainer(agent, card_pool_type="community", language="en")
print(f"Trainer created with card pool size: {len(trainer.card_pool)}")

print("Creating random game state...")
try:
    game_state = trainer._create_random_game_state()
    print(f"Game state created - Player 1 hand: {len(game_state.players[0].hand)}, Player 2 hand: {len(game_state.players[1].hand)}")
    print(f"Player 1 LP: {game_state.players[0].lp}, Player 2 LP: {game_state.players[1].lp}")
except Exception as e:
    print(f"Error creating game state: {e}")
    traceback.print_exc()
    sys.exit(1)

print("Running one CFR iteration...")
try:
    value = trainer._cfr(game_state, 1.0, 1.0)
    print(f"CFR iteration complete, value: {value}")
except Exception as e:
    print(f"Error in CFR iteration: {e}")
    traceback.print_exc()
    sys.exit(1)

print("Test complete!")