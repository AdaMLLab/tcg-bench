#!/usr/bin/env python3
"""Debug script to test training loop."""

import time
from cfr_agent import CFRAgent
from cfr_trainer import CFRTrainer
from abstraction_levels import AbstractionLevel

print("Creating CFR agent...")
agent = CFRAgent(name="TestCFR", abstraction_level=AbstractionLevel.HIGH)

print("Creating trainer...")
trainer = CFRTrainer(agent, card_pool_type="community", language="en")

print("Starting training for 5 iterations...")
start = time.time()

for i in range(5):
    print(f"\nIteration {i}...")
    game_state = trainer._create_random_game_state()
    print(f"  Game state created")

    value = trainer._cfr(game_state, 1.0, 1.0)
    print(f"  CFR complete, value: {value:.4f}")

    agent.iteration_count += 1
    print(f"  Info sets: {agent.get_info_set_count()}, Exploitability: {agent.get_exploitability():.6f}")

print(f"\nTraining complete in {time.time() - start:.2f} seconds")
print(f"Final exploitability: {agent.get_exploitability():.6f}")