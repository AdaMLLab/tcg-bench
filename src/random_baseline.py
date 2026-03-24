"""Random baseline: Random agent (Player 1) vs various opponents."""
import sys
import os
import random
import json
import time
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from game import Card, Player, GameState, resolve_card_effect, combat_phase, get_card_pool
from agents import RandomAgent, RolloutAgent, MCTSAgent


def run_game(language, opponent_type, rollout_count, seed):
    random.seed(seed)
    np.random.seed(seed)

    card_pool = get_card_pool(language)
    p1 = Player("Player 1")
    p2 = Player("Player 2")

    deck1 = [Card(cd).clone() for cd in card_pool]
    deck2 = [Card(cd).clone() for cd in card_pool]
    random.shuffle(deck1)
    random.shuffle(deck2)
    p1.deck, p2.deck = deck1, deck2

    for _ in range(3):
        p1.draw()
        p2.draw()

    game_state = GameState(p1, p2)

    p1.agent = RandomAgent("RandomBaseline")
    if opponent_type == "mcts":
        p2.agent = MCTSAgent("MCTSAgent", rollout_count=rollout_count)
    elif opponent_type == "rollout":
        p2.agent = RolloutAgent("RolloutAgent", rollout_count=rollout_count)

    turn_counter = 0
    MAX_TURNS = 50
    while p1.lp > 0 and p2.lp > 0 and turn_counter < MAX_TURNS:
        current = game_state.current_player()
        current.draw()
        if current.hand:
            card = current.agent.choose_move(game_state, current)
            if card:
                current.play_card(card)
                resolve_card_effect(game_state, current, card)
        combat_phase(game_state)
        game_state.next_turn()
        turn_counter += 1

    if p1.lp <= 0:
        return 1  # opponent wins
    elif p2.lp <= 0:
        return 0  # random baseline wins
    return -1  # draw


def main():
    configs = []
    for lang in ["en", "ar"]:
        for opp in ["mcts", "rollout"]:
            for rc in [1, 10, 50]:
                configs.append((lang, opp, rc))

    n_games = 500
    results = {}

    for lang, opp, rc in configs:
        key = f"{lang.upper()} vs {opp}-{rc}"
        wins = 0
        total = 0
        for i in range(n_games):
            winner = run_game(lang, opp, rc, seed=42 + i)
            if winner == 0:
                wins += 1
            total += 1
        win_rate = wins / total * 100
        ci_low = max(0, win_rate - 1.96 * (win_rate * (100 - win_rate) / total) ** 0.5)
        ci_high = min(100, win_rate + 1.96 * (win_rate * (100 - win_rate) / total) ** 0.5)
        results[key] = {"win_rate": win_rate, "ci_low": ci_low, "ci_high": ci_high}
        print(f"{key:25s} | Random wins: {win_rate:5.1f}% [{ci_low:5.1f}%, {ci_high:5.1f}%]")

    # Save
    os.makedirs("results/random_baseline", exist_ok=True)
    with open("results/random_baseline/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/random_baseline/results.json")


if __name__ == "__main__":
    main()
