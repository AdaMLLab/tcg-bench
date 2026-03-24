#!/usr/bin/env python3
"""Minimal test harness to pit a pickled CFR model against the built‑in
`MCTSAgent`.

The script performs a single game of Sacra Battle, logging each turn and the
winner.  It is intentionally lightweight – no external dependencies beyond the
project's own modules.

Usage:

    python test_cfr_vs_mcts.py --model_path path/to/cfr_model.pkl

The model path must point to a pickle file created by
``train_cfr.py``.  If the file contains a model trained with a different
``abstraction_level`` you may override it by passing ``--abstraction``.
"""

import argparse
import logging
import random
import sys

from game import (
    Card,
    GameState,
    Player,
    combat_phase,
    get_card_pool,
    resolve_card_effect,
)
from cfr.cfr_agent import CFRAgent
from cfr.abstraction_levels import AbstractionLevel
from agents import MCTSAgent

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

DEFAULT_ABSTRACTION = "high"  # default training value


def build_game(
    language: str = "en",
    card_pool: str = "community",
    model_path: str = "cfr_model.pkl",
    abstraction_level: str = DEFAULT_ABSTRACTION,
):
    """Create a :class:`GameState` with a pre‑trained CFR player vs a MCTS one."""
    pool = get_card_pool(language)

    # full decks – simplest for deterministic tests
    deck1 = [Card(cd).clone() for cd in pool]
    deck2 = [Card(cd).clone() for cd in pool]
    random.shuffle(deck1)
    random.shuffle(deck2)

    p1 = Player("CFR")
    p2 = Player("MCTS")
    p1.deck, p2.deck = deck1, deck2
    for _ in range(3):  # draw starting hand
        p1.draw()
        p2.draw()

    game_state = GameState(p1, p2)

    # Attach agents
    # Convert abstraction string to enum; default value is "high"
    abstraction_enum = AbstractionLevel(abstraction_level)
    p1.agent = CFRAgent(
        name="CFR",
        exploration_epsilon=0.0,
        abstraction_level=abstraction_enum,
        model_path=model_path,
    )
    p2.agent = MCTSAgent(name="MCTS", rollout_count=20)  # adjust if needed
    return game_state


def play_one_game(game_state: GameState, max_turns: int = 50):
    """Run a single game, logging decisions and the final winner."""
    while (
        game_state.players[0].lp > 0
        and game_state.players[1].lp > 0
        and game_state.turn_number < max_turns
    ):
        current = game_state.current_player()
        logging.info(
            f"Turn {game_state.turn_number}: {current.name}'s turn – {current.lp} LP"
        )

        # Draw a card
        current.draw()
        logging.debug(f"{current.name} draws {current.hand[-1].name if current.hand else 'none'}")

        # Choose move
        card = current.agent.choose_move(game_state, current)
        if card:
            logging.info(f"{current.name} plays {card.name}")
            resolve_card_effect(game_state, current, card)
        else:
            logging.info(f"{current.name} passes")

        combat_phase(game_state)
        # Advance to next player/turn after combat
        game_state.next_turn()

    winner = "CFR" if game_state.players[0].lp > 0 else "MCTS"
    logging.info(f"Game over – winner: {winner}")
    return winner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_path",
        type=str,
        default="cfr_model.pkl",
        help="Path to the pickled CFR model",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language of the game (en/ar) – must match model language",
    )
    parser.add_argument(
        "--abstraction",
        type=str,
        choices=["high", "medium", "low"],
        default=DEFAULT_ABSTRACTION,
        help="Abstraction level for CFR (high/medium/low). Defaults to model level if omitted.",
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=1,
        help="Number of games to play (for statistics)",
    )
    args = parser.parse_args()

    try:
        wins = {"CFR": 0, "MCTS": 0}
        for i in range(args.num_games):
            gs = build_game(args.language, args.card_pool, args.model_path, args.abstraction)
            winner = play_one_game(gs)
            wins[winner] += 1
            logging.info(f"Game {i+1}/{args.num_games} winner: {winner}")
        # Print average win stats
        logging.info(f"Total games: {args.num_games}")
        logging.info(f"CFR wins: {wins['CFR']} ({wins['CFR']/args.num_games*100:.1f}%)")
        logging.info(f"MCTS wins: {wins['MCTS']} ({wins['MCTS']/args.num_games*100:.1f}%)")
    except Exception as e:
        logging.exception("Test failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
