import sys
import argparse
import json
import logging
import random
import os
import numpy as np
import time
import copy
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
from game import (
    Card,
    Player,
    GameState,
    resolve_card_effect,
    combat_phase,
    get_card_pool,
    SUPPORTED_LANGUAGES
)
from agents import RolloutAgent, MCTSAgent, LLMAgent, RandomAgent
from cfr.abstraction_levels import AbstractionLevel
from utils import compute_confidence_interval
from statistical_analysis import StatisticalAnalyzer, analyze_tcg_results, validate_sample_size
from process_metrics import ProcessMetricsCollector
from multiprocessing import cpu_count

import asyncio
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger("SacraBattle")
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def run_experiment(exp_args: argparse.Namespace, cpu_pool=None):
    """Run all games using asyncio with a ProcessPoolExecutor for rollouts
    and a ThreadPoolExecutor for LLM requests, then summarize."""
    owns_pool = cpu_pool is None
    total_games = exp_args.num_batches * exp_args.games_per_batch
    logger.debug(f"Starting simulation of {total_games} games using async executors.")

    # Build per-game argument dicts
    args_dicts = []
    for batch in range(1, exp_args.num_batches + 1):
        for game in range(1, exp_args.games_per_batch + 1):
            d = vars(exp_args).copy()
            d["seed"] = exp_args.seed + (batch - 1) * exp_args.games_per_batch + game
            d["batch"] = batch
            d["game_index"] = game
            args_dicts.append(d)

    async def run_one_game_async(cfg: dict, cpu_pool: ProcessPoolExecutor) -> dict:
        # Recreate Namespace and game objects
        game_args = argparse.Namespace(**cfg)
        card_pool = get_card_pool(game_args.language)

        # Initialize players & decks
        p1 = Player("Player 1")
        p2 = Player("Player 2")
        if game_args.full_deck:
            deck1 = [Card(cd).clone() for cd in card_pool]
            deck2 = [Card(cd).clone() for cd in card_pool]
            random.shuffle(deck1)
            random.shuffle(deck2)
            p1.deck, p2.deck = deck1, deck2
        else:
            deck_pool = [Card(cd).clone() for cd in card_pool]
            random.shuffle(deck_pool)
            half = len(deck_pool) // 2
            p1.deck = deck_pool[:half]
            p2.deck = [c.clone() for c in p1.deck]
            random.shuffle(p1.deck)
            random.shuffle(p2.deck)

        starting_decks = {
            p1.name: [
                {"id": c.id, "name": c.name, "type": c.card_type} for c in p1.deck
            ],
            p2.name: [
                {"id": c.id, "name": c.name, "type": c.card_type} for c in p2.deck
            ],
        }

        for _ in range(3):
            p1.draw()
            p2.draw()
        game_state = GameState(p1, p2)

        # Rules text based on language
        if game_args.language == "ar":
            rules_text = (
                "قواعد اللعبة:\n"
                "1. يبدأ كل لاعب بـ 10 نقاط حياة.\n"
                "2. يتم بناء السطح من مجموعة البطاقات المشتركة (إما كاملة أو مقسمة).\n"
                "3. كل جولة تتكون من: مرحلة السحب، مرحلة اللعب (لعب بطاقة واحدة)، ومرحلة المعركة.\n"
                "4. في مرحلة المعركة، يمكن للأبطال (الذين ليسوا متأخرين في الاستدعاء) الهجوم.\n"
                "5. الهدف هو تقليل نقاط حياة الخصم من 10 إلى 0."
            )
        else:
            rules_text = (
                "Game Rules:\n"
                "1. Each player starts with 10 Life Points.\n"
                "2. Players build a deck from the common card pool (either full or split).\n"
                "3. Each turn consists of a Draw Phase, a Main Phase (play 1 card), and a Combat Phase.\n"
                "4. In Combat, eligible Champions (those not affected by summoning sickness) may attack.\n"
                "5. The goal is to reduce your opponent's Life Points from 10 to 0."
            )

        include_cards = game_args.add_cards
        llm_agent = LLMAgent(
            "LLMAgent",
            model_name=game_args.model,
            append_cards=game_args.append_cards,
            card_data=(
                [Card(cd).clone() for cd in get_card_pool(game_args.language)]
                if game_args.agent_llm_append_cards
                else []
            ),
            rules_text=rules_text,
            add_rules=game_args.add_rules,
            include_cards=include_cards,
            language=game_args.language,
            llm_type=game_args.llm_type,
            vllm_host=game_args.vllm_host,
            vllm_port=game_args.vllm_port,
            qwen_think=game_args.qwen_think,
            prompt_strategy=getattr(game_args, 'prompt_strategy', 'baseline'),  # Add prompt strategy
            parse_mode=getattr(game_args, 'parse_mode', 'soft'),  # Add parse mode
        )
        if game_args.opponent == "random":
            opponent_agent = RandomAgent("RandomAgent")
        elif game_args.opponent == "rollout":
            opponent_agent = RolloutAgent(
                "RolloutAgent", rollout_count=game_args.rollout_count
            )
        elif game_args.opponent == "mcts":
            opponent_agent = MCTSAgent(
                "MCTSAgent", rollout_count=game_args.rollout_count
            )
        elif game_args.opponent == "cfr":
            if not game_args.cfr_model:
                raise ValueError("--cfr-model must be provided when using 'cfr' opponent")
            from cfr.cfr_agent import CFRAgent
            opponent_agent = CFRAgent(
                name="CFROpponent",
                exploration_epsilon=0.0,
                abstraction_level=AbstractionLevel.HIGH,
                model_path=game_args.cfr_model,
            )
        p1.agent = llm_agent
        p2.agent = opponent_agent
        
        # Initialize process metrics collector
        metrics_collector = ProcessMetricsCollector()
        
        # Play the game
        turn_counter = 0
        MAX_TURNS = 50
        while p1.lp > 0 and p2.lp > 0 and turn_counter < MAX_TURNS:
            current = game_state.current_player()
            game_state.game_log.append(
                f"Turn {game_state.turn_number}: {current.name}'s turn."
            )
            drawn = current.draw()
            if drawn:
                game_state.game_log.append(f"{current.name} draws {drawn.name}.")
            if current.hand:
                if isinstance(current.agent, (RolloutAgent, MCTSAgent)):
                    state_for_mcts = game_state.clone()
                    player_for_mcts = state_for_mcts.players[
                        state_for_mcts.current_player_idx
                    ]

                    card = await asyncio.get_running_loop().run_in_executor(
                        cpu_pool,
                        current.agent.choose_move,
                        state_for_mcts,
                        player_for_mcts,
                    )
                elif isinstance(current.agent, RandomAgent):
                    card = current.agent.choose_move(game_state, current)
                else:
                    # I/O‐bound: LLM call in thread pool
                    start_decision_time = time.time()
                    card = await current.agent.choose_move_async(game_state, current)
                    decision_time = time.time() - start_decision_time
                    metrics_collector.record_decision_time(decision_time, game_state.turn_number)
                    
                if card:
                    # Record move validity (move was valid if we got a card)
                    metrics_collector.record_move_attempt(card.name if card else "No Move", True, game_state)
                    # Record parsing metrics if available
                    if hasattr(current, 'parsing_results') and current.parsing_results:
                        latest_parse = current.parsing_results[-1]
                        metrics_collector.record_parsing_method(latest_parse)
                    current.play_card(card)
                    resolve_card_effect(game_state, current, card)
                else:
                    # Record invalid move attempt
                    metrics_collector.record_move_attempt("Invalid/No Move", False, game_state)
                    # Record parsing failure if available
                    if hasattr(current, 'parsing_results') and current.parsing_results:
                        latest_parse = current.parsing_results[-1]
                        metrics_collector.record_parsing_method(latest_parse)

            if current.board and not current.spell_played:
                combat_phase(game_state)

            if p1.lp <= 0 or p2.lp <= 0:
                break
            game_state.next_turn()
            turn_counter += 1

        # Determine winner
        if p1.lp <= 0 and p2.lp <= 0:
            winner = None
        elif p1.lp > p2.lp:
            winner = 0
        elif p1.lp < p2.lp:
            winner = 1
        else:
            winner = None

        avg_dec_p1 = np.mean(p1.decision_times) if p1.decision_times else 0
        avg_dec_p2 = np.mean(p2.decision_times) if p2.decision_times else 0
        
        # Calculate process metrics summary
        move_validity_rate = np.mean([m["valid"] for m in metrics_collector.metrics["move_validity"]]) if metrics_collector.metrics["move_validity"] else 0.0

        # Calculate parsing metrics
        parsing_stats = metrics_collector.get_parsing_stats()
        
        # Calculate resource efficiency for the winner
        if winner is not None:
            winning_player = p1 if winner == 0 else p2
            resource_efficiency = metrics_collector.calculate_resource_efficiency(game_state, winning_player)
        else:
            resource_efficiency = 0.0

        try:
            return {
                "winner": winner,
                "turns": game_state.turn_number,
                "game_log": game_state.game_log,
                "move_logs": {p1.name: p1.move_log, p2.name: p2.move_log},
                "avg_decision_time": {p1.name: avg_dec_p1, p2.name: avg_dec_p2},
                "token_usage": {p1.name: p1.token_usage, p2.name: p2.token_usage},
                "starting_decks": starting_decks,
                "batch": game_args.batch,
                "process_metrics": {
                    "move_validity_rate": move_validity_rate,
                    "resource_efficiency": resource_efficiency,
                    "decision_times": metrics_collector.metrics["decision_times"],
                    "error_types": dict(metrics_collector.metrics["error_types"]),
                    "parsing_methods": parsing_stats
                }
            }
        finally:
            await llm_agent.client.close()

    async def run_one_game_with_timeout(cfg: dict, cpu_pool: ProcessPoolExecutor) -> dict:
        while True:
            try:
                # 30 minutes = 1800 seconds
                return await asyncio.wait_for(
                    run_one_game_async(cfg, cpu_pool),
                    timeout=2800
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Game (batch={cfg['batch']} game_index={cfg['game_index']}) "
                    "timed out after 30m — dropping."
                )
                return None


    async def run_all_games():
        cpu_pool = ProcessPoolExecutor(max_workers=max(1, cpu_count() - 4))
        tasks = [run_one_game_with_timeout(cfg, cpu_pool) for cfg in args_dicts]
        results = []
        for future in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"Running games for {exp_args.model}",
            unit="game",
        ):
            results.append(await future)

        if owns_pool:
            cpu_pool.shutdown()
        return results

    results = asyncio.run(run_all_games())
    results = [r for r in results if r is not None]
    completed = len(results)
    dropped = total_games - completed

    turns_list = [res["turns"] for res in results]
    win_list = [res["winner"] for res in results if res["winner"] is not None]

    decision_times_p1 = []
    decision_times_p2 = []
    for res in results:
        d = res.get("avg_decision_time", {})
        decision_times_p1.append(d.get("Player 1", 0))
        decision_times_p2.append(d.get("Player 2", 0))

    token_usage_p1 = []
    token_usage_p2 = []
    process_metrics_all = []
    for res in results:
        t = res.get("token_usage", {})
        token_usage_p1 += t.get("Player 1", [])
        token_usage_p2 += t.get("Player 2", [])
        if "process_metrics" in res:
            process_metrics_all.append(res["process_metrics"])

    # Initialize statistical analyzer for bootstrap CIs and effect sizes
    analyzer = StatisticalAnalyzer(confidence_level=0.95, n_bootstrap=10000)
    
    # Token usage analysis with bootstrap CI
    mean_tok_p1, ci_lower_tok_p1, ci_upper_tok_p1 = analyzer.bootstrap_confidence_interval(
        np.array(token_usage_p1) if token_usage_p1 else np.array([0])
    )
    mean_tok_p2, ci_lower_tok_p2, ci_upper_tok_p2 = analyzer.bootstrap_confidence_interval(
        np.array(token_usage_p2) if token_usage_p2 else np.array([0])
    )
    
    # Turn analysis with bootstrap CI
    mean_turns, ci_lower_turns, ci_upper_turns = analyzer.bootstrap_confidence_interval(
        np.array(turns_list)
    )
    median_turns = float(np.median(turns_list))
    std_turns = float(np.std(turns_list, ddof=1)) if len(turns_list) > 1 else 0.0
    min_turns = min(turns_list)
    max_turns = max(turns_list)

    # Win rate analysis
    wins_agent1 = win_list.count(0)
    wins_agent2 = win_list.count(1)
    draws = completed - len(win_list)
    
    # Bootstrap CI for win rates
    win_outcomes_agent1 = [1 if w == 0 else 0 for w in win_list]
    win_outcomes_agent2 = [1 if w == 1 else 0 for w in win_list]
    
    if win_outcomes_agent1:
        win_rate_agent1, ci_lower_wr1, ci_upper_wr1 = analyzer.bootstrap_confidence_interval(
            np.array(win_outcomes_agent1)
        )
        win_rate_agent1 *= 100
        ci_lower_wr1 *= 100
        ci_upper_wr1 *= 100
    else:
        win_rate_agent1 = ci_lower_wr1 = ci_upper_wr1 = 0
    
    if win_outcomes_agent2:
        win_rate_agent2, ci_lower_wr2, ci_upper_wr2 = analyzer.bootstrap_confidence_interval(
            np.array(win_outcomes_agent2)
        )
        win_rate_agent2 *= 100
        ci_lower_wr2 *= 100
        ci_upper_wr2 *= 100
    else:
        win_rate_agent2 = ci_lower_wr2 = ci_upper_wr2 = 0

    # Decision time analysis with bootstrap CI
    mean_dec_time_p1, ci_lower_dec_p1, ci_upper_dec_p1 = analyzer.bootstrap_confidence_interval(
        np.array(decision_times_p1) if decision_times_p1 else np.array([0])
    )
    median_dec_p1 = float(np.median(decision_times_p1))
    std_dec_p1 = (
        float(np.std(decision_times_p1, ddof=1)) if len(decision_times_p1) > 1 else 0.0
    )
    min_dec_p1 = min(decision_times_p1) if decision_times_p1 else 0
    max_dec_p1 = max(decision_times_p1) if decision_times_p1 else 0

    mean_dec_time_p2, ci_lower_dec_p2, ci_upper_dec_p2 = analyzer.bootstrap_confidence_interval(
        np.array(decision_times_p2) if decision_times_p2 else np.array([0])
    )
    median_dec_p2 = float(np.median(decision_times_p2))
    std_dec_p2 = (
        float(np.std(decision_times_p2, ddof=1)) if len(decision_times_p2) > 1 else 0.0
    )
    min_dec_p2 = min(decision_times_p2) if decision_times_p2 else 0
    max_dec_p2 = max(decision_times_p2) if decision_times_p2 else 0
    
    # Calculate effect sizes if comparing two agents
    effect_size = None
    if win_outcomes_agent1 and win_outcomes_agent2:
        effect_size = analyzer.cohen_d(
            np.array(win_outcomes_agent1),
            np.array(win_outcomes_agent2)
        )
        effect_interpretation = analyzer.interpret_cohen_d(effect_size)

    logger.debug(f"Average turns per game: {mean_turns:.2f} [{ci_lower_turns:.2f}, {ci_upper_turns:.2f}]")
    logger.debug(f"Win Rate - Agent1: {win_rate_agent1:.2f}% [{ci_lower_wr1:.2f}%, {ci_upper_wr1:.2f}%]")
    logger.debug(f"Win Rate - Agent2: {win_rate_agent2:.2f}% [{ci_lower_wr2:.2f}%, {ci_upper_wr2:.2f}%]")
    if effect_size is not None:
        logger.debug(f"Effect size (Cohen's d): {effect_size:.3f} ({effect_interpretation})")
    logger.debug(
        f"Agent1 avg decision time: {mean_dec_time_p1:.4f} [{ci_lower_dec_p1:.4f}, {ci_upper_dec_p1:.4f}] sec"
    )
    logger.debug(
        f"Agent2 avg decision time: {mean_dec_time_p2:.4f} [{ci_lower_dec_p2:.4f}, {ci_upper_dec_p2:.4f}] sec"
    )

    agent_label_1 = "LLMAgent"
    agent_label_2 = "RandomAgent" if exp_args.opponent == "random" else "MCTSAgent"

    # Perform sample size validation
    sample_size_validation = validate_sample_size(
        effect_size=0.5,  # Medium effect size
        desired_power=0.8,
        current_sample_size=completed
    )
    
    metrics = {
        "total_games": total_games,
        "completed_games": completed,
        "dropped_games": dropped,
        "statistical_analysis": {
            "confidence_level": 0.95,
            "bootstrap_iterations": 10000,
            "sample_size_validation": sample_size_validation
        },
        "turns": {
            "mean": mean_turns,
            "ci_lower": ci_lower_turns,
            "ci_upper": ci_upper_turns,
            "median": median_turns,
            "std": std_turns,
            "min": min_turns,
            "max": max_turns,
        },
        "wins": {
            agent_label_1: wins_agent1,
            agent_label_2: wins_agent2,
            "draws": draws,
        },
        "win_rate": {
            agent_label_1: {
                "rate": win_rate_agent1,
                "ci_lower": ci_lower_wr1,
                "ci_upper": ci_upper_wr1
            },
            agent_label_2: {
                "rate": win_rate_agent2,
                "ci_lower": ci_lower_wr2,
                "ci_upper": ci_upper_wr2
            },
        },
        "effect_size": {
            "cohen_d": effect_size if effect_size is not None else None,
            "interpretation": effect_interpretation if effect_size is not None else None
        },
        "decision_time": {
            agent_label_1: {
                "mean": mean_dec_time_p1,
                "ci_lower": ci_lower_dec_p1,
                "ci_upper": ci_upper_dec_p1,
                "median": median_dec_p1,
                "std": std_dec_p1,
                "min": min_dec_p1,
                "max": max_dec_p1,
            },
            agent_label_2: {
                "mean": mean_dec_time_p2,
                "ci_lower": ci_lower_dec_p2,
                "ci_upper": ci_upper_dec_p2,
                "median": median_dec_p2,
                "std": std_dec_p2,
                "min": min_dec_p2,
                "max": max_dec_p2,
            },
        },
        "token_usage": {
            agent_label_1: {
                "mean": mean_tok_p1,
                "ci_lower": ci_lower_tok_p1,
                "ci_upper": ci_upper_tok_p1
            },
            agent_label_2: {
                "mean": mean_tok_p2,
                "ci_lower": ci_lower_tok_p2,
                "ci_upper": ci_upper_tok_p2
            },
        },
    }
    
    # Add process metrics if collected
    if process_metrics_all:
        # Aggregate process metrics
        all_validity_rates = [pm["move_validity_rate"] for pm in process_metrics_all]
        all_efficiency_scores = [pm["resource_efficiency"] for pm in process_metrics_all]
        
        # Calculate bootstrap CIs for process metrics
        mean_validity, ci_lower_validity, ci_upper_validity = analyzer.bootstrap_confidence_interval(
            np.array(all_validity_rates)
        )
        mean_efficiency, ci_lower_efficiency, ci_upper_efficiency = analyzer.bootstrap_confidence_interval(
            np.array(all_efficiency_scores)
        )
        
        metrics["process_metrics"] = {
            "move_validity": {
                "mean": mean_validity,
                "ci_lower": ci_lower_validity,
                "ci_upper": ci_upper_validity
            },
            "resource_efficiency": {
                "mean": mean_efficiency,
                "ci_lower": ci_lower_efficiency,
                "ci_upper": ci_upper_efficiency
            }
        }

    # Print parsing metrics summary if available
    if "process_metrics" in metrics and "parsing_methods" in metrics["process_metrics"]:
        parsing_data = metrics["process_metrics"]["parsing_methods"]
        if parsing_data.get("summary"):
            print("\n📊 Parsing Method Statistics:")
            print("  Method Usage:")
            for method, data in parsing_data["summary"].items():
                count = data["count"]
                success_rate = parsing_data["success_rates"].get(method, 0)
                print(f"    - {method}: {count} times ({success_rate:.1%} success)")

    model_name_sanitized = exp_args.model.replace("/", "_")
    results_dir = f"results/{model_name_sanitized}/{exp_args.language}{'_qwen_think' if exp_args.qwen_think else ''}_rollout{exp_args.rollout_count}_{int(time.time())}"
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "benchmark_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(vars(exp_args), f, indent=4)
    with open(
        os.path.join(results_dir, "detailed_game_logs.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    logger.info(f"Benchmark results saved to {results_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run the Sacra Battle benchmark game.")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models to run in parallel (overrides --model)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model to use for LLM agent",
    )
    parser.add_argument(
        "--llm_type",
        type=str,
        choices=["openrouter", "vllm"],
        default="openrouter",
        help="Select LLM endpoint type: OpenRouter or vLLM",
    )
    parser.add_argument(
        "--vllm_host",
        type=str,
        default="localhost",
        help="Hostname for vLLM OpenAI-compatible server",
    )
    parser.add_argument(
        "--vllm_port",
        type=int,
        default=8000,
        help="Port for vLLM OpenAI-compatible server",
    )
    parser.add_argument(
        "--qwen_think",
        action="store_true",
        help="Enables Qwen3 to think",
    )
    parser.add_argument(
        "--append_cards",
        action="store_true",
        help="(Deprecated) Append card details to prompt",
    )
    parser.add_argument(
        "--agent_llm_append_cards",
        action="store_true",
        help="LLMAgent gets full card pool data",
    )
    parser.add_argument(
        "--add_rules",
        action="store_true",
        help="Append detailed game rules and full card details to LLM prompt",
    )
    parser.add_argument(
        "--add_cards",
        action="store_true",
        help="Include all cards in the first prompt",
    )
    parser.add_argument(
        "--num_batches", type=int, default=1, help="Number of batches to run"
    )
    parser.add_argument(
        "--games_per_batch", type=int, default=1, help="Number of games per batch"
    )
    parser.add_argument(
        "--full_deck",
        action="store_true",
        help="Each agent gets the full card pool (shuffled independently)",
    )
    parser.add_argument(
        "--rollout_count", type=int, default=10, help="Number of rollouts for MCTSAgent"
    )
    parser.add_argument(
        "--prompt_strategy",
        type=str,
        default="baseline",
        choices=["baseline", "minimal", "chain_of_thought"],
        help="Prompt strategy for LLM agent: baseline, minimal, or chain_of_thought"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=SUPPORTED_LANGUAGES,
        default="en",
        help=f"Language for the game. Supported languages: {', '.join(SUPPORTED_LANGUAGES)}",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        choices=["random", "rollout", "mcts", "cfr"],
        default="mcts",
        help="Opponent type: 'random' for RandomAgent, 'rollout' for RolloutAgent, 'mcts' for MCTSAgent, 'cfr' for a pre‑trained CFRAgent from a pickle file",
    )
    parser.add_argument(
        "--cfr-model",
        type=str,
        default=None,
        help="Path to a pickle file containing a pre‑trained CFRAgent when using the 'cfr' opponent type",
    )
    parser.add_argument(
        "--parse_mode",
        type=str,
        choices=["strict", "soft"],
        default="soft",
        help="LLM move parsing mode: 'strict' (tags only) or 'soft' (search in text)"
    )
    parser.add_argument("--batch", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--game_index", type=int, default=1, help=argparse.SUPPRESS)

    args = parser.parse_args()

    numeric_level = getattr(logging, args.log_level.upper(), None)
    if isinstance(numeric_level, int):
        logger.setLevel(numeric_level)
        stream_handler.setLevel(numeric_level)
        ts = int(time.time())
    log_file = os.path.join("results", f"{ts}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logger.level)
    logger.addHandler(file_handler)

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.models:
        model_list = [m.strip() for m in args.models.split(",") if m.strip()]
        if not model_list:
            logger.error("No models provided to --models")
            sys.exit(1)

        # One big pool for everybody
        shared_workers = max(1, cpu_count() - 2)  # leave a couple of cores free
        shared_pool = ProcessPoolExecutor(max_workers=shared_workers)
        logger.info(
            f"Created shared ProcessPool with {shared_workers} workers "
            f"for {len(model_list)} models."
        )

        try:
            for model_name in model_list:
                p_args = copy.deepcopy(args)
                p_args.model = model_name
                # num_workers is irrelevant when we pass a pool
                run_experiment(p_args, cpu_pool=shared_pool)
        finally:
            shared_pool.shutdown()
        return

    # Single model run
    run_experiment(args)


if __name__ == "__main__":
    main()
