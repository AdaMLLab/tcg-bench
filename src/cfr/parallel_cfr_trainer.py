"""
Parallel CFR training implementation for Sacra Battle.

Uses data parallelism with periodic synchronization for near-linear speedup.
Workers run independent CFR iterations and periodically merge regret tables.
"""

import logging
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Tuple, Optional
import multiprocessing as mp
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import GameState, get_card_pool
from cfr_agent import CFRAgent
from cfr_trainer import CFRTrainer, MCCFRTrainer
from information_set import InfoSet
from abstraction_levels import AbstractionLevel

logger = logging.getLogger("SacraBattle.ParallelCFR")


def worker_train_batch(
    worker_id: int,
    iterations: int,
    card_pool_type: str,
    language: str,
    max_game_length: int,
    use_mccfr: bool,
    abstraction_level: str,  # Add abstraction level parameter
    seed: Optional[int] = None
) -> Tuple[Dict, Dict, int]:
    """
    Worker function that runs CFR iterations independently.

    Args:
        worker_id: Unique worker identifier
        iterations: Number of iterations to run
        card_pool_type: Unused, kept for compatibility
        language: "en" or "ar"
        max_game_length: Maximum game length
        use_mccfr: Whether to use Monte Carlo CFR
        abstraction_level: Abstraction level as string ("high", "medium", "low")
        seed: Random seed for reproducibility

    Returns:
        Tuple of (regret_sum, strategy_sum, iterations_completed)
    """
    import random

    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed + worker_id)

    # Convert abstraction level string to enum
    level = AbstractionLevel(abstraction_level)

    # Create local agent with empty tables and specified abstraction level
    local_agent = CFRAgent(name=f"Worker_{worker_id}", abstraction_level=level)

    # Create trainer
    if use_mccfr:
        trainer = MCCFRTrainer(
            local_agent,
            language=language,
            max_game_length=max_game_length
        )
    else:
        trainer = CFRTrainer(
            local_agent,
            language=language,
            max_game_length=max_game_length
        )

    # Run iterations
    logger.debug(f"Worker {worker_id} starting {iterations} iterations")

    for i in range(iterations):
        # Generate random game state
        game_state = trainer._create_random_game_state()

        # Run CFR iteration
        trainer._cfr(game_state, 1.0, 1.0)

        # Update iteration count
        local_agent.iteration_count += 1

    logger.debug(f"Worker {worker_id} completed {iterations} iterations")

    # Convert defaultdicts to regular dicts for pickling
    regret_sum = dict(local_agent.regret_sum)
    strategy_sum = dict(local_agent.strategy_sum)

    return regret_sum, strategy_sum, iterations


class ParallelCFRTrainer:
    """
    Parallel CFR trainer using data parallelism.

    Multiple workers run CFR iterations independently and periodically
    synchronize their regret tables for convergence.
    """

    def __init__(
        self,
        cfr_agent: CFRAgent,
        language: str = "en",
        max_game_length: int = 50,
        num_workers: Optional[int] = None,
        sync_interval: int = 250,
        use_mccfr: bool = False,
        **kwargs
    ):
        """
        Initialize parallel CFR trainer.

        Args:
            cfr_agent: The main CFR agent to train
            language: "en" or "ar"
            max_game_length: Maximum turns before draw
            num_workers: Number of parallel workers (default: CPU count)
            sync_interval: Iterations between synchronizations
            use_mccfr: Whether to use Monte Carlo CFR
        """
        self.agent = cfr_agent
        self.card_pool_type = "community"
        self.language = language
        self.max_game_length = max_game_length
        self.num_workers = num_workers or mp.cpu_count()
        self.sync_interval = sync_interval
        self.use_mccfr = use_mccfr
        # Store abstraction level from the agent
        self.abstraction_level = cfr_agent.abstraction_level.value

        logger.info(
            f"Initialized parallel trainer with {self.num_workers} workers, "
            f"sync interval {self.sync_interval}, abstraction: {self.abstraction_level}"
        )

    def train(self, iterations: int):
        """
        Train CFR agent using parallel workers.

        Args:
            iterations: Total number of iterations to run
        """
        logger.info(
            f"Starting parallel CFR training for {iterations} iterations "
            f"using {self.num_workers} workers"
        )

        # Calculate batches
        num_syncs = max(1, iterations // self.sync_interval)
        iterations_per_sync = iterations // num_syncs
        iterations_per_worker = iterations_per_sync // self.num_workers

        logger.info(
            f"Training plan: {num_syncs} sync points, "
            f"{iterations_per_worker} iterations per worker per sync"
        )

        # Track total iterations
        total_iterations = 0

        # Process pool for parallel execution
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:

            for sync_round in range(num_syncs):
                logger.info(
                    f"Sync round {sync_round + 1}/{num_syncs} "
                    f"(iterations {total_iterations}-{total_iterations + iterations_per_sync})"
                )

                # Submit tasks to workers
                futures = []
                for worker_id in range(self.num_workers):
                    # Last worker might get extra iterations due to rounding
                    if worker_id == self.num_workers - 1:
                        worker_iterations = iterations_per_sync - (iterations_per_worker * (self.num_workers - 1))
                    else:
                        worker_iterations = iterations_per_worker

                    future = executor.submit(
                        worker_train_batch,
                        worker_id,
                        worker_iterations,
                        self.card_pool_type,
                        self.language,
                        self.max_game_length,
                        self.use_mccfr,
                        self.abstraction_level,  # Pass abstraction level
                        seed=sync_round * 1000 + worker_id  # Reproducible seeds
                    )
                    futures.append(future)

                # Collect results and aggregate
                for future in as_completed(futures):
                    try:
                        regret_sum, strategy_sum, completed_iterations = future.result()

                        # Aggregate regret sums
                        self._aggregate_tables(regret_sum, strategy_sum)

                        total_iterations += completed_iterations

                    except Exception as e:
                        logger.error(f"Worker failed with error: {e}")
                        raise

                # Update agent iteration count
                self.agent.iteration_count = total_iterations

                # Log progress
                # if (sync_round + 1) % max(1, num_syncs // 10) == 0:
                logger.info(
                    f"Progress: {total_iterations}/{iterations} iterations - "
                    f"Info sets: {self.agent.get_info_set_count()}, "
                    f"Exploitability: {self.agent.get_exploitability():.6f}"
                )

        logger.info(
            f"Parallel training complete. Final stats - "
            f"Total iterations: {total_iterations}, "
            f"Info sets: {self.agent.get_info_set_count()}, "
            f"Exploitability: {self.agent.get_exploitability():.6f}"
        )

    def _aggregate_tables(
        self,
        worker_regret_sum: Dict,
        worker_strategy_sum: Dict
    ):
        """
        Aggregate worker tables into main agent.

        Uses averaging for regrets and summing for strategies.
        """
        # Aggregate regret sums (average across workers)
        for info_set_key, action_regrets in worker_regret_sum.items():
            # Convert dict key back to InfoSet if needed
            if not isinstance(info_set_key, InfoSet):
                info_set = info_set_key  # Already correct type
            else:
                info_set = info_set_key

            # Convert inner dict if needed
            if not isinstance(action_regrets, dict):
                action_regrets = dict(action_regrets)

            for action, regret in action_regrets.items():
                # Average regrets across workers
                self.agent.regret_sum[info_set][action] += regret / self.num_workers

        # Aggregate strategy sums (sum across workers)
        for info_set_key, action_strategies in worker_strategy_sum.items():
            # Convert dict key back to InfoSet if needed
            if not isinstance(info_set_key, InfoSet):
                info_set = info_set_key
            else:
                info_set = info_set_key

            # Convert inner dict if needed
            if not isinstance(action_strategies, dict):
                action_strategies = dict(action_strategies)

            for action, strategy_sum in action_strategies.items():
                self.agent.strategy_sum[info_set][action] += strategy_sum


class AsyncParallelCFRTrainer(ParallelCFRTrainer):
    """
    Asynchronous parallel CFR trainer.

    Workers continuously train and send updates without waiting for sync points.
    More complex but can be more efficient for heterogeneous hardware.
    """

    def train(self, iterations: int):
        """
        Train using asynchronous updates.

        Workers send updates as soon as they complete batches.
        """
        logger.info(
            f"Starting async parallel CFR training for {iterations} iterations "
            f"using {self.num_workers} workers"
        )

        batch_size = min(100, iterations // (self.num_workers * 10))
        total_batches = iterations // batch_size
        batches_per_worker = total_batches // self.num_workers

        completed_iterations = 0

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks upfront
            futures = []

            for worker_id in range(self.num_workers):
                for batch_id in range(batches_per_worker):
                    future = executor.submit(
                        worker_train_batch,
                        worker_id,
                        batch_size,
                        self.card_pool_type,
                        self.language,
                        self.max_game_length,
                        self.use_mccfr,
                        self.abstraction_level,  # Pass abstraction level
                        seed=worker_id * 10000 + batch_id
                    )
                    futures.append(future)

            # Process results as they complete (async)
            for i, future in enumerate(as_completed(futures)):
                try:
                    regret_sum, strategy_sum, batch_iterations = future.result()

                    # Immediate aggregation
                    self._aggregate_tables(regret_sum, strategy_sum)

                    completed_iterations += batch_iterations
                    self.agent.iteration_count = completed_iterations

                    # Progress logging
                    if i % max(1, len(futures) // 20) == 0:
                        logger.info(
                            f"Async progress: {completed_iterations}/{iterations} iterations - "
                            f"Info sets: {self.agent.get_info_set_count()}"
                        )

                except Exception as e:
                    logger.error(f"Async worker failed: {e}")
                    raise

        logger.info(
            f"Async training complete. "
            f"Total iterations: {completed_iterations}, "
            f"Info sets: {self.agent.get_info_set_count()}"
        )