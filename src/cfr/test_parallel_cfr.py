#!/usr/bin/env python3
"""
Test suite for parallel CFR implementation.

Tests:
1. Parallel trainer initialization
2. Worker function correctness
3. Table aggregation
4. Speedup verification
"""

import unittest
import tempfile
import time
import os
import sys
from collections import defaultdict

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfr_agent import CFRAgent
from parallel_cfr_trainer import ParallelCFRTrainer, AsyncParallelCFRTrainer, worker_train_batch
from information_set import InfoSet


class TestParallelCFR(unittest.TestCase):
    """Test parallel CFR implementation."""

    def test_worker_function(self):
        """Test that worker function runs and returns valid results."""
        # Run worker for small number of iterations
        regret_sum, strategy_sum, iterations = worker_train_batch(
            worker_id=0,
            iterations=10,
            card_pool_type="community",
            language="en",
            max_game_length=10,
            use_mccfr=False,
            seed=42
        )

        # Check results
        self.assertEqual(iterations, 10)
        self.assertIsInstance(regret_sum, dict)
        self.assertIsInstance(strategy_sum, dict)

    def test_parallel_trainer_init(self):
        """Test parallel trainer initialization."""
        agent = CFRAgent("TestAgent")
        trainer = ParallelCFRTrainer(
            agent,
            num_workers=2,
            sync_interval=100
        )

        self.assertEqual(trainer.num_workers, 2)
        self.assertEqual(trainer.sync_interval, 100)
        self.assertIs(trainer.agent, agent)

    def test_table_aggregation(self):
        """Test regret and strategy table aggregation."""
        agent = CFRAgent("TestAgent")
        trainer = ParallelCFRTrainer(agent, num_workers=2)

        # Create mock worker results
        info_set = InfoSet(
            phase="early",
            life_diff=0,
            hand_strength="medium",
            board_control="neutral",
            tricks_active=0,
            hand_composition=(1, 1, 0)
        )

        worker1_regrets = {info_set: {"action1": 10.0, "action2": -5.0}}
        worker1_strategy = {info_set: {"action1": 0.7, "action2": 0.3}}

        worker2_regrets = {info_set: {"action1": 5.0, "action2": -10.0}}
        worker2_strategy = {info_set: {"action1": 0.6, "action2": 0.4}}

        # Aggregate
        trainer._aggregate_tables(worker1_regrets, worker1_strategy)
        trainer._aggregate_tables(worker2_regrets, worker2_strategy)

        # Check aggregation (averaged regrets, summed strategies)
        expected_regret_action1 = (10.0 + 5.0) / 2  # Average
        expected_regret_action2 = (-5.0 + -10.0) / 2  # Average

        self.assertAlmostEqual(
            agent.regret_sum[info_set]["action1"],
            expected_regret_action1,
            places=5
        )
        self.assertAlmostEqual(
            agent.regret_sum[info_set]["action2"],
            expected_regret_action2,
            places=5
        )

        # Strategies should be summed
        self.assertAlmostEqual(
            agent.strategy_sum[info_set]["action1"],
            0.7 + 0.6,
            places=5
        )
        self.assertAlmostEqual(
            agent.strategy_sum[info_set]["action2"],
            0.3 + 0.4,
            places=5
        )

    def test_parallel_training_small(self):
        """Test small parallel training run."""
        agent = CFRAgent("TestAgent")
        trainer = ParallelCFRTrainer(
            agent,
            num_workers=2,
            sync_interval=10
        )

        # Train for small number of iterations
        trainer.train(20)

        # Check that training occurred
        self.assertEqual(agent.iteration_count, 20)
        self.assertGreater(agent.get_info_set_count(), 0)

    def test_speedup_comparison(self):
        """Compare sequential vs parallel training speed."""
        from cfr_trainer import CFRTrainer

        # Sequential training
        agent_seq = CFRAgent("Sequential")
        trainer_seq = CFRTrainer(agent_seq, max_game_length=10)

        start_time = time.time()
        trainer_seq.train(50)
        seq_time = time.time() - start_time
        seq_iterations = agent_seq.iteration_count

        # Parallel training with 2 workers
        agent_par = CFRAgent("Parallel")
        trainer_par = ParallelCFRTrainer(
            agent_par,
            num_workers=2,
            sync_interval=25,
            max_game_length=10
        )

        start_time = time.time()
        trainer_par.train(50)
        par_time = time.time() - start_time
        par_iterations = agent_par.iteration_count

        # Check that parallel is faster (allowing for overhead)
        speedup = seq_time / par_time
        print(f"Sequential time: {seq_time:.3f}s")
        print(f"Parallel time: {par_time:.3f}s")
        print(f"Speedup: {speedup:.2f}x")

        # Should be some speedup even with overhead
        self.assertGreater(speedup, 1.2)  # At least 20% faster

        # Should complete same number of iterations
        self.assertEqual(seq_iterations, par_iterations)

    def test_async_parallel_trainer(self):
        """Test asynchronous parallel trainer."""
        agent = CFRAgent("AsyncAgent")
        trainer = AsyncParallelCFRTrainer(
            agent,
            num_workers=2,
            sync_interval=10
        )

        # Train for small number
        trainer.train(20)

        # Check results
        self.assertGreater(agent.iteration_count, 0)
        self.assertGreater(agent.get_info_set_count(), 0)

    def test_deterministic_seeding(self):
        """Test that workers with same seed produce same results."""
        # Run same worker twice with same seed
        regret1, strategy1, _ = worker_train_batch(
            worker_id=0,
            iterations=10,
            card_pool_type="community",
            language="en",
            max_game_length=10,
            use_mccfr=False,
            seed=42
        )

        regret2, strategy2, _ = worker_train_batch(
            worker_id=0,
            iterations=10,
            card_pool_type="community",
            language="en",
            max_game_length=10,
            use_mccfr=False,
            seed=42
        )

        # Results should be identical
        self.assertEqual(set(regret1.keys()), set(regret2.keys()))
        for key in regret1:
            if key in regret2:
                self.assertEqual(regret1[key], regret2[key])

    def test_mccfr_parallel(self):
        """Test parallel training with Monte Carlo CFR."""
        agent = CFRAgent("MCCFRAgent")
        trainer = ParallelCFRTrainer(
            agent,
            num_workers=2,
            sync_interval=10,
            use_mccfr=True  # Use MCCFR
        )

        trainer.train(20)

        # Should complete successfully
        self.assertEqual(agent.iteration_count, 20)
        self.assertGreater(agent.get_info_set_count(), 0)


class TestParallelPerformance(unittest.TestCase):
    """Performance tests for parallel CFR."""

    def test_scaling(self):
        """Test scaling with different worker counts."""
        iterations = 100
        results = {}

        for num_workers in [1, 2, 4]:
            agent = CFRAgent(f"Agent_{num_workers}_workers")
            trainer = ParallelCFRTrainer(
                agent,
                num_workers=num_workers,
                sync_interval=50,
                max_game_length=10
            )

            start_time = time.time()
            trainer.train(iterations)
            elapsed = time.time() - start_time

            results[num_workers] = {
                'time': elapsed,
                'iterations': agent.iteration_count,
                'rate': agent.iteration_count / elapsed
            }

            print(f"{num_workers} workers: {elapsed:.3f}s, "
                  f"{results[num_workers]['rate']:.1f} iters/sec")

        # Check that more workers = faster
        self.assertLess(results[2]['time'], results[1]['time'])
        if 4 in results:
            self.assertLess(results[4]['time'], results[2]['time'])

    def test_memory_usage(self):
        """Test that memory usage is reasonable."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run parallel training
        agent = CFRAgent("MemoryTest")
        trainer = ParallelCFRTrainer(
            agent,
            num_workers=2,
            sync_interval=50
        )
        trainer.train(100)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(f"Memory increase: {memory_increase:.1f} MB")

        # Should not use excessive memory (less than 500MB increase)
        self.assertLess(memory_increase, 500)


if __name__ == "__main__":
    # Run tests
    unittest.main()