#!/usr/bin/env python3
"""
Training script for CFR agent.

Usage:
    python train_cfr.py --iterations 100000 --output models/cfr_model.pkl
    python train_cfr.py --iterations 1000000 --output models/cfr_model.pkl --parallel --workers 8
"""

import argparse
import logging
import os
import sys
import time

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfr_agent import CFRAgent
from cfr_trainer import CFRTrainer, MCCFRTrainer
from parallel_cfr_trainer import ParallelCFRTrainer, AsyncParallelCFRTrainer
from abstraction_levels import AbstractionLevel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CFR.Train")


def main():
    parser = argparse.ArgumentParser(description="Train CFR agent for Sacra Battle")

    parser.add_argument(
        "--iterations",
        type=int,
        default=100000,
        help="Number of training iterations (default: 100000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cfr_model.pkl",
        help="Output path for trained model (default: cfr_model.pkl)"
    )
    parser.add_argument(
        "--algorithm",
        choices=["vanilla", "mccfr"],
        default="vanilla",
        help="CFR algorithm variant (default: vanilla)"
    )
    parser.add_argument(
        "--language",
        choices=["en", "ar"],
        default="en",
        help="Language for cards (default: en)"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10000,
        help="Save checkpoint every N iterations (default: 10000)"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        help="Load model from checkpoint to continue training"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel training (default: False)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)"
    )
    parser.add_argument(
        "--sync-interval",
        type=int,
        default=1000,
        help="Iterations between synchronizations in parallel mode (default: 1000)"
    )
    parser.add_argument(
        "--async-mode",
        action="store_true",
        help="Use asynchronous parallel training (default: False)"
    )
    parser.add_argument(
        "--abstraction",
        choices=["high", "medium", "low"],
        default="high",
        help="Abstraction level for state space (default: high)"
    )

    args = parser.parse_args()

    # Convert abstraction level string to enum
    abstraction_level = AbstractionLevel(args.abstraction)

    # Create CFR agent with specified abstraction level
    agent = CFRAgent(name="CFR_Trainee", abstraction_level=abstraction_level)
    logger.info(f"Using {args.abstraction} abstraction level")

    # Load checkpoint if provided
    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        logger.info(f"Loading checkpoint from {args.load_checkpoint}")
        agent.load_model(args.load_checkpoint)

    # Track training time
    start_time = time.time()

    # Create trainer based on mode
    if args.parallel:
        # Parallel training
        logger.info(f"Using parallel training with {args.workers or 'auto'} workers")

        if args.async_mode:
            logger.info("Using asynchronous parallel CFR")
            trainer = AsyncParallelCFRTrainer(
                agent,
                                language=args.language,
                num_workers=args.workers,
                sync_interval=args.sync_interval,
                use_mccfr=(args.algorithm == "mccfr")
            )
        else:
            logger.info("Using synchronous parallel CFR")
            trainer = ParallelCFRTrainer(
                agent,
                                language=args.language,
                num_workers=args.workers,
                sync_interval=args.sync_interval,
                use_mccfr=(args.algorithm == "mccfr")
            )

        # Run parallel training (handles checkpointing internally)
        trainer.train(args.iterations)

    else:
        # Sequential training
        if args.algorithm == "mccfr":
            logger.info("Using Monte Carlo CFR (MCCFR) algorithm")
            trainer = MCCFRTrainer(
                agent,
                                language=args.language
            )
        else:
            logger.info("Using Vanilla CFR algorithm")
            trainer = CFRTrainer(
                agent,
                                language=args.language
            )

        # Training with checkpointing
        iterations_completed = 0
        while iterations_completed < args.iterations:
            # Train for checkpoint interval or remaining iterations
            batch_size = min(
                args.checkpoint_interval,
                args.iterations - iterations_completed
            )

            logger.info(
                f"Training batch: {iterations_completed} to "
                f"{iterations_completed + batch_size}"
            )

            trainer.train(batch_size)
            iterations_completed += batch_size

            # Save checkpoint
            if iterations_completed % args.checkpoint_interval == 0:
                checkpoint_path = f"{args.output}.checkpoint_{iterations_completed}"
                agent.save_model(checkpoint_path)
                logger.info(f"Checkpoint saved: {checkpoint_path}")

    # Calculate training time
    training_time = time.time() - start_time

    # Save final model
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    agent.save_model(args.output)
    logger.info(f"Training complete. Model saved to {args.output}")

    # Print final statistics
    logger.info(f"Final statistics:")
    logger.info(f"  Total iterations: {agent.iteration_count}")
    logger.info(f"  Information sets: {agent.get_info_set_count()}")
    logger.info(f"  Estimated exploitability: {agent.get_exploitability():.6f}")
    logger.info(f"  Training time: {training_time:.2f} seconds")
    logger.info(f"  Iterations per second: {agent.iteration_count / training_time:.2f}")

    if args.parallel:
        speedup = (agent.iteration_count / training_time) / (agent.iteration_count / training_time * (args.workers or 1))
        logger.info(f"  Parallel efficiency: {speedup:.1%}")


if __name__ == "__main__":
    main()