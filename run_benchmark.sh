#!/bin/bash

# run_benchmark.sh - Run TCG-Bench with specified parameters
# Usage: ./run_benchmark.sh [model_name] [language] [opponent] [rollout]

# Default values
MODEL=${1:-"gpt-4o"}
LANGUAGE=${2:-"en"}
OPPONENT=${3:-"mcts"}
ROLLOUT=${4:-5}
BATCHES=${5:-1}
GAMES_PER_BATCH=${6:-1}

# Create results directory if it doesn't exist
mkdir -p results

# Log settings
echo "Running TCG-Bench with the following settings:"
echo "- Model: $MODEL"
echo "- Language: $LANGUAGE"
echo "- Opponent: $OPPONENT"
echo "- Rollout count: $ROLLOUT"
echo "- Batches: $BATCHES"
echo "- Games per batch: $GAMES_PER_BATCH"
echo ""

# Export OpenRouter API key if it's not already set
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "Warning: OPENROUTER_API_KEY is not set. Using dummy key for testing."
    export OPENROUTER_API_KEY="dummy_key_for_testing"
fi

# Run the benchmark
python main.py \
    --model "$MODEL" \
    --language "$LANGUAGE" \
    --opponent "$OPPONENT" \
    --rollout_count "$ROLLOUT" \
    --num_batches "$BATCHES" \
    --games_per_batch "$GAMES_PER_BATCH" \
    --add_rules \
    --add_cards \
    --agent_llm_append_cards \
    --full_deck \
    --seed 42 \
    --log_level INFO

# Check if the command executed successfully
if [ $? -eq 0 ]; then
    echo "Benchmark completed successfully."
    echo "Results saved to the 'results/' directory."
else
    echo "Benchmark execution failed. Check logs for errors."
    exit 1
fi