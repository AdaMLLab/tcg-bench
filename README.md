# TCG-Bench

A contamination-resistant benchmark for evaluating strategic reasoning in large language models through a trading card game (Sacra Battle).

## Overview

TCG-Bench pits LLM agents against algorithmic opponents (MCTS, rollout-based evaluation, random) in a turn-based card game with 15 cards across three types: Champions (units with Power/Guard stats), Spells (one-time effects), and Tricks (trigger-based reactions). The benchmark supports English and Arabic, enabling multilingual evaluation.

Each turn follows a fixed structure: Draw, Main Phase (play one card), Combat. The game ends when a player reaches 0 life points.

## Setup

```bash
pip install openai tiktoken numpy tqdm scipy statsmodels python-dotenv
```

Set your OpenRouter API key in a `.env` file or as an environment variable:

```bash
export OPENROUTER_API_KEY=your_key_here
```

## Usage

```bash
# LLM vs MCTS (50 rollouts), 100 games, English
python src/main.py \
    --model openai/gpt-4o \
    --language en \
    --opponent mcts \
    --rollout_count 50 \
    --num_batches 100 \
    --games_per_batch 1 \
    --full_deck \
    --add_rules \
    --add_cards \
    --agent_llm_append_cards \
    --seed 42

# LLM vs rollout agent, Arabic
python src/main.py \
    --model openai/gpt-4o \
    --language ar \
    --opponent rollout \
    --rollout_count 10 \
    --num_batches 100 \
    --games_per_batch 1 \
    --full_deck \
    --add_rules \
    --add_cards \
    --agent_llm_append_cards \
    --seed 42

# Multiple models in sequence
python src/main.py \
    --models openai/gpt-4o,deepseek-ai/DeepSeek-R1 \
    --language en \
    --opponent mcts \
    --rollout_count 50 \
    --num_batches 100 \
    --games_per_batch 1 \
    --full_deck \
    --add_rules \
    --add_cards \
    --agent_llm_append_cards \
    --seed 42

# Random baseline (no API calls)
python src/random_baseline.py
```

### Opponent types

| Opponent | Flag | Description |
|----------|------|-------------|
| MCTS | `--opponent mcts` | Monte Carlo Tree Search with UCB1 exploration. Rollout count controls strength. |
| Rollout | `--opponent rollout` | Flat per-card evaluation via random playouts. Evaluates every card independently. |
| Random | `--opponent random` | Uniform random card selection. |
| CFR | `--opponent cfr` | Counterfactual Regret Minimization (requires pre-trained model via `--cfr-model`). |

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `gpt-4o` | OpenRouter model identifier |
| `--language` | `en` | `en` or `ar` |
| `--opponent` | `mcts` | `random`, `rollout`, `mcts`, or `cfr` |
| `--rollout_count` | `10` | Rollouts for MCTS/rollout agents |
| `--num_batches` | `1` | Number of games |
| `--games_per_batch` | `1` | Games per batch |
| `--full_deck` | off | Give each player the full 15-card pool |
| `--add_rules` | off | Include game rules in LLM prompt |
| `--add_cards` | off | Include card descriptions in LLM prompt |
| `--parse_mode` | `soft` | `strict` (tags only) or `soft` (substring search) |
| `--llm_type` | `openrouter` | `openrouter` or `vllm` |
| `--seed` | `42` | Random seed |

## Output

Results are saved to `results/{model_name}/{language}_rollout{count}_{timestamp}/`:

- `benchmark_metrics.json`: Win rates with bootstrap 95% CIs, effect sizes, decision times, token usage, process metrics
- `config.json`: Full run configuration
- `detailed_game_logs.json`: Per-game turn-by-turn logs

## Architecture

```
src/
  main.py                 # Benchmark orchestration (async game runner)
  game.py                 # Core game state, Card/Player/GameState classes
  community_engine.py     # Card pool (15 cards, EN/AR) and effect resolution
  agents.py               # LLMAgent, MCTSAgent, RolloutAgent, RandomAgent
  statistical_analysis.py # Bootstrap CIs, Cohen's d, power analysis
  process_metrics.py      # Move validity, resource efficiency tracking
  random_baseline.py      # Random-vs-opponent baseline runner
  utils.py                # Confidence intervals, multilingual analysis
  cfr/                    # Counterfactual Regret Minimization agent and trainer
```

Games run fully in parallel via asyncio. LLM calls are async I/O; MCTS/rollout moves run in a ProcessPoolExecutor.

## Citation

If you use TCG-Bench in your research, please cite:

```bibtex
@inproceedings{alrashed-etal-2026-cards,
    title = "Cards Against Contamination: {TCG}-Bench for Difficulty-Scalable Multilingual {LLM} Reasoning",
    author = "AlRashed, Sultan and Wang, Jianghui and Orabona, Francesco",
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2026",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-eacl.353/",
    pages = "6710--6724",
}
```
