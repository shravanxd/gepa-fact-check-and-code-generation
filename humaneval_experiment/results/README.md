# HumanEval Experiment Results

This directory contains results from GEPA optimization experiments on the HumanEval dataset.

## Directory Structure

Each experiment run creates a timestamped subdirectory:

```
gepa_humaneval_results_YYYYMMDD_HHMMSS/
├── optimized_prompt.txt        # Best prompt found by GEPA
├── seed_prompt.txt             # Initial seed prompt
├── experiment_config.json      # Hyperparameters and settings
├── gepa_state.bin             # GEPA checkpoint (for resuming)
├── eval_checkpoints/          # Evaluation checkpoints
│   ├── seed_checkpoint.json
│   └── optimized_checkpoint.json
├── test_evaluation_report_*.json    # Evaluation summary
└── test_detailed_results_*.json     # Per-problem results
```

## Finding the Latest Run

The `evaluate_test_set.py` script automatically finds the latest run directory if one isn't specified.

## Comparing Runs

To compare different runs:
1. Look at `experiment_config.json` for hyperparameters
2. Compare `test_evaluation_report_*.json` for Pass@1 scores
3. Review `optimized_prompt.txt` to see how prompts evolved
