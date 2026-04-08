# HumanEval Data Splits

This directory contains deterministic train/val/test splits for the HumanEval experiment.

## Files

- `humaneval_train.csv` - Training set (80 problems)
- `humaneval_val.csv` - Validation set (20 problems)
- `humaneval_test.csv` - Test set (64 problems)

## Generation

These files are automatically created on the first run of `train_humaneval.py` using:
- Random seed: 42
- Total HumanEval problems: 164

## Format

Each CSV has the following columns:

| Column | Description |
|--------|-------------|
| `task_id` | HumanEval task ID (e.g., "HumanEval/0") |
| `entry_point` | Function name to implement |
| `input` | Problem prompt (signature + docstring) |
| `answer` | Canonical solution (reference only) |
| `test` | Unit test code |

## Reproducibility

Once created, these files ensure that:
1. All experiments use the same train/val/test splits
2. Results are reproducible across runs
3. Evaluation is fair (test set never seen during training)

## Regenerating Splits

To regenerate splits (not recommended unless necessary):
1. Delete all CSV files in this directory
2. Run `train_humaneval.py` again
