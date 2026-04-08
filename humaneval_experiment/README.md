# HumanEval Experiment with GEPA

This directory contains the implementation for optimizing code generation prompts using GEPA (Genetic-Pareto prompt evolution) on the HumanEval benchmark.

## Overview

**HumanEval** is an execution-based code benchmark where:
- Each problem has a function signature + docstring
- The model generates the function body
- Correctness is measured by running unit tests

**Metric**: Pass@1 (probability that a single generated solution passes all tests)

## Directory Structure

```
humaneval_experiment/
├── train_humaneval.py       # Main GEPA optimization script
├── evaluate_test_set.py     # Test evaluation with seed vs optimized
├── humaneval_adapter.py     # Custom GEPAAdapter for code generation
├── code_executor.py         # Safe Python execution with timeout
├── experiment_logger.py     # Logging utilities
├── requirements.txt         # Dependencies
├── README.md               # This file
├── data/                   # Deterministic data splits
│   ├── README.md
│   ├── humaneval_train.csv  # Training set (created on first run)
│   ├── humaneval_val.csv    # Validation set (created on first run)
│   └── humaneval_test.csv   # Test set (created on first run)
├── results/                # All experiment results (timestamped)
│   └── gepa_humaneval_results_YYYYMMDD_HHMMSS/
│       ├── optimized_prompt.txt
│       ├── seed_prompt.txt
│       ├── experiment_config.json
│       ├── eval_checkpoints/
│       └── test_evaluation_report_*.json
└── evaluation/
    ├── __init__.py
    └── data_formatter.py   # HumanEval → GEPA format conversion
```

## Setup

1. **Install dependencies**:
```bash
cd humaneval_experiment
pip install -r requirements.txt
```

2. **Set API key**:
```bash
export OPENAI_API_KEY='your-key-here'
```

Or create a `.env` file in the project root:
```
OPENAI_API_KEY=your-key-here
```

## Usage

### Step 1: Run GEPA Optimization

```bash
python train_humaneval.py
```

This will:
- **First run**: Create deterministic CSV splits in `data/` directory
- Load the HumanEval dataset (164 problems)
- Split into train (80) / val (20) / test (64) sets
- Run GEPA to optimize the code generation prompt
- Save results to `./results/gepa_humaneval_results_<timestamp>/`

**Subsequent runs** will load existing CSV splits for reproducibility.

### Step 2: Evaluate on Test Set

```bash
python evaluate_test_set.py
```

This will:
- Load seed and optimized prompts from the results directory
- Load test set from `data/humaneval_test.csv` (deterministic)
- Evaluate both prompts on the held-out test set
- Generate comparison report with Pass@1 metrics

## Data Splits (Deterministic)

On first run, the following CSV files are created in `data/`:

| File | Size | Purpose |
|------|------|---------|
| `humaneval_train.csv` | 80 problems | GEPA optimization |
| `humaneval_val.csv` | 20 problems | Validation during optimization |
| `humaneval_test.csv` | 64 problems | Final evaluation |

**Random seed**: 42 (ensures reproducibility)

These files ensure that all experiments use the same splits.

## Configuration

### train_humaneval.py

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRAIN_SIZE` | 80 | Number of training problems |
| `VAL_SIZE` | 20 | Number of validation problems |
| `MAX_METRIC_CALLS` | 150 | GEPA optimization budget |
| `REFLECTION_MINIBATCH_SIZE` | 5 | Examples per reflection batch |
| `TASK_LM` | gpt-4.1-mini | LLM for code generation |
| `REFLECTION_LM` | gpt-5 | LLM for reflection |
| `EXECUTION_TIMEOUT` | 5.0 | Seconds per code execution |

### evaluate_test_set.py

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RUN_DIR` | ./results/... | Path to results directory |
| `TEST_SIZE` | 64 | Number of test problems |
| `BATCH_SIZE` | 10 | Examples per checkpoint |

## Key Components

### HumanEvalAdapter

The adapter handles:
1. **Code Generation**: Calls LLM with the prompt and problem
2. **Execution**: Runs generated code against unit tests
3. **Feedback**: Provides rich error information for reflection

Key methods:
- `evaluate()`: Generate code → Execute → Return Pass@1 scores
- `make_reflective_dataset()`: Build feedback with tracebacks for GEPA

### CodeExecutor

Safe code execution with:
- Multiprocessing isolation (prevents crashes from affecting main process)
- Timeout handling (kills infinite loops)
- Rich error capture (syntax, runtime, assertion, timeout errors)

### DataFormatter

Converts HumanEval format to GEPA format:
```python
# HumanEval → GEPA
{
    "task_id": "HumanEval/0",
    "prompt": "def add(a, b):...",
    "test": "assert add(1,2)==3",
    "entry_point": "add"
}
↓
{
    "input": "def add(a, b):...",
    "answer": "<canonical solution>",
    "additional_context": {
        "task_id": "HumanEval/0",
        "test": "assert add(1,2)==3",
        "entry_point": "add"
    }
}
```

## Expected Results

Based on our HoVer experiments, we expect:
- **Seed Prompt Pass@1**: ~50-60%
- **GEPA Optimized Pass@1**: ~55-70%
- **Improvement**: +5-10% absolute

The key advantage of GEPA for code generation is the rich feedback from:
- Tracebacks showing exact errors
- Assertion failures with test case details
- Syntax errors with line numbers

## Comparison with HoVer Experiment

| Aspect | HoVer | HumanEval |
|--------|-------|-----------|
| Task | Fact verification | Code generation |
| Metric | Accuracy | Pass@1 |
| Feedback | Binary (correct/wrong) | Rich (tracebacks, errors) |
| Evaluation | String matching | Code execution |

## Troubleshooting

### Rate Limits
Reduce `LITELLM_MAX_WORKERS` to 1 or 2 if hitting API rate limits.

### Timeout Errors
Increase `EXECUTION_TIMEOUT` if complex problems are timing out.

### Memory Issues
Reduce `TRAIN_SIZE` or `BATCH_SIZE` if running out of memory.

## References

- [HumanEval Paper](https://arxiv.org/abs/2107.03374)
- [GEPA Paper](https://arxiv.org/abs/2507.19457)
- [HumanEval Dataset on HuggingFace](https://huggingface.co/datasets/openai/openai_humaneval)
