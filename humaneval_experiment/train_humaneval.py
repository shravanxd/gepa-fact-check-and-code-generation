"""
HumanEval Dataset Optimization with GEPA
=========================================

This script optimizes prompts for code generation on the HumanEval dataset using GEPA.

HumanEval is an execution-based code benchmark where:
- Each problem has a function signature + docstring
- The model generates the function body
- Correctness is measured by running unit tests (Pass@1)

Usage:
    pip install -r requirements.txt
    python train_humaneval.py
"""

import os
import csv
import json
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import gepa
from datasets import load_dataset
from dotenv import load_dotenv

from humaneval_adapter import HumanEvalAdapter
from experiment_logger import ExperimentLogger
from evaluation.data_formatter import DataFormatter


# ============================================================================
# Environment Setup
# ============================================================================

def _load_env():
    """Load environment variables from .env file (searching upward)."""
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        candidate = parent / ".env"
        if candidate.exists():
            load_dotenv(candidate)
            break
    else:
        load_dotenv()

_load_env()


# ============================================================================
# CONFIGURATION - Modify these variables as needed
# ============================================================================

# Checkpoint behavior
USE_TIMESTAMP_DIR = True  # Set to False to resume from existing directory
RESUME_FROM_CHECKPOINT = True  # Set to True to resume from checkpoint

# Dataset sizes
# HumanEval has 164 problems total (50 + 50 + 64 = 164)
TRAIN_SIZE = 50  # Training set size
VAL_SIZE = 50    # Validation set size

# GEPA optimization parameters
MAX_METRIC_CALLS = 500  # Budget for metric evaluations
REFLECTION_MINIBATCH_SIZE = 5  # Examples per reflection batch

# LLM Configuration
LITELLM_MAX_WORKERS = 10  # Parallel requests for faster evaluation
TASK_LM = "gpt-3.5-turbo"  # LLM for code generation (weaker model for more room to improve)
REFLECTION_LM = "gpt-5"   # LLM for reflection/prompt optimization

# Code execution
EXECUTION_TIMEOUT = 5.0  # Seconds per test execution

# Reproducibility
RANDOM_SEED = 42

# Data split files (for deterministic splits)
EXPERIMENT_DIR = Path(__file__).parent  # humaneval_experiment directory
DATA_DIR = EXPERIMENT_DIR / "data"
RESULTS_DIR = EXPERIMENT_DIR / "results"
TRAIN_CSV = DATA_DIR / "humaneval_train.csv"
VAL_CSV = DATA_DIR / "humaneval_val.csv"
TEST_CSV = DATA_DIR / "humaneval_test.csv"

# Set seed for reproducibility
random.seed(RANDOM_SEED)


# ============================================================================
# Seed Prompt Definition
# ============================================================================

SEED_PROMPT = {
    "system_prompt": """Complete the below function."""
}


# ============================================================================
# Data Split Management
# ============================================================================

def save_examples_to_csv(examples: List[Dict[str, Any]], filepath: Path) -> None:
    """
    Save examples to CSV file for deterministic loading.
    
    Args:
        examples: List of GEPA-formatted examples
        filepath: Path to save CSV
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['task_id', 'entry_point', 'input', 'answer', 'test'])
        writer.writeheader()
        
        for ex in examples:
            row = {
                'task_id': ex['additional_context']['task_id'],
                'entry_point': ex['additional_context']['entry_point'],
                'input': ex['input'],
                'answer': ex['answer'],
                'test': ex['additional_context']['test']
            }
            writer.writerow(row)
    
    print(f"✓ Saved {len(examples)} examples to {filepath}")


def load_examples_from_csv(filepath: Path) -> List[Dict[str, Any]]:
    """
    Load examples from CSV file.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        List of GEPA-formatted examples
    """
    examples = []
    
    with open(filepath, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            example = {
                'input': row['input'],
                'answer': row['answer'],
                'additional_context': {
                    'task_id': row['task_id'],
                    'entry_point': row['entry_point'],
                    'test': row['test']
                }
            }
            examples.append(example)
    
    print(f"✓ Loaded {len(examples)} examples from {filepath}")
    return examples


def create_or_load_splits() -> tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Create deterministic train/val/test splits or load existing ones.
    
    If CSV files exist, load from them (deterministic).
    If not, create new splits, save to CSV, then use them.
    
    Returns:
        Tuple of (trainset, valset, testset)
    """
    # Check if all split files exist
    splits_exist = TRAIN_CSV.exists() and VAL_CSV.exists() and TEST_CSV.exists()
    
    if splits_exist:
        print("\n" + "=" * 60)
        print("LOADING EXISTING DATA SPLITS (deterministic)")
        print("=" * 60)
        
        trainset = load_examples_from_csv(TRAIN_CSV)
        valset = load_examples_from_csv(VAL_CSV)
        testset = load_examples_from_csv(TEST_CSV)
        
        print(f"\nLoaded splits:")
        print(f"  Training: {len(trainset)} examples")
        print(f"  Validation: {len(valset)} examples")
        print(f"  Test: {len(testset)} examples")
        
        return trainset, valset, testset
    
    # Create new splits
    print("\n" + "=" * 60)
    print("CREATING NEW DATA SPLITS")
    print("=" * 60)
    print(f"Split files not found. Creating deterministic splits...")
    print(f"  Train CSV: {TRAIN_CSV}")
    print(f"  Val CSV: {VAL_CSV}")
    print(f"  Test CSV: {TEST_CSV}")
    
    # Load HumanEval dataset
    print("\nLoading HumanEval dataset from HuggingFace...")
    ds = load_dataset("openai/openai_humaneval")
    full_data = ds['test']
    print(f"Total problems: {len(full_data)}")
    
    # Convert to GEPA format
    print("Converting to GEPA format...")
    all_examples = [DataFormatter.humaneval_to_gepa_format(ex) for ex in full_data]
    
    # Shuffle with seed for reproducibility
    random.seed(RANDOM_SEED)
    random.shuffle(all_examples)
    
    # Split
    trainset = all_examples[:TRAIN_SIZE]
    valset = all_examples[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    testset = all_examples[TRAIN_SIZE + VAL_SIZE:]
    
    # Save to CSV files
    print("\nSaving splits to CSV files...")
    save_examples_to_csv(trainset, TRAIN_CSV)
    save_examples_to_csv(valset, VAL_CSV)
    save_examples_to_csv(testset, TEST_CSV)
    
    print(f"\n✓ Created deterministic splits:")
    print(f"  Training: {len(trainset)} examples")
    print(f"  Validation: {len(valset)} examples")
    print(f"  Test: {len(testset)} examples")
    
    return trainset, valset, testset


# ============================================================================
# Main Training Script
# ============================================================================

def humaneval_to_gepa_format(example):
    """
    Convert a HumanEval example to GEPA format using the shared DataFormatter.
    
    Args:
        example: HumanEval dataset example
        
    Returns:
        Dict in GEPA format
    """
    return DataFormatter.humaneval_to_gepa_format(example)


def main():
    """Main entry point for HumanEval GEPA optimization."""
    
    # ========================================================================
    # Step 1: Create or load deterministic data splits
    # ========================================================================
    trainset, valset, testset = create_or_load_splits()
    
    # ========================================================================
    # Step 2: Preview one example
    # ========================================================================
    print("\n" + "=" * 60)
    print("EXAMPLE PROBLEM:")
    print("=" * 60)
    example = trainset[0]
    print(f"Task ID: {example['additional_context']['task_id']}")
    print(f"Entry Point: {example['additional_context']['entry_point']}")
    print(f"\nInput (prompt):\n{example['input'][:500]}...")
    print("=" * 60 + "\n")
    
    # ========================================================================
    # Step 3: Check for API key
    # ========================================================================
    if "OPENAI_API_KEY" not in os.environ:
        print("WARNING: OPENAI_API_KEY not found in environment variables!")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    # ========================================================================
    # Step 4: Set up adapter
    # ========================================================================
    print(f"\n{'=' * 60}")
    print("Setting up HumanEvalAdapter with:")
    print(f"  - Task LM: {TASK_LM}")
    print(f"  - Execution timeout: {EXECUTION_TIMEOUT}s")
    print(f"  - Max workers: {LITELLM_MAX_WORKERS}")
    print(f"{'=' * 60}\n")
    
    adapter = HumanEvalAdapter(
        model=TASK_LM,
        timeout=EXECUTION_TIMEOUT,
        max_litellm_workers=LITELLM_MAX_WORKERS
    )
    
    # ========================================================================
    # Step 5: Determine run directory (inside humaneval_experiment/results/)
    # ========================================================================
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if USE_TIMESTAMP_DIR:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RESULTS_DIR / f"gepa_humaneval_results_{timestamp}"
        print(f"Creating new run directory: {run_dir}")
    else:
        run_dir = RESULTS_DIR / "gepa_humaneval_results_latest"
        if RESUME_FROM_CHECKPOINT and (run_dir / "gepa_state.bin").exists():
            print(f"Resuming from checkpoint in: {run_dir}")
        elif (run_dir / "gepa_state.bin").exists():
            print(f"WARNING: Checkpoint exists but RESUME_FROM_CHECKPOINT=False")
            print(f"Deleting old checkpoint to start fresh...")
            import shutil
            shutil.rmtree(run_dir)
        else:
            print(f"Starting fresh in: {run_dir}")
    
    # Create run directory
    run_dir.mkdir(parents=True, exist_ok=True)
    run_dir_str = str(run_dir)  # Convert to string for GEPA compatibility
    
    # ========================================================================
    # Step 6: Run GEPA optimization
    # ========================================================================
    print("\n" + "=" * 60)
    print("Starting GEPA optimization...")
    print("=" * 60)
    print(f"Training set: {len(trainset)} problems")
    print(f"Validation set: {len(valset)} problems")
    print(f"Test set (held out): {len(testset)} problems")
    print(f"Minibatch size: {REFLECTION_MINIBATCH_SIZE} examples per iteration")
    print(f"Budget: {MAX_METRIC_CALLS} metric calls")
    print(f"Task LM: {TASK_LM}")
    print(f"Reflection LM: {REFLECTION_LM}\n")
    
    result = gepa.optimize(
        seed_candidate=SEED_PROMPT,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=REFLECTION_LM,
        reflection_minibatch_size=REFLECTION_MINIBATCH_SIZE,
        max_metric_calls=MAX_METRIC_CALLS,
        run_dir=run_dir_str,
        display_progress_bar=True
    )
    
    # ========================================================================
    # Step 7: Display results
    # ========================================================================
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 60)
    
    print(f"\nOriginal Seed Prompt:")
    print("-" * 60)
    print(SEED_PROMPT['system_prompt'][:500] + "..." if len(SEED_PROMPT['system_prompt']) > 500 else SEED_PROMPT['system_prompt'])
    
    print("\n" + "=" * 60)
    print(f"\nOptimized Prompt:")
    print("-" * 60)
    optimized = result.best_candidate['system_prompt']
    print(optimized[:800] + "..." if len(optimized) > 800 else optimized)
    
    print("\n" + "=" * 60)
    
    # Get best score
    best_score = result.val_aggregate_scores[result.best_idx]
    print(f"\nBest Validation Score (Pass@1): {best_score:.2%}")
    print(f"Total Metric Calls: {result.total_metric_calls}")
    print(f"Number of Candidates Evaluated: {result.num_candidates}")
    print("=" * 60)
    
    # ========================================================================
    # Step 8: Save artifacts
    # ========================================================================
    
    # Save optimized prompt
    output_file = run_dir / "optimized_prompt.txt"
    with open(output_file, "w", encoding='utf-8') as f:
        f.write(result.best_candidate['system_prompt'])
    print(f"\n✓ Optimized prompt saved to: {output_file}")
    
    # Save seed prompt
    seed_file = run_dir / "seed_prompt.txt"
    with open(seed_file, "w", encoding='utf-8') as f:
        f.write(SEED_PROMPT['system_prompt'])
    print(f"✓ Seed prompt saved to: {seed_file}")
    
    # Save experiment configuration
    hyperparams = {
        "train_size": len(trainset),
        "val_size": len(valset),
        "test_size": len(testset),
        "reflection_minibatch_size": REFLECTION_MINIBATCH_SIZE,
        "max_metric_calls": MAX_METRIC_CALLS,
        "task_lm": TASK_LM,
        "reflection_lm": REFLECTION_LM,
        "execution_timeout": EXECUTION_TIMEOUT,
        "litellm_max_workers": LITELLM_MAX_WORKERS,
        "random_seed": RANDOM_SEED,
        "data_files": {
            "train_csv": str(TRAIN_CSV),
            "val_csv": str(VAL_CSV),
            "test_csv": str(TEST_CSV),
        }
    }
    
    logger = ExperimentLogger(run_dir_str)
    logger.save_experiment_config(result, hyperparams, SEED_PROMPT)
    
    print("\n" + "=" * 60)
    print("Training complete! Next steps:")
    print("1. Review the optimized prompt")
    print(f"2. Run evaluate_test_set.py (uses {TEST_CSV})")
    print(f"3. Results saved in: {run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
