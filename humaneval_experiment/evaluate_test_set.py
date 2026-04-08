"""
Test Set Evaluation Script for HumanEval
=========================================

Evaluates both seed and optimized prompts on the HumanEval test set.
Compares performance and generates detailed reports.

Uses deterministic test split from data/humaneval_test.csv

Usage:
    python evaluate_test_set.py
"""

import os
import csv
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime

from datasets import load_dataset
from dotenv import load_dotenv

from humaneval_adapter import HumanEvalAdapter
from experiment_logger import ExperimentLogger
from evaluation.data_formatter import DataFormatter


# ============================================================================
# Environment Setup
# ============================================================================

def _load_env():
    """Load environment variables from .env file."""
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
# USER CONFIGURABLE VARIABLES
# ============================================================================

# Base directories (inside humaneval_experiment)
EXPERIMENT_DIR = Path(__file__).parent  # humaneval_experiment directory
RESULTS_DIR = EXPERIMENT_DIR / "results"
DATA_DIR = EXPERIMENT_DIR / "data"

# Path to the run directory containing optimized_prompt.txt
# Update this to point to your specific run directory
RUN_DIR = RESULTS_DIR / "gepa_humaneval_results_20251203_084234"

# Test configuration
TEST_PERCENTAGE = 100  # Percentage of test set to use (1-100). Use 100 for full evaluation
TEST_CSV = DATA_DIR / "humaneval_test.csv"

# LLM Configuration
TASK_LM = "gpt-3.5-turbo"  # Must match the model used during training
MAX_WORKERS = 10
EXECUTION_TIMEOUT = 5.0

# Batch size for checkpointing
BATCH_SIZE = 10

# Reproducibility
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ============================================================================
# Checkpoint Manager
# ============================================================================

class CheckpointManager:
    """Manages evaluation checkpoints for interruption recovery."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, prompt_type: str, results: List[Dict], progress: int):
        """Save evaluation checkpoint."""
        checkpoint = {
            "results": results,
            "progress": progress,
            "timestamp": datetime.now().isoformat()
        }
        filepath = self.checkpoint_dir / f"{prompt_type}_checkpoint.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2)
    
    def load_checkpoint(self, prompt_type: str) -> Optional[Dict]:
        """Load evaluation checkpoint if exists."""
        filepath = self.checkpoint_dir / f"{prompt_type}_checkpoint.json"
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def clear_checkpoint(self, prompt_type: str):
        """Clear checkpoint after successful completion."""
        filepath = self.checkpoint_dir / f"{prompt_type}_checkpoint.json"
        if filepath.exists():
            filepath.unlink()


# ============================================================================
# Prompt Evaluator
# ============================================================================

class PromptEvaluator:
    """Evaluates prompts on HumanEval with checkpoint support."""
    
    def __init__(self, adapter: HumanEvalAdapter, checkpoint_manager: CheckpointManager):
        self.adapter = adapter
        self.checkpoint_manager = checkpoint_manager
    
    def evaluate_with_checkpoints(
        self,
        prompt: Dict[str, str],
        test_examples: List[Dict],
        prompt_type: str,
        batch_size: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate a prompt with checkpoint support.
        
        Args:
            prompt: Prompt dictionary with 'system_prompt' key
            test_examples: List of test examples in GEPA format
            prompt_type: 'seed' or 'optimized'
            batch_size: Number of examples per checkpoint
            
        Returns:
            Dict with evaluation results
        """
        print(f"\nEvaluating {prompt_type} prompt on {len(test_examples)} examples...")
        
        # Try to resume from checkpoint
        checkpoint = self.checkpoint_manager.load_checkpoint(prompt_type)
        
        if checkpoint:
            results = checkpoint['results']
            start_idx = checkpoint['progress']
            print(f"  ↻ Resuming from example {start_idx}")
        else:
            results = []
            start_idx = 0
            print(f"  ▸ Starting fresh evaluation")
        
        # Process in batches
        try:
            for batch_start in range(start_idx, len(test_examples), batch_size):
                batch_end = min(batch_start + batch_size, len(test_examples))
                batch = test_examples[batch_start:batch_end]
                
                print(f"  Processing examples {batch_start + 1}-{batch_end}...")
                
                # Evaluate batch
                batch_results = self.adapter.evaluate(
                    batch=batch,
                    candidate=prompt
                )
                
                # Store results with task info
                for i, (output, score) in enumerate(zip(batch_results.outputs, batch_results.scores)):
                    example = batch[i]
                    task_id = example['additional_context']['task_id']
                    entry_point = example['additional_context']['entry_point']
                    test_code = example['additional_context'].get('test', '')
                    passed = output['passed']
                    
                    # Print result for each problem
                    status = "✓ PASS" if passed else "✗ FAIL"
                    error_info = f" ({output.get('error_type', 'Unknown')})" if not passed else ""
                    print(f"\n    [{task_id}] {entry_point}: {status}{error_info}")
                    
                    # Show test cases (truncated)
                    if test_code:
                        test_preview = test_code[:300].replace('\n', '\n      ')
                        print(f"      Tests:\n      {test_preview}...")
                    
                    # Show code snippet (first 200 chars)
                    code_snippet = output.get('extracted_code', '')[:300]
                    if code_snippet:
                        code_preview = code_snippet.replace('\n', '\n      ')
                        print(f"      Generated Code:\n      {code_preview}...")
                    
                    # Show error message if failed
                    if not passed and output.get('error_message'):
                        error_msg = output.get('error_message', '')[:200]
                        print(f"      Error: {error_msg}")
                    
                    result_entry = {
                        "task_id": task_id,
                        "entry_point": entry_point,
                        "passed": passed,
                        "score": score,
                        "error_type": output.get('error_type'),
                        "error_message": output.get('error_message'),
                        "extracted_code": output.get('extracted_code', '')[:500],  # Truncate
                    }
                    results.append(result_entry)
                
                # Save checkpoint
                self.checkpoint_manager.save_checkpoint(prompt_type, results, len(results))
            
            # Calculate metrics
            metrics = self._calculate_metrics(results)
            
            # Clear checkpoint on success
            self.checkpoint_manager.clear_checkpoint(prompt_type)
            
            return {
                "pass_at_1": metrics["pass_at_1"],
                "passed": metrics["passed"],
                "total": metrics["total"],
                "error_distribution": metrics["error_distribution"],
                "detailed_results": results
            }
        
        except KeyboardInterrupt:
            print(f"\n  ⚠ Evaluation interrupted! Progress saved to checkpoint.")
            print(f"  Run again to resume from example {len(results)}")
            raise
        except Exception as e:
            print(f"\n  ✗ Error during evaluation: {e}")
            print(f"  Progress saved. Run again to resume.")
            raise
    
    def _calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate evaluation metrics."""
        total = len(results)
        passed = sum(1 for r in results if r['passed'])
        
        # Error distribution
        error_counts = {}
        for r in results:
            if not r['passed']:
                error_type = r.get('error_type', 'Unknown')
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return {
            "pass_at_1": passed / total if total > 0 else 0.0,
            "passed": passed,
            "total": total,
            "error_distribution": error_counts
        }
    
    def evaluate_pass_at_k(
        self,
        prompt: Dict[str, str],
        test_examples: List[Dict],
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate Pass@k metric by generating k samples per problem.
        
        A problem is considered solved if ANY of the k samples passes.
        
        Args:
            prompt: Prompt dictionary with 'system_prompt' key
            test_examples: List of test examples
            k: Number of samples per problem
            
        Returns:
            Dict with pass@k results
        """
        print(f"\n  Evaluating Pass@{k} ({k} samples per problem)...")
        
        passed_problems = 0
        total_problems = len(test_examples)
        detailed_results = []
        
        for idx, example in enumerate(test_examples):
            task_id = example['additional_context']['task_id']
            entry_point = example['additional_context']['entry_point']
            test_code = example['additional_context'].get('test', '')
            
            print(f"\n    [{idx + 1}/{total_problems}] {task_id} ({entry_point}):")
            
            # Show test cases (truncated)
            if test_code:
                test_preview = test_code[:250].replace('\n', '\n        ')
                print(f"      Tests:\n        {test_preview}...")
            
            # Generate k samples for this problem
            samples_passed = 0
            sample_results = []
            
            for sample_idx in range(k):
                # Evaluate single example
                batch_result = self.adapter.evaluate(
                    batch=[example],
                    candidate=prompt
                )
                
                output = batch_result.outputs[0]
                passed = output['passed']
                
                # Print each sample result
                status = "✓" if passed else "✗"
                error_info = f" ({output.get('error_type', '')})" if not passed and output.get('error_type') else ""
                code_snippet = output.get('extracted_code', '')[:150].replace('\n', ' ')
                print(f"      Sample {sample_idx + 1}: {status}{error_info}")
                print(f"        Code: {code_snippet}...")
                
                # Show error if failed
                if not passed and output.get('error_message'):
                    error_msg = output.get('error_message', '')[:150]
                    print(f"        Error: {error_msg}")
                
                sample_results.append({
                    "sample": sample_idx + 1,
                    "passed": passed,
                    "error_type": output.get('error_type'),
                    "code_snippet": output.get('extracted_code', '')[:200],
                })
                
                if passed:
                    samples_passed += 1
            
            # Problem passes if ANY sample passed
            problem_passed = samples_passed > 0
            if problem_passed:
                passed_problems += 1
            
            # Print problem summary
            result_status = "✓ SOLVED" if problem_passed else "✗ FAILED"
            print(f"      → {result_status} ({samples_passed}/{k} samples passed)")
            
            detailed_results.append({
                "task_id": task_id,
                "problem_passed": problem_passed,
                "samples_passed": samples_passed,
                "samples_total": k,
                "sample_results": sample_results
            })
        
        pass_at_k = passed_problems / total_problems if total_problems > 0 else 0.0
        
        return {
            "k": k,
            "pass_at_k": pass_at_k,
            "passed": passed_problems,
            "total": total_problems,
            "detailed_results": detailed_results
        }


# ============================================================================
# Report Generator
# ============================================================================

class ReportGenerator:
    """Generates evaluation reports."""
    
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
    
    def generate_report(
        self,
        seed_results: Dict,
        optimized_results: Dict,
        test_size: int
    ) -> Dict:
        """Generate comparison report."""
        seed_pass = seed_results['pass_at_1']
        opt_pass = optimized_results['pass_at_1']
        
        improvement = opt_pass - seed_pass
        relative_improvement = (improvement / seed_pass * 100) if seed_pass > 0 else 0
        
        # Get Pass@3 and Pass@5 if available
        seed_pass3 = seed_results.get('pass_at_3', None)
        opt_pass3 = optimized_results.get('pass_at_3', None)
        seed_pass5 = seed_results.get('pass_at_5', None)
        opt_pass5 = optimized_results.get('pass_at_5', None)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "test_size": test_size,
            "seed_prompt": {
                "pass_at_1": seed_pass,
                "pass_at_3": seed_pass3,
                "pass_at_5": seed_pass5,
                "passed": seed_results['passed'],
                "total": seed_results['total'],
                "error_distribution": seed_results['error_distribution']
            },
            "optimized_prompt": {
                "pass_at_1": opt_pass,
                "pass_at_3": opt_pass3,
                "pass_at_5": opt_pass5,
                "passed": optimized_results['passed'],
                "total": optimized_results['total'],
                "error_distribution": optimized_results['error_distribution']
            },
            "comparison": {
                "absolute_improvement": improvement,
                "relative_improvement_percent": relative_improvement,
                "additional_problems_solved": optimized_results['passed'] - seed_results['passed']
            }
        }
        
        # Add Pass@3 comparison if available
        if seed_pass3 is not None and opt_pass3 is not None:
            pass3_improvement = opt_pass3 - seed_pass3
            report["comparison"]["pass_at_3_improvement"] = pass3_improvement
        
        # Add Pass@5 comparison if available
        if seed_pass5 is not None and opt_pass5 is not None:
            pass5_improvement = opt_pass5 - seed_pass5
            report["comparison"]["pass_at_5_improvement"] = pass5_improvement
        
        return report
    
    def save_report(
        self,
        report: Dict,
        seed_results: Dict,
        optimized_results: Dict
    ):
        """Save report and detailed results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary report
        report_path = self.run_dir / f"test_evaluation_report_{timestamp}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print(f"✓ Report saved to: {report_path}")
        
        # Save detailed results
        detailed = {
            "seed_results": seed_results['detailed_results'],
            "optimized_results": optimized_results['detailed_results']
        }
        detailed_path = self.run_dir / f"test_detailed_results_{timestamp}.json"
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(detailed, f, indent=2)
        print(f"✓ Detailed results saved to: {detailed_path}")
    
    def print_summary(self, report: Dict):
        """Print summary to console."""
        print("\n" + "=" * 70)
        print("TEST SET EVALUATION SUMMARY")
        print("=" * 70)
        
        seed = report['seed_prompt']
        opt = report['optimized_prompt']
        comp = report['comparison']
        
        print(f"\nTest Set Size: {report['test_size']} examples")
        
        print("\n" + "-" * 70)
        print("SEED PROMPT PERFORMANCE")
        print("-" * 70)
        print(f"Pass@1: {seed['pass_at_1']:.2%}")
        if seed.get('pass_at_3') is not None:
            print(f"Pass@3: {seed['pass_at_3']:.2%}")
        if seed.get('pass_at_5') is not None:
            print(f"Pass@5: {seed['pass_at_5']:.2%}")
        print(f"Problems Solved (Pass@1): {seed['passed']} / {seed['total']}")
        
        print("\n" + "-" * 70)
        print("OPTIMIZED PROMPT PERFORMANCE")
        print("-" * 70)
        print(f"Pass@1: {opt['pass_at_1']:.2%}")
        if opt.get('pass_at_3') is not None:
            print(f"Pass@3: {opt['pass_at_3']:.2%}")
        if opt.get('pass_at_5') is not None:
            print(f"Pass@5: {opt['pass_at_5']:.2%}")
        print(f"Problems Solved (Pass@1): {opt['passed']} / {opt['total']}")
        
        print("\n" + "-" * 70)
        print("COMPARISON (GAIN)")
        print("-" * 70)
        print(f"Pass@1 Absolute Improvement: {comp['absolute_improvement']:+.2%}")
        print(f"Pass@1 Relative Improvement: {comp['relative_improvement_percent']:+.1f}%")
        if comp.get('pass_at_3_improvement') is not None:
            print(f"Pass@3 Improvement: {comp['pass_at_3_improvement']:+.2%}")
        if comp.get('pass_at_5_improvement') is not None:
            print(f"Pass@5 Improvement: {comp['pass_at_5_improvement']:+.2%}")
        print(f"Additional Problems Solved: {comp['additional_problems_solved']:+d}")
        
        # Determine verdict
        if comp['absolute_improvement'] > 0:
            verdict = "IMPROVED ✓"
        elif comp['absolute_improvement'] < 0:
            verdict = "DEGRADED ✗"
        else:
            verdict = "NO CHANGE"
        print(f"Verdict: {verdict}")
        
        print("\n" + "-" * 70)
        print("ERROR DISTRIBUTION (Optimized Prompt):")
        for error_type, count in opt['error_distribution'].items():
            print(f"  {error_type}: {count}")
        
        print("=" * 70)
    
    def save_comparison_json(self, report: Dict, seed_results: Dict, optimized_results: Dict):
        """
        Save comparison results JSON (matching HoVer format).
        
        Args:
            report: The generated report
            seed_results: Seed prompt evaluation results
            optimized_results: Optimized prompt evaluation results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        comparison = {
            "timestamp": timestamp,
            "test_size": report['test_size'],
            "baseline": {
                "pass_at_1": seed_results['pass_at_1'],
                "pass_at_3": seed_results.get('pass_at_3'),
                "pass_at_5": seed_results.get('pass_at_5'),
                "passed": seed_results['passed'],
                "total": seed_results['total'],
            },
            "optimized": {
                "pass_at_1": optimized_results['pass_at_1'],
                "pass_at_3": optimized_results.get('pass_at_3'),
                "pass_at_5": optimized_results.get('pass_at_5'),
                "passed": optimized_results['passed'],
                "total": optimized_results['total'],
                "delta_absolute": report['comparison']['absolute_improvement'],
                "delta_percent": report['comparison']['relative_improvement_percent'],
                "additional_problems_solved": report['comparison']['additional_problems_solved'],
            },
            "summary": {
                "seed_pass_at_1": f"{seed_results['pass_at_1']:.2%}",
                "seed_pass_at_3": f"{seed_results.get('pass_at_3', 0):.2%}" if seed_results.get('pass_at_3') is not None else None,
                "seed_pass_at_5": f"{seed_results.get('pass_at_5', 0):.2%}" if seed_results.get('pass_at_5') is not None else None,
                "optimized_pass_at_1": f"{optimized_results['pass_at_1']:.2%}",
                "optimized_pass_at_3": f"{optimized_results.get('pass_at_3', 0):.2%}" if optimized_results.get('pass_at_3') is not None else None,
                "optimized_pass_at_5": f"{optimized_results.get('pass_at_5', 0):.2%}" if optimized_results.get('pass_at_5') is not None else None,
                "improvement": f"{report['comparison']['absolute_improvement']:+.2%}",
                "verdict": "IMPROVED" if report['comparison']['absolute_improvement'] > 0 
                          else ("DEGRADED" if report['comparison']['absolute_improvement'] < 0 else "NO_CHANGE")
            }
        }
        
        comparison_path = self.run_dir / f"comparison_results_{timestamp}.json"
        with open(comparison_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2)
        print(f"✓ Comparison results saved to: {comparison_path}")


# ============================================================================
# Test Evaluation Orchestrator
# ============================================================================

class TestEvaluationOrchestrator:
    """Main orchestrator for test set evaluation."""
    
    def __init__(
        self,
        run_dir: str,
        test_size: int,
        task_lm: str,
        max_workers: int,
        timeout: float,
        batch_size: int = 10
    ):
        self.run_dir = Path(run_dir)
        self.test_size = test_size
        self.task_lm = task_lm
        self.max_workers = max_workers
        self.timeout = timeout
        self.batch_size = batch_size
        
        # Initialize components
        self.checkpoint_manager = CheckpointManager(self.run_dir / "eval_checkpoints")
        self.report_generator = ReportGenerator(self.run_dir)
    
    def run(self):
        """Execute the full evaluation pipeline."""
        
        # Step 1: Load test data
        test_examples = self._load_test_data()
        
        # Step 2: Load prompts
        seed_prompt, optimized_prompt = self._load_prompts()
        
        # Step 3: Initialize adapter and evaluator
        adapter = HumanEvalAdapter(
            model=self.task_lm,
            timeout=self.timeout,
            max_litellm_workers=self.max_workers
        )
        evaluator = PromptEvaluator(adapter, self.checkpoint_manager)
        
        # Step 4: Evaluate seed prompt
        print("\n" + "=" * 70)
        print("EVALUATING SEED PROMPT")
        print("=" * 70)
        seed_results = evaluator.evaluate_with_checkpoints(
            seed_prompt, test_examples, "seed", self.batch_size
        )
        print(f"✓ Seed Prompt Pass@1: {seed_results['pass_at_1']:.2%}")
        
        # Step 4b: Evaluate Pass@3 for seed prompt
        print("\n" + "-" * 70)
        print("EVALUATING SEED PROMPT Pass@3")
        print("-" * 70)
        seed_pass3_results = evaluator.evaluate_pass_at_k(
            seed_prompt, test_examples, k=3
        )
        seed_results['pass_at_3'] = seed_pass3_results['pass_at_k']
        seed_results['pass_at_3_details'] = seed_pass3_results
        print(f"✓ Seed Prompt Pass@3: {seed_pass3_results['pass_at_k']:.2%} ({seed_pass3_results['passed']}/{seed_pass3_results['total']})")
        
        # Step 4c: Evaluate Pass@5 for seed prompt (to give it a fair chance)
        print("\n" + "-" * 70)
        print("EVALUATING SEED PROMPT Pass@5")
        print("-" * 70)
        seed_pass5_results = evaluator.evaluate_pass_at_k(
            seed_prompt, test_examples, k=5
        )
        seed_results['pass_at_5'] = seed_pass5_results['pass_at_k']
        seed_results['pass_at_5_details'] = seed_pass5_results
        print(f"✓ Seed Prompt Pass@5: {seed_pass5_results['pass_at_k']:.2%} ({seed_pass5_results['passed']}/{seed_pass5_results['total']})")
        
        # Step 5: Evaluate optimized prompt
        print("\n" + "=" * 70)
        print("EVALUATING OPTIMIZED PROMPT")
        print("=" * 70)
        optimized_results = evaluator.evaluate_with_checkpoints(
            optimized_prompt, test_examples, "optimized", self.batch_size
        )
        print(f"✓ Optimized Prompt Pass@1: {optimized_results['pass_at_1']:.2%}")
        
        # Step 5b: Evaluate Pass@3 for optimized prompt
        print("\n" + "-" * 70)
        print("EVALUATING OPTIMIZED PROMPT Pass@3")
        print("-" * 70)
        opt_pass3_results = evaluator.evaluate_pass_at_k(
            optimized_prompt, test_examples, k=3
        )
        optimized_results['pass_at_3'] = opt_pass3_results['pass_at_k']
        optimized_results['pass_at_3_details'] = opt_pass3_results
        print(f"✓ Optimized Prompt Pass@3: {opt_pass3_results['pass_at_k']:.2%} ({opt_pass3_results['passed']}/{opt_pass3_results['total']})")
        
        # Step 5c: Evaluate Pass@5 for optimized prompt
        print("\n" + "-" * 70)
        print("EVALUATING OPTIMIZED PROMPT Pass@5")
        print("-" * 70)
        opt_pass5_results = evaluator.evaluate_pass_at_k(
            optimized_prompt, test_examples, k=5
        )
        optimized_results['pass_at_5'] = opt_pass5_results['pass_at_k']
        optimized_results['pass_at_5_details'] = opt_pass5_results
        print(f"✓ Optimized Prompt Pass@5: {opt_pass5_results['pass_at_k']:.2%} ({opt_pass5_results['passed']}/{opt_pass5_results['total']})")
        
        # Step 6: Generate and save report
        print("\n" + "=" * 70)
        print("GENERATING EVALUATION REPORT")
        print("=" * 70)
        report = self.report_generator.generate_report(
            seed_results, optimized_results, len(test_examples)
        )
        self.report_generator.save_report(report, seed_results, optimized_results)
        
        # Step 7: Save comparison JSON (matching HoVer format)
        self.report_generator.save_comparison_json(report, seed_results, optimized_results)
        
        # Step 8: Print summary
        self.report_generator.print_summary(report)
    
    def _load_test_data(self) -> List[Dict]:
        """Load test data from deterministic CSV split."""
        
        # Primary: Load from deterministic test CSV
        if TEST_CSV.exists():
            print(f"Loading test set from deterministic CSV: {TEST_CSV}")
            test_examples = self._load_examples_from_csv(TEST_CSV)
            print(f"✓ Loaded {len(test_examples)} test examples from CSV")
            return test_examples[:self.test_size]
        
        # Fallback 1: Try to load from saved testset.json in run_dir
        testset_path = self.run_dir / "testset.json"
        if testset_path.exists():
            print(f"Loading test set from: {testset_path}")
            with open(testset_path, 'r', encoding='utf-8') as f:
                test_examples = json.load(f)
            print(f"✓ Loaded {len(test_examples)} test examples from saved file")
            return test_examples[:self.test_size]
        
        # Fallback 2: Load from HuggingFace (not deterministic!)
        print("WARNING: No deterministic test CSV found!")
        print("Loading HumanEval dataset from HuggingFace (results may not be reproducible)...")
        ds = load_dataset("openai/openai_humaneval")
        full_data = ds['test']
        
        print(f"Total problems available: {len(full_data)}")
        
        # Convert to GEPA format
        formatter = DataFormatter()
        all_examples = [formatter.humaneval_to_gepa_format(ex) for ex in full_data]
        
        # Shuffle and select test examples
        random.shuffle(all_examples)
        test_examples = all_examples[:self.test_size]
        
        print(f"✓ Using {len(test_examples)} test examples")
        return test_examples
    
    def _load_examples_from_csv(self, filepath: Path) -> List[Dict[str, Any]]:
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
        
        return examples
    
    def _load_prompts(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Load seed and optimized prompts."""
        print(f"\nLoading prompts from {self.run_dir}...")
        
        seed_prompt_file = self.run_dir / "seed_prompt.txt"
        optimized_prompt_file = self.run_dir / "optimized_prompt.txt"
        
        # Load seed prompt
        if not seed_prompt_file.exists():
            # Try to extract from config
            config_file = self.run_dir / "experiment_config.json"
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                seed_text = config.get('seed_prompt', {}).get('system_prompt', '')
                if seed_text:
                    with open(seed_prompt_file, 'w', encoding='utf-8') as f:
                        f.write(seed_text)
        
        if not seed_prompt_file.exists():
            raise FileNotFoundError(f"Seed prompt not found at {seed_prompt_file}")
        
        with open(seed_prompt_file, 'r', encoding='utf-8') as f:
            seed_prompt = {"system_prompt": f.read().strip()}
        
        # Load optimized prompt
        if not optimized_prompt_file.exists():
            raise FileNotFoundError(f"Optimized prompt not found at {optimized_prompt_file}")
        
        with open(optimized_prompt_file, 'r', encoding='utf-8') as f:
            optimized_prompt = {"system_prompt": f.read().strip()}
        
        print("✓ Prompts loaded successfully")
        return seed_prompt, optimized_prompt


# ============================================================================
# CLI Entry Point
# ============================================================================

def find_latest_run_dir() -> Path:
    """
    Find the latest results directory by timestamp.
    
    Returns:
        Path to the latest run directory
    """
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Results directory not found: {RESULTS_DIR}")
    
    # Find all gepa_humaneval_results_* directories
    run_dirs = list(RESULTS_DIR.glob("gepa_humaneval_results_*"))
    
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {RESULTS_DIR}")
    
    # Sort by name (timestamp) and get the latest
    run_dirs.sort(key=lambda x: x.name, reverse=True)
    latest = run_dirs[0]
    
    print(f"Found {len(run_dirs)} run directories")
    print(f"Using latest: {latest.name}")
    
    return latest


def main():
    """Main entry point."""
    
    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: OPENAI_API_KEY not found in environment variables!")
        return
    
    # Determine run directory
    run_dir = RUN_DIR
    
    # If RUN_DIR doesn't exist, try to find the latest
    if not run_dir.exists():
        print(f"Configured RUN_DIR not found: {run_dir}")
        print("Searching for latest run directory...")
        try:
            run_dir = find_latest_run_dir()
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            print("Please run train_humaneval.py first to create a results directory.")
            return
    
    print(f"\n{'=' * 70}")
    print(f"EVALUATION RUN DIRECTORY: {run_dir}")
    print(f"{'=' * 70}")
    
    # Calculate test size from percentage
    # First, count total examples in test CSV
    total_test_examples = 64  # Default
    if TEST_CSV.exists():
        with open(TEST_CSV, 'r', newline='', encoding='utf-8') as f:
            import csv as csv_module
            reader = csv_module.DictReader(f)
            total_test_examples = sum(1 for _ in reader)
    
    test_size = max(1, int(total_test_examples * TEST_PERCENTAGE / 100))
    print(f"Test set: {test_size} examples ({TEST_PERCENTAGE}% of {total_test_examples})")
    
    try:
        orchestrator = TestEvaluationOrchestrator(
            run_dir=run_dir,
            test_size=test_size,
            task_lm=TASK_LM,
            max_workers=MAX_WORKERS,
            timeout=EXECUTION_TIMEOUT,
            batch_size=BATCH_SIZE
        )
        orchestrator.run()
    except KeyboardInterrupt:
        print("\n\n⚠ Evaluation interrupted by user. Progress has been saved.")
        print("Run the same command again to resume from checkpoint.")
    except Exception as e:
        print(f"\n\n✗ Error during evaluation: {e}")
        print("Progress has been saved. Run again to resume.")
        raise


if __name__ == "__main__":
    main()
