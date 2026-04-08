import json
import os
from pathlib import Path
from typing import Dict, List, Any
from reflector_invoker import run_reflection_batch, logger


def extract_failures_from_test_results(
    test_results_file: str,
    limit: int = None
) -> List[Dict[str, Any]]:
    
    if not os.path.exists(test_results_file):
        logger.error(f"Test results file not found: {test_results_file}")
        return []
    
    failures = []
    
    with open(test_results_file, "r") as f:
        test_results = json.load(f)
    
    results_list = []
    if isinstance(test_results, dict):
        for key in test_results:
            if isinstance(test_results[key], list):
                results_list.extend(test_results[key])
    elif isinstance(test_results, list):
        results_list = test_results
    
    for result in results_list:
        if isinstance(result, dict) and (result.get("status") == "FAILED" or result.get("passed") is False):
            task_id = result.get("task_id", "unknown")
            failure_entry = {
                "task_id": task_id,
                "entry_point": result.get("entry_point", "unknown"),
                "error_type": result.get("error_type", "UNKNOWN"),
                "error_message": result.get("error_message", "No error message"),
                "traceback": result.get("traceback", ""),
                "guidance": f"Failed on problem {task_id}"
            }
            failures.append(failure_entry)
            
            if limit and len(failures) >= limit:
                break
    
    logger.info(f"Extracted {len(failures)} failures from {test_results_file}")
    return failures


def process_experiment_run(
    experiment_dir: str,
    run_name: str,
    use_wandb: bool = True,
    sample_size: int = 10
) -> Dict[str, Any]:
    
    experiment_path = Path(experiment_dir)
    
    if not experiment_path.exists():
        logger.error(f"Experiment directory not found: {experiment_dir}")
        return {}
    
    test_results_files = list(experiment_path.glob("test_detailed_results_*.json"))
    
    if not test_results_files:
        logger.warning(f"No test_detailed_results files found in {experiment_dir}")
        return {}
    
    test_results_file = sorted(test_results_files)[-1]
    logger.info(f"Using test results file: {test_results_file}")
    
    failures = extract_failures_from_test_results(
        str(test_results_file),
        limit=sample_size
    )
    
    if not failures:
        logger.info("No failures to process")
        return {"run_name": run_name, "failures_found": 0}
    
    logger.info(f"Processing {len(failures)} failures with reflector")
    
    results, log_file = run_reflection_batch(
        failures=failures,
        run_name=run_name,
        use_wandb=use_wandb
    )
    
    summary = {
        "run_name": run_name,
        "experiment_dir": experiment_dir,
        "failures_found": len(failures),
        "reflections_generated": len(results),
        "log_file": log_file,
        "reflections": []
    }
    
    for result in results:
        summary["reflections"].append({
            "task_id": result["task_id"],
            "error_type": result["error_type"],
            "mutation_suggested": result["reflector_output"].split("SUGGESTED_PROMPT_CHANGE:")[-1].strip() if "SUGGESTED_PROMPT_CHANGE:" in result["reflector_output"] else "See log"
        })
    
    return summary


def main():
    humaneval_experiment = "humaneval_experiment/results/gepa_humaneval_results_20251203_084234"
    hover_experiment = "hover_experiment/results/gepa_hover_results_20251109_003303"
    
    logger.info("Starting GEPA Reflector Log Generation")
    
    humaneval_summary = process_experiment_run(
        experiment_dir=humaneval_experiment,
        run_name="humaneval_reflection_run",
        use_wandb=True,
        sample_size=5
    )
    
    hover_summary = process_experiment_run(
        experiment_dir=hover_experiment,
        run_name="hover_reflection_run",
        use_wandb=True,
        sample_size=5
    )
    
    print("\n" + "="*80)
    print("REFLECTION PROCESSING COMPLETE")
    print("="*80)
    print(f"HumanEval: {humaneval_summary['failures_found']} failures processed")
    if humaneval_summary.get("reflections"):
        print(f"Log saved to: {humaneval_summary['log_file']}")
    print()
    print(f"HoVer: {hover_summary['failures_found']} failures processed")
    if hover_summary.get("reflections"):
        print(f"Log saved to: {hover_summary['log_file']}")
    print("="*80)
    print("\nCheck Weights & Biases dashboard for detailed metrics:")
    print("https://wandb.ai/")


if __name__ == "__main__":
    main()
