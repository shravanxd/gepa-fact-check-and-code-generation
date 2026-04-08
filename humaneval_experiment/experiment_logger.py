"""
Experiment Logger Module
========================

Handles logging and saving experiment configurations, results, and metrics.
Follows Single Responsibility Principle - only handles logging operations.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class ExperimentLogger:
    """
    Logger for GEPA experiments.
    
    Handles saving:
    - Experiment configuration
    - Hyperparameters
    - Results and metrics
    - Prompts (seed and optimized)
    """
    
    def __init__(self, run_dir: str):
        """
        Initialize the logger.
        
        Args:
            run_dir: Directory where experiment artifacts will be saved
        """
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
    
    def save_experiment_config(
        self,
        result: Any,
        hyperparams: Dict[str, Any],
        seed_prompt: Dict[str, str]
    ) -> str:
        """
        Save experiment configuration to JSON.
        
        Args:
            result: GEPA optimization result object
            hyperparams: Dictionary of hyperparameters
            seed_prompt: The seed prompt used
            
        Returns:
            Path to saved config file
        """
        config = {
            "timestamp": datetime.now().isoformat(),
            "hyperparameters": hyperparams,
            "seed_prompt": seed_prompt,
            "results": {
                "best_score": float(result.val_aggregate_scores[result.best_idx]),
                "total_metric_calls": result.total_metric_calls,
                "num_candidates": result.num_candidates,
            },
            "best_candidate": result.best_candidate
        }
        
        config_path = self.run_dir / "experiment_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Experiment config saved to: {config_path}")
        return str(config_path)
    
    def save_prompt(self, prompt: str, filename: str) -> str:
        """
        Save a prompt to a text file.
        
        Args:
            prompt: The prompt text to save
            filename: Name of the file (e.g., "seed_prompt.txt")
            
        Returns:
            Path to saved file
        """
        filepath = self.run_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(prompt)
        
        print(f"✓ Prompt saved to: {filepath}")
        return str(filepath)
    
    def save_results(self, results: Dict[str, Any], filename: str) -> str:
        """
        Save results dictionary to JSON.
        
        Args:
            results: Results dictionary to save
            filename: Name of the file
            
        Returns:
            Path to saved file
        """
        filepath = self.run_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Results saved to: {filepath}")
        return str(filepath)
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """
        Log a metric (for future integration with tracking systems).
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step number
        """
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "name": name,
            "value": value,
            "step": step
        }
        
        # Append to metrics log file
        metrics_file = self.run_dir / "metrics.jsonl"
        with open(metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_message(self, message: str, level: str = "INFO"):
        """
        Log a message with timestamp.
        
        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] [{level}] {message}"
        
        # Print to console
        print(log_line)
        
        # Append to log file
        log_file = self.run_dir / "experiment.log"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_line + '\n')
