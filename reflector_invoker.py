import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
from dotenv import load_dotenv
import wandb
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class StructuredLogger:
    def __init__(self, run_name: str, log_dir: str = "reflector_logs"):
        self.run_name = run_name
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
        self.entries = []

    def log_event(self, event_type: str, data: Dict[str, Any]):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        }
        self.entries.append(entry)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info(f"Logged {event_type}")

    def get_entries(self) -> List[Dict]:
        return self.entries


def invoke_reflector(
    task_id: int,
    entry_point: str,
    error_type: str,
    error_message: str,
    traceback_text: str = None,
    guidance_signal: str = None,
    model: str = "gpt-4"
) -> Dict[str, Any]:
    
    reflection_prompt = f"""You are a prompt optimization reflector for a GEPA (Evolutionary Prompt Adaptation) system.

Task Context:
- Task ID: {task_id}
- Function/Entry Point: {entry_point}
- Error Type: {error_type}
- Error Message: {error_message}

Failure Signal Details:
{f"Traceback: {traceback_text}" if traceback_text else ""}
{f"Guidance: {guidance_signal}" if guidance_signal else ""}

Your job: Suggest a concise, actionable modification to the system prompt that would prevent this error in the next iteration.

Output format (plain text, no formatting):
MUTATION_ANALYSIS: [brief analysis of root cause]
SUGGESTED_PROMPT_CHANGE: [specific instruction to add or modify in the system prompt]
PRIORITY_LEVEL: [HIGH/MEDIUM/LOW based on error frequency impact]"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": reflection_prompt}],
        temperature=0.7,
        max_tokens=500
    )
    
    reflector_output = response.choices[0].message.content
    
    result = {
        "task_id": task_id,
        "entry_point": entry_point,
        "error_type": error_type,
        "error_message": error_message,
        "reflector_output": reflector_output,
        "model_used": model,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
    }
    
    return result


def run_reflection_batch(
    failures: List[Dict[str, Any]],
    run_name: str,
    use_wandb: bool = True
) -> List[Dict[str, Any]]:
    
    if use_wandb and WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY)
        wandb.init(project="gepa-reflector", name=run_name, config={"model": "gpt-4"})
    
    logger_instance = StructuredLogger(run_name=run_name)
    results = []
    
    logger.info(f"Starting reflection batch: {run_name} with {len(failures)} failures")
    logger_instance.log_event(
        "batch_start",
        {"run_name": run_name, "num_failures": len(failures)}
    )
    
    for i, failure in enumerate(failures):
        logger.info(f"Processing failure {i+1}/{len(failures)}: Task {failure.get('task_id')}")
        
        result = invoke_reflector(
            task_id=failure.get("task_id"),
            entry_point=failure.get("entry_point"),
            error_type=failure.get("error_type"),
            error_message=failure.get("error_message"),
            traceback_text=failure.get("traceback"),
            guidance_signal=failure.get("guidance")
        )
        
        results.append(result)
        logger_instance.log_event("reflector_invocation", result)
        
        if use_wandb and WANDB_API_KEY:
            wandb.log({
                "task_id": result["task_id"],
                "error_type": result["error_type"],
                "tokens_used": result["usage"]["total_tokens"],
                "batch_index": i
            })
        
        logger.info(f"Reflector response received for Task {result['task_id']}")
    
    logger_instance.log_event(
        "batch_complete",
        {"run_name": run_name, "num_processed": len(results)}
    )
    
    if use_wandb and WANDB_API_KEY:
        wandb.finish()
    
    logger.info(f"Reflection batch complete. Results saved to {logger_instance.log_file}")
    return results, logger_instance.log_file


if __name__ == "__main__":
    sample_failures = [
        {
            "task_id": 29,
            "entry_point": "filter_by_prefix",
            "error_type": "MissingFunction",
            "error_message": "Function 'filter_by_prefix' not found in generated code",
            "guidance": "The prompt should clearly specify the function name and require a complete function definition."
        },
        {
            "task_id": 15,
            "entry_point": "string_sequence",
            "error_type": "SyntaxError",
            "error_message": "invalid syntax (<string>, line 3)",
            "guidance": "Model output had invalid Python syntax. Prompt should emphasize syntactic correctness."
        }
    ]
    
    results, log_file = run_reflection_batch(
        failures=sample_failures,
        run_name="humaneval_reflection_batch_001",
        use_wandb=True
    )
    
    for result in results:
        print("\n" + "="*80)
        print(f"TASK {result['task_id']} ({result['entry_point']})")
        print(f"ERROR TYPE: {result['error_type']}")
        print("-"*80)
        print("REFLECTOR RESPONSE:")
        print(result["reflector_output"])
        print("="*80)
