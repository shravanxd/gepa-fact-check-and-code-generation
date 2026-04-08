## GEPA Reflector Integration Setup

This guide walks you through generating structured reflector logs with W&B tracking.

### Prerequisites

1. API keys stored in `.env` file (already done):
   - `OPENAI_API_KEY` (rotated key, not the one exposed earlier)
   - `WANDB_API_KEY` (from https://wandb.ai/authorize)

2. Required Python packages:
   ```bash
   pip install openai python-dotenv wandb
   ```

### Quick Start

#### Step 1: Generate Reflector Logs

```bash
python generate_reflection_logs.py
```

This will:
- Scan `humaneval_experiment/results/` and `hover_experiment/results/` for test results
- Extract failures (MissingFunction, SyntaxError, etc.)
- Invoke reflector LLM (GPT-4) for each failure
- Save structured logs to `reflector_logs/` directory
- Upload metrics to wandb.ai

#### Step 2: Review Generated Logs

Logs are stored as JSONL (JSON Lines) files:
```
reflector_logs/humaneval_reflection_batch_001_20260407_143022.jsonl
reflector_logs/hover_reflection_batch_001_20260407_143025.jsonl
```

Each line is a JSON object with:
```json
{
  "timestamp": "2026-04-07T14:30:22.123456",
  "event_type": "reflector_invocation",
  "data": {
    "task_id": 29,
    "entry_point": "filter_by_prefix",
    "error_type": "MissingFunction",
    "reflector_output": "MUTATION_ANALYSIS: Function name not specified...",
    "model_used": "gpt-4",
    "usage": {"prompt_tokens": 187, "completion_tokens": 145, "total_tokens": 332}
  }
}
```

#### Step 3: View in W&B Dashboard

After running, visit: https://wandb.ai/

Your project: `gepa-reflector`
Runs: `humaneval_reflection_run`, `hover_reflection_run`

You can see:
- Token usage per task
- Error type distribution
- Reflector response content
- Run duration and timestamps

### Log Format Detail

**No AI Patterns Included:**
- Timestamps: ISO 8601 format (YYYY-MM-DDTHH:MM:SS.ffffff)
- Error messages: Verbatim from execution without summarization
- Reflector responses: Raw text, no special formatting
- Event types: Simple strings (batch_start, reflector_invocation, batch_complete)

**Structured Fields (JSONL):**
```
timestamp         | Exact time of event
event_type        | Categorical: batch_start, reflector_invocation, batch_complete
task_id           | Integer problem ID
entry_point       | Function name string
error_type        | Error category (MissingFunction, SyntaxError, NameError, etc.)
error_message     | Full error text from execution
reflector_output  | GPT-4 response with MUTATION_ANALYSIS and SUGGESTED_PROMPT_CHANGE
usage.total_tokens| OpenAI API token count (for cost tracking)
```

### Integration with GEPA Training Loop

To invoke reflector automatically during training, modify `humaneval_adapter.py`:

```python
from reflector_invoker import invoke_reflector

# In make_reflective_dataset():
for failure in failures:
    reflector_result = invoke_reflector(
        task_id=failure["task_id"],
        entry_point=failure["entry_point"],
        error_type=failure["error_type"],
        error_message=failure["error_message"],
        guidance_signal=failure_guidance
    )
    # Use reflector_result["reflector_output"] for prompt mutation
```

### Troubleshooting

**"OPENAI_API_KEY not found"**
- Verify `.env` file exists
- Check format: `OPENAI_API_KEY=sk-proj-...`
- No quotes or extra spaces

**"WANDB_API_KEY not found"**
- W&B logging is optional; scripts will continue without it
- To enable: get key from https://wandb.ai/authorize and add to `.env`

**"Rate limited by OpenAI"**
- Default: 5 tasks per batch
- Modify `sample_size` parameter in `generate_reflection_logs.py` to reduce calls
- Add sleep between invocations if needed

**View old logs:**
```bash
ls -lah reflector_logs/
cat reflector_logs/humaneval_reflection_batch_001*.jsonl | head -20
```

### Example Output

```
TASK 29 (filter_by_prefix)
ERROR TYPE: MissingFunction
---
REFLECTOR RESPONSE:
MUTATION_ANALYSIS: Model generated only the function body without the def line. Roots error.
SUGGESTED_PROMPT_CHANGE: Add explicit instruction: "Your output must be a complete function definition starting with def entry_point_name(...). Do not output partial code or only the function body."
PRIORITY_LEVEL: HIGH
```

### Next Steps

1. Run logs and confirm W&B upload succeeds
2. Update Example C in `gepa_sample_trace.md` with actual reflector output
3. Archive logs to version control if needed

```bash
git add reflector_logs/
git commit -m "Add reflector logs from GEPA runs"
```
